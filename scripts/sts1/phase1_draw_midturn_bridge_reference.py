"""Fail-closed stdin/stdout bridge for STS1 CommunicationMod.

CommunicationMod owns the game process and launches this Python process. It
sends one JSON object per line on stdin and expects one command per line on
stdout. This module intentionally keeps that protocol tiny so the validation
track does not depend on the older ``spirecomm`` package at runtime.

The bridge also keeps a tiny *derived* turn trace for public reconstruction.
It records only counters for commands this controller actually sent and never
stores draw order, RNG state, card UUIDs, or a replayable action history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, TextIO


RECONSTRUCTION_AUX_SCHEMA = "sts1-public-reconstruction-aux-v1"
RECONSTRUCTION_AUX_SOURCE = "communicationmod_command_trace_v1"

# V1 is deliberately narrow. Cards outside this set still play normally, but
# the reconstruction trace becomes incomplete so Phase 1 must fail closed.
# Pommel Strike and Shrug It Off are the first audited draw-card expansions;
# neither auto-plays nor discards another card.
_TRACE_SAFE_CARD_IDS = frozenset(
    {
        "STRIKE_R",
        "STRIKE_RED",
        "DEFEND_R",
        "DEFEND_RED",
        "BASH",
        "POMMEL_STRIKE",
        "SHRUG_IT_OFF",
    }
)


class BridgeProtocolError(RuntimeError):
    """CommunicationMod input or command sequencing violated the contract."""


class RemoteCommandError(BridgeProtocolError):
    """CommunicationMod reported an error for the previous command."""


@dataclass(frozen=True)
class CommunicationState:
    """One newline-delimited state message emitted by CommunicationMod."""

    available_commands: tuple[str, ...]
    ready_for_command: bool
    in_game: bool
    game_state: Mapping[str, Any] | None
    error: str | None
    raw: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CommunicationState":
        commands_raw = payload.get("available_commands", ())
        if commands_raw is None:
            commands_raw = ()
        if not isinstance(commands_raw, list | tuple) or not all(isinstance(item, str) for item in commands_raw):
            raise BridgeProtocolError("available_commands_must_be_string_sequence")

        ready = payload.get("ready_for_command", False)
        in_game = payload.get("in_game", False)
        if not isinstance(ready, bool):
            raise BridgeProtocolError("ready_for_command_must_be_bool")
        if not isinstance(in_game, bool):
            raise BridgeProtocolError("in_game_must_be_bool")

        game_state = payload.get("game_state")
        if game_state is not None and not isinstance(game_state, Mapping):
            raise BridgeProtocolError("game_state_must_be_mapping_or_null")

        error_raw = payload.get("error")
        error = None if error_raw is None else str(error_raw)
        return cls(
            available_commands=tuple(commands_raw),
            ready_for_command=ready,
            in_game=in_game,
            game_state=game_state,
            error=error,
            raw=dict(payload),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    return []


def _normalize_card_id(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text.replace(" ", "_").replace("-", "_")


def _combat_turn(state: CommunicationState | None) -> int | None:
    if state is None or not state.in_game:
        return None
    game = _mapping(state.game_state)
    if str(game.get("room_phase") or "").upper() != "COMBAT":
        return None
    combat = _mapping(game.get("combat_state"))
    turn = combat.get("turn")
    if isinstance(turn, bool) or not isinstance(turn, int):
        return None
    return turn


class CommunicationBridge:
    """Safe command sequencer for CommunicationMod.

    A gameplay command may be sent only after a fresh inbound state says the
    game is ready. After one command is sent, another command is refused until
    another state message arrives. This prevents duplicate/stale actions even
    if caller code accidentally invokes the policy twice.

    The reconstruction trace is intentionally not policy state. It is only a
    fail-closed bookkeeping aid for rebuilding simulator internals that are not
    inferable from one snapshot after draw effects.
    """

    def __init__(self, input_stream: TextIO, output_stream: TextIO) -> None:
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.last_state: CommunicationState | None = None
        self.message_sequence = 0
        self.command_sequence = 0
        self.remote_error_count = 0
        self._waiting_for_update = False
        self._ready_signaled = False

        self._trace_turn: int | None = None
        self._trace_complete = False
        self._trace_cards_played = 0
        self._trace_attacks_played = 0
        self._trace_skills_played = 0
        self._pending_play: tuple[str | None, bool] | None = None
        self._pending_invalidates_trace = False

    @property
    def waiting_for_update(self) -> bool:
        return self._waiting_for_update

    @property
    def reconstruction_aux(self) -> dict[str, Any]:
        """Return bounded reconstruction counters, never a hidden/action-history feature."""

        active = self._trace_turn is not None
        return {
            "schema_version": RECONSTRUCTION_AUX_SCHEMA,
            "source": RECONSTRUCTION_AUX_SOURCE,
            "turn": self._trace_turn,
            "complete": bool(active and self._trace_complete and not self._waiting_for_update),
            "cards_played_this_turn": self._trace_cards_played if active else 0,
            "attacks_played_this_turn": self._trace_attacks_played if active else 0,
            "skills_played_this_turn": self._trace_skills_played if active else 0,
            "cards_discarded_this_turn": 0,
        }

    def signal_ready(self) -> None:
        if self._ready_signaled:
            raise BridgeProtocolError("ready_already_signaled")
        self._write_line("ready")
        self._ready_signaled = True

    def read_state(self) -> CommunicationState:
        line = self.input_stream.readline()
        if line == "":
            raise EOFError("communicationmod_closed_input")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeProtocolError("invalid_json_state") from exc
        if not isinstance(payload, Mapping):
            raise BridgeProtocolError("state_payload_must_be_mapping")

        state = CommunicationState.from_payload(payload)
        self._advance_reconstruction_trace(state)
        self.last_state = state
        self.message_sequence += 1
        self._waiting_for_update = False
        if state.error is not None:
            self.remote_error_count += 1
        return state

    def wait_for_decision(self, *, max_messages: int = 100) -> CommunicationState:
        if max_messages < 1:
            raise ValueError("max_messages_must_be_positive")
        for _ in range(max_messages):
            state = self.read_state()
            if state.error is not None:
                raise RemoteCommandError(state.error)
            if state.ready_for_command:
                return state
        raise BridgeProtocolError("decision_state_not_reached_within_message_bound")

    def send_command(self, command: str) -> None:
        if not isinstance(command, str) or not command.strip():
            raise BridgeProtocolError("command_must_be_nonempty_string")
        if self.last_state is None:
            raise BridgeProtocolError("no_state_received")
        if self.last_state.error is not None:
            raise BridgeProtocolError("cannot_send_after_remote_error")
        if self._waiting_for_update:
            raise BridgeProtocolError("stale_or_duplicate_action_blocked")
        if not self.last_state.ready_for_command:
            raise BridgeProtocolError("game_not_ready_for_command")

        normalized_command = command.strip()
        verb = normalized_command.split(maxsplit=1)[0].lower()
        allowed = {item.lower() for item in self.last_state.available_commands}
        if verb not in allowed:
            raise BridgeProtocolError(f"command_not_advertised:{verb}")

        self._stage_reconstruction_effect(normalized_command)
        self._write_line(normalized_command)
        self.command_sequence += 1
        self._waiting_for_update = True

    def _stage_reconstruction_effect(self, command: str) -> None:
        self._pending_play = None
        self._pending_invalidates_trace = False
        if _combat_turn(self.last_state) is None:
            return

        parts = command.split()
        verb = parts[0].lower()
        if verb == "end":
            return
        if verb != "play":
            self._pending_invalidates_trace = True
            return

        card_type: str | None = None
        safe = False
        try:
            hand_index = int(parts[1])
        except (IndexError, ValueError):
            self._pending_play = (None, False)
            return

        game = _mapping(self.last_state.game_state if self.last_state is not None else None)
        combat = _mapping(game.get("combat_state"))
        hand = _sequence(combat.get("hand"))
        if hand_index < 1 or hand_index > len(hand):
            self._pending_play = (None, False)
            return

        card = _mapping(hand[hand_index - 1])
        raw_type = card.get("type")
        if isinstance(raw_type, str):
            normalized_type = raw_type.strip().upper()
            if normalized_type in {"ATTACK", "SKILL", "POWER"}:
                card_type = normalized_type

        card_id = _normalize_card_id(card.get("id") or card.get("name"))
        safe = card_type is not None and card_id in _TRACE_SAFE_CARD_IDS
        self._pending_play = (card_type, safe)

    def _advance_reconstruction_trace(self, state: CommunicationState) -> None:
        previous_turn = self._trace_turn
        current_turn = _combat_turn(state)

        if self._pending_play is not None:
            card_type, safe = self._pending_play
            if state.error is None and previous_turn is not None and current_turn == previous_turn:
                self._trace_cards_played += 1
                if card_type == "ATTACK":
                    self._trace_attacks_played += 1
                elif card_type == "SKILL":
                    self._trace_skills_played += 1
                if not safe:
                    self._trace_complete = False
            self._pending_play = None

        if self._pending_invalidates_trace:
            if state.error is None and previous_turn is not None and current_turn == previous_turn:
                self._trace_complete = False
            self._pending_invalidates_trace = False

        if current_turn is None:
            self._reset_trace(None, complete=False)
            return

        if previous_turn is None:
            complete = self.last_state is not None and _combat_turn(self.last_state) is None
            self._reset_trace(current_turn, complete=complete)
            return

        if current_turn != previous_turn:
            self._reset_trace(current_turn, complete=True)

    def _reset_trace(self, turn: int | None, *, complete: bool) -> None:
        self._trace_turn = turn
        self._trace_complete = bool(complete and turn is not None)
        self._trace_cards_played = 0
        self._trace_attacks_played = 0
        self._trace_skills_played = 0
        self._pending_play = None
        self._pending_invalidates_trace = False

    def _write_line(self, message: str) -> None:
        self.output_stream.write(message + "\n")
        self.output_stream.flush()


__all__ = [
    "BridgeProtocolError",
    "CommunicationBridge",
    "CommunicationState",
    "RECONSTRUCTION_AUX_SCHEMA",
    "RECONSTRUCTION_AUX_SOURCE",
    "RemoteCommandError",
]

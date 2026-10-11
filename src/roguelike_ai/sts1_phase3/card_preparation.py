"""Read-only card facts and Train diagnostics; no readiness or policy claims."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import re

UNKNOWN = "NOT_VERIFIED"
SOURCE_COMMIT = "7476a81954020087da31d41d16fddf475746ec2d"
SOURCE_HASHES = {
    "include/constants/Cards.h": "a963ece62d7e95708f82385d3618519b7c9e338c96f086dfa123530c3ae7506e",
    "src/combat/BattleContext.cpp": "0dcc8b06c23d361613fa10073e458d66043430d990a3c7d51c341e9ca76ff8cf",
    "src/combat/CardInstance.cpp": "125618b3ea788ddb916d8c273745ccf91d41dc12a62abf95ceb417454df3672d",
    "src/combat/Actions.cpp": "0cd49b95416ace90acfa899c94a12204d3f0104626736ea409028a66568f05b3",
    "src/combat/Player.cpp": "cb7b8abba9db74164da71d29b91cf8fc1160fc815d6292906cfff5c9a2c689c7",
}

# Positive, reviewed effects only. Missing effects are unknown, not absent.
# Values are base/upgraded; damage/block are pre-modifier amounts, not outcomes.
REVIEWED = {
    "Clash": {"damage": (14, 18), "condition": "all_cards_in_hand_are_attacks"},
    "Anger": {"damage": (6, 8), "condition": "creates_copy_in_discard"},
    "Wild Strike": {"damage": (12, 17), "condition": "adds_wound_to_draw_pile"},
    "Bash": {"damage": (8, 10), "vulnerable": (2, 3)},
    "Pommel Strike": {"damage": (9, 10), "draw": (1, 2)},
    "Shrug It Off": {"block": (8, 11), "draw": (1, 1)},
    "Battle Trance": {"draw": (3, 4), "condition": "no_more_draw_this_turn"},
    "Offering": {"draw": (3, 5), "energy": (2, 2), "hp_cost": (6, 6)},
    "Dropkick": {"damage": (5, 8), "conditional_draw": (1, 1),
                 "conditional_energy": (1, 1), "condition": "target_is_targetable_and_vulnerable"},
    "Inflame": {"persistent_strength": (2, 3)},
    "Demon Form": {"strength_per_turn": (2, 3)},
    "Spot Weakness": {"conditional_strength": (3, 4), "condition": "target_is_attacking"},
    "Flex": {"temporary_strength": (2, 4), "condition": "applies_lose_strength_debuff_may_be_blocked"},
    "Seeing Red": {"energy": (2, 2)},
    "Dark Embrace": {"conditional_draw_per_exhaust": (1, 1), "condition": "requires_exhaust_event"},
    "Feel No Pain": {"conditional_block_per_exhaust": (3, 4), "condition": "requires_exhaust_event"},
    "Metallicize": {"block_at_end_of_turn": (3, 4)},
    "Armaments": {"block": (5, 5), "condition": "upgrade_one_hand_card_or_all_when_upgraded"},
    "Flame Barrier": {"block": (12, 16), "retaliation": (4, 6)},
    "Impervious": {"block": (30, 40)},
    "Power Through": {"block": (15, 20), "condition": "adds_two_wounds_to_hand"},
    "True Grit": {"block": (7, 9), "condition": "random_hand_exhaust_or_chosen_when_upgraded"},
    "Second Wind": {"conditional_block_per_card": (5, 7), "condition": "exhaust_nonattacks_in_hand"},
}


def base_name(name):
    """Only the native display upgrade suffix is normalized; never vocab IDs."""
    if not isinstance(name, str) or not name or name.strip() != name:
        return None
    return re.sub(r"\+(?:\d+)?$", "", name)


def parse_native_types(text):
    def array(name):
        match = re.search(r"\b" + name + r"\[\]\s*=\s*\{([^}]*)\}", text)
        if match is None:
            raise ValueError("missing native array")
        return match.group(1)
    names = re.findall(r'"([^"\n]*)"', array("cardNames"))
    types = re.findall(r"CardType::(\w+)", array("cardTypes"))
    colors = re.findall(r"CardColor::(\w+)", array("cardColors"))
    if not names or len(names) != len(types) or len(names) != len(colors):
        raise ValueError("native card arrays misaligned")
    catalog = {}
    for name, kind in zip(names, types):
        # Card type is useful for foreign/generated cards too. Never infer it
        # from color or filter out a card already present in the observed deck.
        if kind == "INVALID":
            continue
        if kind not in {"ATTACK", "SKILL", "POWER", "CURSE", "STATUS"}:
            raise ValueError("unknown native card type")
        if name in catalog and catalog[name] != kind:
            raise ValueError("ambiguous native card name")
        catalog[name] = kind
    if catalog.get("Clash") != "ATTACK" or catalog.get("Defend") != "SKILL":
        raise ValueError("Ironclad type table unavailable")
    return catalog


def load_native_types(root):
    root = Path(root)
    for relative, expected in SOURCE_HASHES.items():
        if sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("reviewed native source drift")
    return parse_native_types((root / "include/constants/Cards.h").read_text(encoding="utf-8"))


def card_facts(card, types):
    name, upgrades = card.get("name"), card.get("upgrades")
    base = base_name(name)
    valid_upgrade = type(upgrades) is int and upgrades >= 0
    # A display suffix inconsistent with the upgrade counter is not trusted.
    suffix = re.search(r"\+(\d*)$", name) if isinstance(name, str) else None
    consistent = (valid_upgrade and (int(suffix[1] or "1") == upgrades if suffix else upgrades == 0))
    facts = {"name": base, "type": types.get(base, UNKNOWN), "effects": UNKNOWN,
             "complete_effect_catalog": False, "readiness": UNKNOWN}
    if base in REVIEWED and consistent and upgrades in {0, 1}:
        facts["effects"] = {k: (v[upgrades] if isinstance(v, tuple) else v)
                            for k, v in REVIEWED[base].items()}
    return facts


def deck_facts(deck, types):
    if not isinstance(deck, list) or any(not isinstance(c, dict) for c in deck):
        raise ValueError("deck list required")
    cards = [card_facts(c, types) for c in deck]
    effect_counts = Counter()
    for card in cards:
        if isinstance(card["effects"], dict):
            effect_counts.update(k for k in card["effects"] if k != "condition")
    return {"size": len(cards), "type_counts": dict(sorted(Counter(c["type"] for c in cards).items())),
            "reviewed_positive_effect_card_counts": dict(sorted(effect_counts.items())),
            "unreviewed_effect_cards": sum(c["effects"] == UNKNOWN for c in cards),
            "readiness": UNKNOWN}


def shadow_clash_skip(decision, types):
    """Unexecuted hypothesis: skip recommended Clash in a fully typed mixed deck."""
    recommended = decision["selected_index"]
    choices = decision["choices"]
    if decision["screen"] != "REWARDS":
        return recommended, "not_reward"
    if (type(recommended) is not int or not 0 <= recommended < len(choices)
            or not isinstance(choices[recommended], dict)):
        return recommended, "incomplete_recommendation"
    choice = choices[recommended]
    if choice.get("kind") != "card" or choice.get("choice") != "card" or base_name(choice.get("card_name")) != "Clash":
        return recommended, "not_clash"
    deck = decision["before"].get("deck")
    if not isinstance(deck, list) or not deck or any(not isinstance(c, dict) for c in deck):
        return recommended, "unknown_deck"
    kinds = [types.get(base_name(c.get("name")), UNKNOWN) for c in deck]
    if UNKNOWN in kinds:
        return recommended, "unknown_card_type"
    if all(k == "ATTACK" for k in kinds):
        return recommended, "attack_only_deck"
    skips = [i for i, c in enumerate(choices) if isinstance(c, dict)
             and c == {"kind": "card", "choice": "skip"}]
    if len(skips) != 1:
        return recommended, "ambiguous_or_missing_skip"
    return skips[0], "shadow_clash_mixed_deck"


def analyze_profiles(profiles, types):
    """Prospective behavior is NOT simulated by substituting these indexes."""
    rewards, outcomes, unknown, duplicates, shadow, shadow_runs = Counter(), Counter(), Counter(), 0, 0, 0
    outcomes.update({"card": 0, "skip": 0, "empty_reward": 0})
    act_rewards, selected_cards, deck_types, effects = Counter(), Counter(), Counter(), Counter()
    for profile in profiles:
        if profile.get("schema_version") != "sts1-boss-preparation-diagnostic-v1":
            raise ValueError("preparation schema required")
        hits = 0
        for d in profile["decision_records"]:
            if d["screen"] != "REWARDS":
                continue
            choices, index = d["choices"], d["selected_index"]
            if choices == [] and index == -1:
                outcomes["empty_reward"] += 1
                continue
            if (type(index) is not int or not 0 <= index < len(choices)
                    or d["selected"] != choices[index]):
                raise ValueError("reward choice mismatch")
            selected = choices[index]
            if not isinstance(selected, dict) or selected.get("kind") != "card" or selected.get("choice") not in {"card", "skip"}:
                raise ValueError("reward semantics incomplete")
            facts = deck_facts(d["before"]["deck"], types)
            outcomes[selected["choice"]] += 1
            act_rewards[str(d["act"])] += 1
            rewards["decisions_with_unique_skip"] += sum(c == {"kind": "card", "choice": "skip"} for c in choices) == 1
            if selected["choice"] == "card":
                name = base_name(selected.get("card_name"))
                known = name in types
                selected_cards[name if known else UNKNOWN] += 1
                unknown["selected_card_identity"] += not known
                if known:
                    duplicates += any(base_name(c.get("name")) == name for c in d["before"]["deck"])
            unknown["reward_decks_with_unknown_type"] += facts["type_counts"].get(UNKNOWN, 0) > 0
            proposed, reason = shadow_clash_skip(d, types)
            if proposed != index:
                shadow += 1
                hits += 1
        shadow_runs += hits > 0
        for e in profile["encounter_records"]:
            if e["act"] == 3 and e["observed_boss_at_entry"] is not None:
                facts = deck_facts(e["resources"]["deck"], types)
                deck_types.update(facts["type_counts"])
                effects.update(facts["reviewed_positive_effect_card_counts"])
                unknown["act3_unreviewed_effect_cards"] += facts["unreviewed_effect_cards"]
    return {"schema_version": "sts1-card-preparation-analysis-v1", "policy_changed": False,
            "profiles": len(profiles), "reward_outcomes": dict(outcomes), "rewards_by_act": dict(act_rewards),
            "reward_coverage": dict(rewards), "selected_card_counts": dict(sorted(selected_cards.items())),
            "selected_known_duplicates": duplicates, "unknowns": dict(unknown),
            "act3_matched_deck_type_totals": dict(deck_types),
            "act3_reviewed_positive_effect_totals": dict(effects),
            "shadow_clash_skip_opportunities": shadow, "shadow_clash_runs": shadow_runs,
            "shadow_is_counterfactual_rollout": False, "readiness": UNKNOWN,
            "win_rate_improvement": UNKNOWN}

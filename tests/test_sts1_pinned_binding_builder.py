from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.sts1 import build_sts1_pinned_binding as builder


def test_pinned_source_hash_requires_exact_checkout_bytes(tmp_path):
    source = b"first line\nsecond line\n"
    lf_path = tmp_path / "source-lf.cpp"
    crlf_path = tmp_path / "source-crlf.cpp"
    changed_path = tmp_path / "source-changed.cpp"
    bare_cr_path = tmp_path / "source-bare-cr.cpp"
    lf_path.write_bytes(source)
    crlf_path.write_bytes(source.replace(b"\n", b"\r\n"))
    changed_path.write_bytes(source.replace(b"second", b"changed"))
    bare_cr_path.write_bytes(b"first\rsecond\n")

    expected = hashlib.sha256(source).hexdigest()
    assert builder._sha256(lf_path) == expected
    assert builder._sha256(crlf_path) != expected
    assert builder._sha256(changed_path) != expected
    assert builder._sha256(bare_cr_path) != expected


def test_pinned_hashes_match_the_registered_raw_source_blobs():
    assert builder.BASE_BINDING_SHA256 == (
        "2b7220cb976205ee094143915825b3f004f2d3d69338e07dbaf27a56c8d45169"
    )
    assert builder.GAME_CONTEXT_SHA256 == (
        "b78fd16448b96370fd78740a14baa00b4f6811c820dc544dd5c38cba01f73133"
    )
    assert builder.POTIONS_HEADER_SHA256 == (
        "998ce625b620c1922222e5b1b7fdc904003d3cb8b765b783f8d9a1688dc7b134"
    )
    assert builder.POTION_PATCH_SHA256 == (
        "fe75bb6fee0f83689bd5ab541a14ffead4f0df6b5b855c4f30096d308a4f9c7b"
    )
    patch = Path(builder.__file__).resolve().parents[2] / builder.PATCH_RELATIVE_PATH
    assert builder._sha256(patch) == builder.POTION_PATCH_SHA256


def test_compile_compatibility_preserves_reference_flags_and_cache():
    cache = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG -DUSER_FLAG=1"}
    original = dict(cache)
    assert builder._compatibility_cxx_flags(cache) == (
        "CMAKE_CXX_FLAGS_RELEASE", "-O3 -DNDEBUG -DUSER_FLAG=1 -include algorithm"
    )
    assert cache == original
    assert builder._compatibility_cxx_flags({"CMAKE_BUILD_TYPE": "Debug", "CMAKE_CXX_FLAGS_DEBUG": "-g"}) == (
        "CMAKE_CXX_FLAGS_DEBUG", "-g -include algorithm"
    )
    with pytest.raises(builder.BuildInputError, match=r"configuration C\+\+ flags are missing"):
        builder._compatibility_cxx_flags({})


def test_local_clone_binding_is_added_once_to_the_scratch_source(tmp_path):
    source_root = tmp_path / "simulator"
    binding = source_root / "bindings" / "slaythespire.cpp"
    binding.parent.mkdir(parents=True)
    binding.write_text(
        'pybind11::class_<GameContext> gameContext(m, "GameContext");\n'
        '    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())\n'
        '        .def("pick_reward_card", &sts::py::pickRewardCard, "pick")\n',
        encoding="utf-8",
        newline="\n",
    )

    digest = builder._add_local_clone_binding(source_root)
    content = binding.read_text(encoding="utf-8")
    assert content.count('.def("clone", [](const GameContext &gc)') == 1
    assert digest == builder._sha256(binding)
    with pytest.raises(builder.BuildInputError, match="already present"):
        builder._add_local_clone_binding(source_root)


def test_live_combat_binding_uses_battle_slots_readonly_and_rejects_reapplication(tmp_path):
    source = tmp_path / "source"
    binding = source / "bindings/slaythespire.cpp"
    binding.parent.mkdir(parents=True)
    anchor = '        .def_readonly("player", &BattleContext::player)\n'
    binding.write_text(anchor, encoding="utf-8")
    digest = builder._add_local_combat_potion_binding(source)
    text = binding.read_text(encoding="utf-8")
    assert '.def_property_readonly("combat_potions"' in text
    assert "bc.potions" in text and "gc.potions" not in text
    assert "index < count" in text and '"INVALID"' in text
    assert digest == builder._sha256(binding)
    with pytest.raises(builder.BuildInputError, match="already patched"):
        builder._add_local_combat_potion_binding(source)


@pytest.mark.parametrize("prefix", [[], ["// shifted by registered hook"]])
def test_versioned_potion_patch_preserves_unmarked_blank_context(tmp_path, prefix):
    source_root = tmp_path / "simulator"
    binding = source_root / "bindings" / "slaythespire.cpp"
    binding.parent.mkdir(parents=True)
    source_lines = [f"unchanged line {line}" for line in range(1, 90)]
    source_lines[9:14] = [
        "#include <sstream>",
        "#include <algorithm>",
        "",
        '#include "sim/ConsoleSimulator.h"',
        '#include "sim/search/ScumSearchAgent2.h"',
    ]
    source_lines[67:69] = [
        '        .def("get_card_reward", &sts::py::getCardReward, "return the current card reward list")',
        '        .def_property_readonly("encounter", [](const GameContext &gc) { return gc.info.encounter; })',
    ]
    binding.write_text("\n".join(prefix + source_lines) + "\n", encoding="utf-8", newline="\n")
    patch_path = Path(builder.__file__).resolve().parents[2] / builder.PATCH_RELATIVE_PATH

    builder._apply_versioned_patch(source_root, patch_path)

    patched = binding.read_text(encoding="utf-8")
    assert patched.count('.def_property_readonly("potions"') == 1
    assert "#include <cstddef>" in patched
    assert "#include <string>" in patched
    assert "#include <vector>" in patched


def test_pinned_source_verification_rejects_a_wrong_base_bytes(tmp_path, monkeypatch):
    files = {
        "bindings/slaythespire.cpp": b"binding source\n",
        "include/game/GameContext.h": b"game context\n",
        "include/constants/Potions.h": b"potion enum\n",
    }
    digests = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    monkeypatch.setattr(builder, "BASE_BINDING_SHA256", digests["bindings/slaythespire.cpp"])
    monkeypatch.setattr(builder, "GAME_CONTEXT_SHA256", digests["include/game/GameContext.h"])
    monkeypatch.setattr(builder, "POTIONS_HEADER_SHA256", digests["include/constants/Potions.h"])
    source_root = tmp_path / "simulator"
    for relative, content in files.items():
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    assert builder._verify_pinned_source_hashes(source_root) == digests
    (source_root / "bindings/slaythespire.cpp").write_bytes(b"different base\n")
    with pytest.raises(builder.BuildInputError, match="hash mismatch: bindings/slaythespire.cpp"):
        builder._verify_pinned_source_hashes(source_root)


def test_patch_chain_rejects_wrong_base_and_enforces_registered_order(tmp_path):
    source_root = tmp_path / "simulator"
    binding = source_root / "bindings" / "slaythespire.cpp"
    binding.parent.mkdir(parents=True)
    binding.write_text(
        "// base\n"
        'pybind11::class_<GameContext> gameContext(m, "GameContext");\n'
        "    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())\n"
        '        .def("pick_reward_card", &sts::py::pickRewardCard, "pick")\n'
        '        .def_readonly("player", &BattleContext::player)\n',
        encoding="utf-8",
        newline="\n",
    )
    hooks_patch = tmp_path / "hooks.patch"
    hooks_patch.write_text(
        "diff --git a/bindings/slaythespire.cpp b/bindings/slaythespire.cpp\n"
        "--- a/bindings/slaythespire.cpp\n"
        "+++ b/bindings/slaythespire.cpp\n"
        "@@ -1,2 +1,3 @@\n"
        " // base\n"
        "+// hook\n"
        ' pybind11::class_<GameContext> gameContext(m, "GameContext");\n',
        encoding="utf-8",
        newline="\n",
    )
    potion_patch = tmp_path / "potion.patch"
    potion_patch.write_text(
        "diff --git a/bindings/slaythespire.cpp b/bindings/slaythespire.cpp\n"
        "--- a/bindings/slaythespire.cpp\n"
        "+++ b/bindings/slaythespire.cpp\n"
        "@@ -1,3 +1,4 @@\n"
        " // base\n"
        " // hook\n"
        "+// potion patch follows the hook stage\n"
        ' pybind11::class_<GameContext> gameContext(m, "GameContext");\n'
        "@@ -5,2 +6,3 @@\n"
        '         .def("clone", [](const GameContext &gc) { return GameContext(gc); },\n'
        '              "copy the full run state for deterministic out-of-combat branching")\n'
        '+        .def_property_readonly("potions", [](const GameContext &) { return 5; })\n',
        encoding="utf-8",
        newline="\n",
    )

    wrong_base = tmp_path / "wrong-base"
    wrong_binding = wrong_base / "bindings" / "slaythespire.cpp"
    wrong_binding.parent.mkdir(parents=True)
    wrong_binding.write_text("// wrong base\n", encoding="utf-8", newline="\n")
    with pytest.raises(builder.BuildInputError, match="hook patch did not apply cleanly"):
        builder._apply_upstream_hook_patch(wrong_base, hooks_patch)
    assert wrong_binding.read_text(encoding="utf-8") == "// wrong base\n"

    wrong_order = tmp_path / "wrong-order"
    wrong_order_binding = wrong_order / "bindings" / "slaythespire.cpp"
    wrong_order_binding.parent.mkdir(parents=True)
    wrong_order_binding.write_text(binding.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    with pytest.raises(builder.BuildInputError, match="pinned source does not match versioned patch context"):
        builder._apply_versioned_patch(wrong_order, potion_patch)

    stages = builder._apply_binding_patch_chain(source_root, hooks_patch, potion_patch)
    assert stages["patch_stage_order"] == [
        "upstream_hook_patch",
        "local_clone_binding",
        "potion_inventory_binding_patch",
        "live_combat_potion_binding",
    ]
    final_text = binding.read_text(encoding="utf-8")
    assert final_text.count('.def_property_readonly("potions"') == 1
    assert "// potion patch follows the hook stage" in final_text

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


GAMEPLAY_COMMIT = "7476a81954020087da31d41d16fddf475746ec2d"
GAMEPLAY_REPOSITORY = "https://github.com/gamerpuppy/sts_lightspeed"
HOOK_PATCH_BLOB_SHA1 = "83d3a89ba0f7639e93f35df6b8f27bf2fe6326a8"
POTION_PATCH_SHA256 = "fe75bb6fee0f83689bd5ab541a14ffead4f0df6b5b855c4f30096d308a4f9c7b"
BASE_BINDING_SHA256 = "2b7220cb976205ee094143915825b3f004f2d3d69338e07dbaf27a56c8d45169"
GAME_CONTEXT_SHA256 = "b78fd16448b96370fd78740a14baa00b4f6811c820dc544dd5c38cba01f73133"
POTIONS_HEADER_SHA256 = "998ce625b620c1922222e5b1b7fdc904003d3cb8b765b783f8d9a1688dc7b134"
PYBIND11_OVERRIDE_COMMIT = "3e9dfa2866941655c56877882565e7577de6fc7b"
PATCH_RELATIVE_PATH = Path(
    "control/sts1-g7-improvement/native-patches/potion-inventory-binding.patch"
)


class BuildInputError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _verify_pinned_source_hashes(source_root: Path) -> dict[str, str]:
    expected_hashes = {
        "bindings/slaythespire.cpp": BASE_BINDING_SHA256,
        "include/game/GameContext.h": GAME_CONTEXT_SHA256,
        "include/constants/Potions.h": POTIONS_HEADER_SHA256,
    }
    observed_hashes: dict[str, str] = {}
    for relative, expected in expected_hashes.items():
        path = source_root / relative
        if not path.is_file():
            raise BuildInputError(f"pinned native source is missing: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise BuildInputError(f"pinned native source hash mismatch: {relative}")
        observed_hashes[relative] = observed
    return observed_hashes


def _compatibility_cxx_flags(cache: dict[str, str]) -> tuple[str, str]:
    # Pinned GCC/Clang builds cannot rely on transitive STL includes.
    # Preserve the reference flags and supply the missing header at build time.
    build_type = cache.get("CMAKE_BUILD_TYPE") or "Release"
    key = f"CMAKE_CXX_FLAGS_{build_type.upper()}"
    if key not in cache:
        raise BuildInputError("reference configuration C++ flags are missing")
    # Upstream overwrites CMAKE_CXX_FLAGS; configuration flags are preserved.
    return key, (cache[key] + " -include algorithm").strip()


def _patch_target_paths(patch_path: Path) -> list[Path]:
    targets: list[Path] = []
    for line in patch_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("+++ b/"):
            continue
        raw_path = line[6:].split("\t", 1)[0]
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise BuildInputError("pinned simulator patch contains an unsafe target path")
        target = Path(*relative.parts)
        if target not in targets:
            targets.append(target)
    if not targets:
        raise BuildInputError("pinned simulator hook patch has no source targets")
    return targets


def _apply_upstream_hook_patch(source_root: Path, hooks_patch: Path) -> None:
    source_root = source_root.resolve(strict=True)
    targets = _patch_target_paths(hooks_patch)
    for relative in targets:
        path = (source_root / relative).resolve(strict=True)
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise BuildInputError("pinned simulator patch target escaped the source tree") from exc
        if not path.is_file():
            raise BuildInputError("pinned simulator patch target is not a regular file")

    for arguments in (("--check",), ()):
        completed = subprocess.run(
            ["git", "apply", *arguments, str(hooks_patch)],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise BuildInputError("pinned simulator hook patch did not apply cleanly")


def _add_local_clone_binding(source_root: Path) -> str:
    binding = source_root / "bindings" / "slaythespire.cpp"
    text = binding.read_text(encoding="utf-8")
    if "\r" in text:
        raise BuildInputError("pinned binding contains non-LF bytes before local clone patch")
    needle = (
        '    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())\n'
        '        .def("pick_reward_card", &sts::py::pickRewardCard, '
    )
    replacement = (
        '    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())\n'
        '        .def("clone", [](const GameContext &gc) { return GameContext(gc); },\n'
        '             "copy the full run state for deterministic out-of-combat branching")\n'
        '        .def("pick_reward_card", &sts::py::pickRewardCard, '
    )
    if '.def("clone", [](const GameContext &gc)' in text:
        raise BuildInputError("local GameContext clone binding is already present")
    if text.count(needle) != 1:
        raise BuildInputError("pinned GameContext clone binding anchor is missing or ambiguous")
    binding.write_text(text.replace(needle, replacement, 1), encoding="utf-8", newline="\n")
    return _sha256(binding)


def _parse_cmake_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or ":" not in line:
            continue
        key, typed_value = line.split(":", 1)
        if "=" not in typed_value:
            continue
        _value_type, value = typed_value.split("=", 1)
        values[key] = value
    return values


def _apply_versioned_patch(source_root: Path, patch_path: Path) -> None:
    relative = Path("bindings/slaythespire.cpp")
    patch_lines = patch_path.read_text(encoding="utf-8").splitlines()
    expected_header = f"diff --git a/{relative.as_posix()} b/{relative.as_posix()}"
    if patch_lines.count(expected_header) != 1:
        raise BuildInputError("versioned patch must target only bindings/slaythespire.cpp")
    if (
        f"--- a/{relative.as_posix()}" not in patch_lines
        or f"+++ b/{relative.as_posix()}" not in patch_lines
    ):
        raise BuildInputError("versioned patch file headers are invalid")

    source_lines = (source_root / relative).read_text(encoding="utf-8").splitlines()
    hunks: list[tuple[int, int, int, int, list[str], list[str]]] = []
    cursor = 0
    hunk_re = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@(?:.*)$")
    while cursor < len(patch_lines):
        match = hunk_re.match(patch_lines[cursor])
        if match is None:
            cursor += 1
            continue
        old_start, old_count, new_start, new_count = map(int, match.groups())
        cursor += 1
        old_lines: list[str] = []
        new_lines: list[str] = []
        while cursor < len(patch_lines) and not patch_lines[cursor].startswith("@@ "):
            line = patch_lines[cursor]
            if line == "":
                old_lines.append("")
                new_lines.append("")
            elif line[0] == " ":
                old_lines.append(line[1:])
                new_lines.append(line[1:])
            elif line[0] == "+":
                new_lines.append(line[1:])
            elif line[0] == "-":
                raise BuildInputError("read-only potion binding patch cannot delete source lines")
            elif line.startswith("\\ No newline"):
                pass
            else:
                raise BuildInputError("versioned patch has an unsupported hunk line")
            cursor += 1
        if len(old_lines) != old_count or len(new_lines) != new_count:
            raise BuildInputError("versioned patch hunk line counts are invalid")
        hunks.append((old_start, old_count, new_start, new_count, old_lines, new_lines))
    if len(hunks) != 2:
        raise BuildInputError("versioned patch must contain exactly two source hunks")

    offset = 0
    for old_start, old_count, new_start, new_count, old_lines, new_lines in hunks:
        if new_start != old_start + offset:
            raise BuildInputError("versioned patch hunk positions are inconsistent")
        index = old_start - 1 + offset
        if source_lines[index:index + old_count] != old_lines:
            raise BuildInputError("pinned source does not match versioned patch context")
        source_lines[index:index + old_count] = new_lines
        offset += new_count - old_count
    target = source_root / relative
    target.write_text("\n".join(source_lines) + "\n", encoding="utf-8", newline="\n")


def _hash_patch_targets(source_root: Path, targets: list[Path]) -> dict[str, str]:
    return {path.as_posix(): _sha256(source_root / path) for path in targets}


def _apply_binding_patch_chain(
    source_root: Path, hooks_patch: Path, potion_patch: Path
) -> dict[str, Any]:
    binding = source_root / "bindings" / "slaythespire.cpp"
    hook_targets = _patch_target_paths(hooks_patch)
    hook_inputs = _hash_patch_targets(source_root, hook_targets)
    _apply_upstream_hook_patch(source_root, hooks_patch)
    hook_outputs = _hash_patch_targets(source_root, hook_targets)

    clone_binding_sha256 = _add_local_clone_binding(source_root)
    potion_input_binding_sha256 = _sha256(binding)
    _apply_versioned_patch(source_root, potion_patch)
    potion_output_binding_sha256 = _sha256(binding)
    patched_text = binding.read_text(encoding="utf-8")
    if patched_text.count('.def_property_readonly("potions"') != 1:
        raise BuildInputError("patched binding does not contain exactly one read-only potion property")

    return {
        "patch_stage_order": [
            "upstream_hook_patch",
            "local_clone_binding",
            "potion_inventory_binding_patch",
        ],
        "hook_patch_input_sha256": hook_inputs,
        "hook_patch_output_sha256": hook_outputs,
        "clone_binding_sha256": clone_binding_sha256,
        "potion_patch_input_binding_sha256": potion_input_binding_sha256,
        "potion_patch_output_binding_sha256": potion_output_binding_sha256,
    }


def _normal(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _require_file(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise BuildInputError(f"{label} is not a file")
    return path


def _require_directory(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_dir():
        raise BuildInputError(f"{label} is not a directory")
    return path


def _validate_inputs(
    *,
    repo_root: Path,
    source_root: Path,
    hooks_patch: Path,
    reference_build: Path,
    cmake: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    if source_root.name.casefold() != f"sts_lightspeed-{GAMEPLAY_COMMIT}".casefold():
        raise BuildInputError("native source folder does not match the pinned gameplay commit")

    upstream_path = repo_root / "external/sts_lightspeed/UPSTREAM.json"
    agent_upstream_path = repo_root / "external/sts-rl-agent/UPSTREAM.json"
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    agent_upstream = json.loads(agent_upstream_path.read_text(encoding="utf-8"))
    if (
        upstream.get("repository") != GAMEPLAY_REPOSITORY
        or upstream.get("commit") != GAMEPLAY_COMMIT
        or upstream.get("pybind11_build_compat", {}).get("override_commit")
        != PYBIND11_OVERRIDE_COMMIT
    ):
        raise BuildInputError("registered gameplay or pybind11 pin differs from this build")
    if agent_upstream.get("patch_git_blob_sha") != HOOK_PATCH_BLOB_SHA1:
        raise BuildInputError("registered simulator hook patch identity differs")
    if _git_blob_sha1(hooks_patch) != HOOK_PATCH_BLOB_SHA1:
        raise BuildInputError("local simulator hook patch does not match its registered Git blob")

    source_hashes = _verify_pinned_source_hashes(source_root)

    patch_path = repo_root / PATCH_RELATIVE_PATH
    if _sha256(patch_path) != POTION_PATCH_SHA256:
        raise BuildInputError("versioned potion binding patch hash mismatch")

    cache_path = reference_build / "CMakeCache.txt"
    if not cache_path.is_file():
        raise BuildInputError("reference CMakeCache.txt is missing")
    cache = _parse_cmake_cache(cache_path)
    if _normal(Path(cache.get("CMAKE_HOME_DIRECTORY", ""))) != _normal(source_root):
        raise BuildInputError("reference build was not configured from the pinned source tree")
    if cache.get("CMAKE_GENERATOR") != "Ninja":
        raise BuildInputError("reference build generator is not Ninja")

    paths = {
        "compiler": _require_file(cache.get("CMAKE_CXX_COMPILER", ""), "pinned C++ compiler"),
        "ninja": _require_file(cache.get("CMAKE_MAKE_PROGRAM", ""), "pinned Ninja executable"),
        "python": _require_file(
            cache.get("Python_EXECUTABLE") or cache.get("PYTHON_EXECUTABLE", ""),
            "pinned Python executable",
        ),
        "python_include": _require_directory(cache.get("Python_INCLUDE_DIR", ""), "Python include directory"),
        "python_library": _require_file(cache.get("Python_LIBRARY", ""), "Python library"),
        "pybind_include": _require_directory(
            cache.get("PYBIND11_INCLUDE_DIR", ""), "pinned pybind11 include directory"
        ),
        "cmake": cmake,
    }
    if paths["pybind_include"].name.casefold() != "include":
        raise BuildInputError("CMake reference does not resolve to the registered pybind11 include tree")
    if paths["python"].resolve() != _require_file(
        cache.get("PYTHON_EXECUTABLE", ""), "pinned legacy Python executable"
    ).resolve():
        raise BuildInputError("reference Python executable entries disagree")

    return {
        "source_root": source_root,
        "patch_path": patch_path.resolve(strict=True),
        "source_hashes": source_hashes,
        "paths": paths,
        "cache": cache,
        "hook_patch_blob_sha1": HOOK_PATCH_BLOB_SHA1,
        "potion_patch_sha256": POTION_PATCH_SHA256,
    }


def _run_logged(
    command: list[str], *, cwd: Path, log_path: Path, label: str,
    env: dict[str, str] | None = None,
) -> None:
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            env=env,
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        raise BuildInputError(
            f"{label} failed with exit code {completed.returncode}; log tail:\n"
            + "\n".join(tail)
        )


def _run_native_smoke(module_path: Path, python: Path) -> dict[str, Any]:
    smoke = r"""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("slaythespire", sys.argv[1])
if spec is None or spec.loader is None:
    raise RuntimeError("compiled native binding did not load")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
context = module.GameContext(module.CharacterClass.IRONCLAD, 0, 0)
slots = list(context.potions)
expected = ["EMPTY_POTION_SLOT"] * 5
if slots != expected:
    raise AssertionError(f"expected five exact empty-slot names; received {slots!r}")
try:
    context.potions = ["FIRE_POTION"] * 5
except AttributeError:
    pass
else:
    raise AssertionError("potion inventory binding is writable")
print(json.dumps({"potion_slots": slots, "episodes_started": 0}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python), "-c", smoke, str(module_path)],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise BuildInputError(
            "native binding smoke failed: " + completed.stderr[-2000:].strip()
        )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise BuildInputError("native binding smoke returned no canonical JSON result") from exc
    if result != {"episodes_started": 0, "potion_slots": ["EMPTY_POTION_SLOT"] * 5}:
        raise BuildInputError("native binding smoke result is incomplete")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the pinned STS1 binding with only the registered read-only potion snapshot patch."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hooks-patch", type=Path, required=True)
    parser.add_argument("--reference-build", type=Path, required=True)
    parser.add_argument("--cmake", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    source_root = args.source_root.resolve(strict=True)
    hooks_patch = _require_file(str(args.hooks_patch), "registered simulator hook patch")
    reference_build = args.reference_build.resolve(strict=True)
    cmake = _require_file(str(args.cmake), "CMake executable")
    scratch_root = args.scratch_root.expanduser().resolve()
    if scratch_root.exists():
        raise BuildInputError("scratch root already exists; refusing to overwrite prior evidence")
    if scratch_root == source_root or source_root in scratch_root.parents:
        raise BuildInputError("scratch root must not be inside the preserved upstream source tree")
    if scratch_root == reference_build or reference_build in scratch_root.parents:
        raise BuildInputError("scratch root must not be inside the preserved reference build")

    verified = _validate_inputs(
        repo_root=repo_root,
        source_root=source_root,
        hooks_patch=hooks_patch,
        reference_build=reference_build,
        cmake=cmake,
    )
    paths = verified["paths"]
    scratch_root.mkdir(parents=True, exist_ok=False)
    copied_source = scratch_root / "source"
    build_root = scratch_root / "build"
    shutil.copytree(source_root, copied_source, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    copied_source_hashes = _verify_pinned_source_hashes(copied_source)
    if copied_source_hashes != verified["source_hashes"]:
        raise BuildInputError("pinned source bytes changed while copying into the scratch tree")

    binding = copied_source / "bindings/slaythespire.cpp"
    patch_stage_hashes = _apply_binding_patch_chain(
        copied_source,
        hooks_patch,
        verified["patch_path"],
    )

    compatibility_key, compatibility_flags = _compatibility_cxx_flags(verified["cache"])
    configure = [
        str(cmake),
        "-S",
        str(copied_source),
        "-B",
        str(build_root),
        "-G",
        "Ninja",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        f"-DCMAKE_MAKE_PROGRAM={paths['ninja']}",
        f"-DCMAKE_CXX_COMPILER={paths['compiler']}",
        f"-D{compatibility_key}={compatibility_flags}",
        f"-DCMAKE_BUILD_TYPE={verified['cache'].get('CMAKE_BUILD_TYPE', 'Release')}",
        f"-DPython_EXECUTABLE={paths['python']}",
        f"-DPYTHON_EXECUTABLE={paths['python']}",
        f"-DPython_INCLUDE_DIR={paths['python_include']}",
        f"-DPython_LIBRARY={paths['python_library']}",
        f"-DPYBIND11_INCLUDE_DIR={copied_source / 'pybind11/include'}",
        "-DPYBIND11_PYTHONLIBS_OVERWRITE=ON",
    ]
    build_env = os.environ.copy()
    build_env["ZIG_LOCAL_CACHE_DIR"] = str(scratch_root / "zig-local-cache")
    build_env["ZIG_GLOBAL_CACHE_DIR"] = str(scratch_root / "zig-global-cache")
    Path(build_env["ZIG_LOCAL_CACHE_DIR"]).mkdir()
    Path(build_env["ZIG_GLOBAL_CACHE_DIR"]).mkdir()
    _run_logged(
        configure,
        cwd=scratch_root,
        log_path=scratch_root / "configure.log",
        label="CMake configure",
        env=build_env,
    )
    _run_logged(
        [str(cmake), "--build", str(build_root), "--target", "slaythespire", "--parallel", "2"],
        cwd=scratch_root,
        log_path=scratch_root / "build.log",
        label="native binding build",
        env=build_env,
    )

    modules = list(build_root.rglob("slaythespire*.pyd"))
    if len(modules) != 1:
        raise BuildInputError(f"expected one built Python extension, found {len(modules)}")
    module_path = modules[0].resolve(strict=True)
    smoke = _run_native_smoke(module_path, paths["python"])
    result = {
        "status": "PASS",
        "gameplay_source_commit": GAMEPLAY_COMMIT,
        "hook_patch_git_blob_sha1": HOOK_PATCH_BLOB_SHA1,
        "potion_patch_sha256": POTION_PATCH_SHA256,
        "patched_paths": ["bindings/slaythespire.cpp"],
        "build_compatibility_cxx_flags": {compatibility_key: compatibility_flags},
        "source_input_sha256": verified["source_hashes"],
        "base_binding_sha256": BASE_BINDING_SHA256,
        **patch_stage_hashes,
        "result_binding_sha256": _sha256(binding),
        "native_module_sha256": _sha256(module_path),
        "module_path": str(module_path),
        "test_only_initializer_seed": 0,
        **smoke,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildInputError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "NOT_VERIFIED", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2)

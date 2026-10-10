# G7 native binding compile repair

Issue #19; prerequisite repair for H19/Round012 research. Base PR #28 HEAD:
`f585bb568737b0076ef831ac9e73574dfacc5d38`.

The Windows pinned-binding check (run `38021302877`) failed while compiling
`src/combat/Actions.cpp`: `std::sort` was unavailable because the source did not
include `<algorithm>` directly.

The first repair passed Actions.cpp, then run `38024509411` exposed the same
missing header in BattleContext.cpp. The final repair supplies `-include algorithm`
through configuration-specific CMake flags for the pinned GCC/Clang toolchains,
preserving all reference C++ flags. Run `38024740783` showed that upstream
overwrites the general CMAKE_CXX_FLAGS variable; the final repair uses the
reference build-type flags and rejects missing configuration flags.
No combat source is rewritten. Existing exact-source validation and the
registered binding patch chain remain intact; build output records the full flags.

Local CPython 3.12 verification with the existing native module: 8 passed across
`test_sts1_pinned_binding_builder.py` and `test_sts1_h19_potion_binding.py`.
This is not proof of the new native build; the exact-head CI must pass.
Whitespace check passed. All four changed paths are within Issue #19 scope.

No episodes, new pools, model training, or Champion changes occurred.
H19 Round011 remains Train/Probe net zero; its failed Probe coverage gate and
consumed pools are unchanged. New win-rate improvement, fresh confirmation,
and real-game acceptance remain NOT_VERIFIED. Native build/smoke must pass
before this prerequisite is considered complete.

Run `38024930056` successfully built the native module and passed the zero-episode
smoke, but its regression step failed because DLL isolation deliberately removed
Git from PATH while patch-chain unit tests require Git. The workflow now runs
builder tests before PATH isolation and retains the isolated native-module test
under the original restricted PATH. It does not add Git or compiler DLL paths to
the native smoke. A test regex was also made a raw string to remove a warning.

## Bounded next-direction inventory

Read-only Train-only Sample(10): streamed the ten Round011 parent traces
(29,650,339 bytes), retaining only the last encounter-start and terminal record
per trace. All ten terminals were complete defeats. Terminal act bands were
1/5/4 for Act 1 / Act 2 / Act 3+. Eight final encounters started at at least
50% HP and two below 50%. Bronze Automaton was present in three last-encounter
snapshots; no other named encounter was represented in three traces.

These aggregates do not establish a correctable policy factor or a causal
intervention. They argue against assuming that low starting HP alone explains
this sample. A next strategy hypothesis still requires decision-time evidence
and a matched train counterfactual; no new Candidate is registered here.
Probe/Dev traces and formal outcomes were not read for this audit. Raw state,
seed IDs and provenance remained private. Provider token usage is NOT_MEASURED;
no policy provider call was added. Context-mode tools were not exposed, so the
fallback emitted only bounded aggregates.

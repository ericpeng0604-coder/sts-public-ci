# G7 native binding compile repair

Issue #19; prerequisite repair for H19/Round012 research. Base PR #28 HEAD:
`f585bb568737b0076ef831ac9e73574dfacc5d38`.

The Windows pinned-binding check (run `38021302877`) failed while compiling
`src/combat/Actions.cpp`: `std::sort` was unavailable because the source did not
include `<algorithm>` directly.

The first repair passed Actions.cpp, then run `38024509411` exposed the same
missing header in BattleContext.cpp. The final repair supplies `-include algorithm`
through CMake for the pinned GCC/Clang toolchains, preserving all reference C++
flags. No combat source is rewritten. Existing exact-source validation and the
registered binding patch chain remain intact; build output records the full flags.

Local CPython 3.12 verification with the existing native module: 8 passed across
`test_sts1_pinned_binding_builder.py` and `test_sts1_h19_potion_binding.py`.
This is not proof of the new native build; the exact-head CI must pass.
Whitespace check passed. All three changed paths are within Issue #19 scope.

No episodes, new pools, model training, or Champion changes occurred.
H19 Round011 remains Train/Probe net zero; its failed Probe coverage gate and
consumed pools are unchanged. New win-rate improvement, fresh confirmation,
and real-game acceptance remain NOT_VERIFIED. Native build/smoke must pass
before this prerequisite is considered complete.

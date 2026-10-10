# G7 native binding compile repair

Issue #19; prerequisite repair for H19/Round012 research. Base PR #28 HEAD:
`f585bb568737b0076ef831ac9e73574dfacc5d38`.

The Windows pinned-binding check (run `38021302877`) failed while compiling
`src/combat/Actions.cpp`: `std::sort` was unavailable because the source did not
include `<algorithm>` directly.

The builder now checks the exact pinned Actions.cpp Git blob
`93db96fad413174009b54c13f6814ea0e6347073` before adding only that include to its
scratch copy. Unknown bytes fail closed before any write. The preserved upstream
source and gameplay logic remain unchanged. The build result records both the
input blob and patched SHA-256 and includes Actions.cpp in its patched-path list.

Local CPython 3.12 verification: 8 passed, 1 skipped across
`test_sts1_pinned_binding_builder.py` and `test_sts1_h19_potion_binding.py`.
The skip requires the compiled native module and is not runtime proof.
Whitespace check passed. All three changed paths are within Issue #19 scope.

No episodes, new pools, model training, or Champion changes occurred.
H19 Round011 remains Train/Probe net zero; its failed Probe coverage gate and
consumed pools are unchanged. New win-rate improvement, fresh confirmation,
and real-game acceptance remain NOT_VERIFIED. Native build/smoke must pass
before this prerequisite is considered complete.

# Isolated Courier pricing correction

The native simulator revision used by the active frozen cohort has three verified
restock-pricing defects in `src/game/Shop.cpp`:

- Potion/relic price generation overwrites the item's base price with a random
  multiplier near 1 and ignores discount calculation results.
- Card restocking checks Courier twice, so its extra 50% discount applies without
  Membership Card and Membership Card alone gives no discount.
- Relic restocking computes a new price but does not assign it to the shop slot.

`courier-pricing.patch` corrects these three defects. It preserves the existing
number of merchant RNG draws in each pricing function. `test_pricing.cpp` tests
real native C++ pricing functions across 64 seeds, all discount combinations,
colorless and class cards, and the charge for a second restocked relic purchase.

Local verification against the pinned public native source:

- Original native implementation: 1024 checks, 576 failures (expected regression
  reproduction).
- Corrected native implementation: 1024 checks, 0 failures.

This patch is NOT applied to the currently executing recovery/training cohort.
That cohort retains its original native engine and is explicitly prohibited from
supporting real-game performance claims. A corrected engine requires collecting
new pre-decision states and labels with its own engine provenance. Existing
labels must not be mixed across these engine versions without independently
proving their trajectories were unaffected. Other native simulator fidelity
issues, including high-ascension pricing, are outside this targeted A0 correction.

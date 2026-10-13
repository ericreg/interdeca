# Review order for the fish research PR

1. Read [scope and reproduced results](paper-reproduction.md) and the
   [review disposition](review-assessment.md). They distinguish execution,
   selected headline agreement, strict numerical parity and inference.
2. Review authored runner/bundle orchestration in `experiments/fishy/paper.py`
   and `release.py`, then boundary records/schemas in `records.py` and `serde.py`.
   Native generation/import/transfer is a separate integration capability;
   the archived paper campaign did not reconstruct a new atlas with it.
3. Review the [comparison policy](comparison-policy.md), its implementation and
   behavioral regressions. Review the [inference supplement](inference.md)
   separately from the preserved author diagnostics.
4. Run the [portable CPU tests](../../tests/README.md), then check
   [published evidence integrity](evidence/2026-10-13/README.md).
5. Inspect the concise evidence summaries first; use full differences and
   manifests to investigate specific values or source/input identity questions.
   An accessible complete private-data handoff is still needed for a full rerun.

`experiments/fishy/reference/` is an attributed, pinned scientific snapshot.
Its `upstream.json` records individual source hashes. Review its provenance and
scientific assumptions; do not confuse verbatim archived source with newly
authored integration code. Historic evidence under `docs/fishy/evidence/2026-10-06/`
is preserved unchanged. New receipts identify the source they actually tested,
even when published by a later commit.

Core `src/interdeca/`, Slicer adapters, the original `../3d_color_package/`, and
the paper's numerical reference are outside this review fix's modification scope.
No formal proof, production certification, or strict author parity is claimed.

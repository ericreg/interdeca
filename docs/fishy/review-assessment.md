# Research-repository review: verified disposition

The supplied review was checked against branch head
`0d9fe7df1beaa67e82624a4268409fc4d4d33863`, the committed October 6 artifacts,
the retained clean-run receipts, and executable causal probes. Its final
sentence was truncated in the supplied attachment; no missing recommendation
has been inferred. Formal proofs are deferred while these concrete issues are
addressed.

## Findings that hold up

- **Strict author parity fails.** The implementation reproduces execution and
  selected headline results, not all original numerical outputs. No source
  evidence establishes a single cause for every divergence. Original-author
  input hashes and platform/BLAS provenance remain unavailable.
- **Latest evidence was not committed.** The PR's later clean-run comparison
  and focused E1 claim pointed to local records that reviewers could not inspect.
  The [published evidence index](evidence/2026-10-13/README.md) now provides the
  complete historical comparisons, source/input/output identities, and raw
  versus normalized fidelity hashes. Raw E1 JSON bytes differ in `seconds`;
  numerical leaves, NPZ bytes and PNG bytes match the prior same-host run.
- **The old array gate could hide errors.** A large reference element could
  mask another element's discrepancy. Near-zero floating values lacked an
  explicit absolute allowance. The revised [comparison policy](comparison-policy.md)
  uses symmetric, per-element bounds, exact discrete values and structure,
  explicit undefined evidence, bounded memory, and a declared sensitivity grid.
  All real saved-result comparisons still fail parity.
- **Repeated-fold inference was miscalibrated.** The archived follow-up uses
  ordinary paired t-tests and SD/√20 on overlapping repeated folds. These
  statistics remain historical diagnostics, not calibrated population evidence.
  A separate [donor-held-out supplement](inference.md) reports descriptive
  contrasts without p values or intervals. It does not certify end-to-end
  feature-construction holdout.
- **There were no committed automated tests or CI.** The initial repository
  policy explicitly deferred tests. The user's new request supersedes that
  deferral. [Behavioral tests](../../tests/README.md) now exercise validation,
  historical layouts, stage resume/invalidation, comparison errors, inference
  and evidence integrity in a pinned CPU environment; CI runs those checks.
- **Some external settings were coercible and mutable.** The base generation
  schemas are now strict, frozen and finite, with immutable collections and an
  explicit legacy CSV adapter. Finite numerical guards reject invalid direct
  generator inputs while preserving seeded draws and spline results.
- **Manifest dispatch inferred its kind from content.** Bundle versus legacy
  export writes now select the schema explicitly. Expected boundary failures
  in the paper, bundle and native CLI entrypoints have concise error messages;
  their library methods retain typed exceptions.

## Claims needing qualification

- **702/37 was not an arithmetic mistake.** October 6 versus author is
  714/8,110 leaves and 38/146 array fields; the later full-clean versus author
  is 702/8,110 and 37/146. Full-clean versus October 6 is 178/8,110 and 9/146.
  The missing public campaign distinction was the defect. Preserved historical
  counts must not be relabelled as results from the new comparison policy.
- **The old JSON formula was already symmetric.** The old array formula was
  asymmetric. Both methods needed clearer tolerance interpretation, but the
  blanket assertion of asymmetric scalar comparison was inaccurate.
- **CompositionIdentity is a deliberate projection.** It reads IDs from a
  richer core stage document. Ignoring unrelated stage fields is intentional;
  forbidding them would reject valid inputs. Its selected identities now have
  strict, frozen, nonempty contracts. Finiteness of unused unrelated fields is
  outside that identity-only contract.
- **Object arrays were a policy gap, not observed missing coverage here.** The
  required real comparison scope contains no object-array fields. The comparator
  still refuses pickle loading, and any future object field prevents a complete
  parity pass rather than silently being accepted.
- **Embedding alignment is a separate question.** Rotations, reflections or
  other non-identifiability can motivate geometric diagnostics; alignment cannot
  establish raw numerical identity and is excluded from strict comparison.
- **Research grade is not one binary universal certification.** The verified
  claim remains an explicitly scoped execution/headline reproduction with known
  differences. Production readiness and original-paper numerical equivalence
  are separate claims requiring separate evidence.

## External limits and reviewability

No LICENSE was found that resolves redistribution of the attributed author code
or native dataset. Attribution and local possession do not establish permission.
No license grant has been invented and no raw dataset has been publicly uploaded.
Complete public data access and archival distribution remain unresolved.

The large PR contains preserved author code, historic result JSONs and figures,
alongside new integration code. Those materials already occupy separate
`reference/`, `docs/fishy/evidence/` and integration paths. The
[reviewer guide](reviewer-guide.md) gives a review order and scope; large generated
artifacts are evidence, not authored application logic. The changes remain a
draft PR and have not been merged.

# Study B guarded exact-refit rep=100 QA

- All six methods use the same generated data within each `(cell, rep)`.
- E-CGL is the main method; E-ACGL is the adaptive extension.
- E-CGL/E-ACGL use BIC-after exact centered-support refits.
- Shortlist40 is guarded: a winner at rank 38 or higher triggers full refitting.
- Failed or non-converged exact refits are ineligible, and any unresolved exact candidate fails QA.
- Standard-method computational failures remain explicit; metric means are conditional on successful fits and retain their valid counts.
- Zero-support outcomes remain in unconditional F1/ARI summaries as zero rather than being dropped.

- Audited cells passing integrity checks: 12/12.
- Standard-method computational failures: 1/4,800 attempts.
- Exact candidates requiring deterministic continuation: 24.
- Maximum exact-refit outer iterations after continuation: 566.
- Total final method-replicate rows: 7200 (expected 7200).
- Maximum winner BIC-before rank: 34.
- Full-fallback method-replicates: 0.
- Zero-support final rows: 0.
- Maximum selected group constraint error: 3.553e-15.
- Summary groups complete: TRUE (72 expected).
- Paired E-method groups complete: TRUE (1,200 pairs expected).
- Paired metric counts complete: TRUE (100 per cell expected).
- Calibration identical across n within each design: TRUE.
- Calibration targets inside independent MC intervals: 8/12 cells.
- Maximum absolute target-achieved error: 0.0023.

This output is a computational audit artifact. Interpretation and document updates require a separate review.

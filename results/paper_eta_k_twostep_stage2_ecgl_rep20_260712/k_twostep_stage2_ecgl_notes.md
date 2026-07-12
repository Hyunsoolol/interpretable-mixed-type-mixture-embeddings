# Study B paired two-step E-CGL diagnostic

## Procedure

- Stage 1: dense free-kappa vMF; K selected by independent test NLL over K=2,...,8.
- Stage 2: selected K fixed; E-CGL path followed by BIC-after-exact-centered-refit.
- n=1000, d=200, target eB=0.05, reps=20 per scenario, Eta path=240.
- The independent test selector is a simulation diagnostic, not a deployable real-data rule.

## Results

| scenario | K=4 rate | selected q | common q | decision q | noise q | F1 | ARI | MSE eta | exact support rate |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| equal | 1.000 | 16.050 | 0.000 | 16.000 | 0.050 | 0.998 | 0.861 | 0.060 | 0.950 |
| heterogeneous | 1.000 | 16.050 | 0.000 | 16.000 | 0.050 | 0.998 | 0.869 | 0.060 | 0.950 |

## QA

- Final rows: 40/40.
- QA checks passed: 9/9.
- Maximum Stage-1 dense log-likelihood reproduction difference: 4.948e-10.
- Exact candidate refits: 1600; ineligible: 0.
- Full-shortlist fallback reps: 0.
- Total elapsed time: 253.1 seconds.

## Interpretation boundary

- Labels are used only for ARI and support-recovery evaluation.
- Stage-1 independent test NLL uses the known simulation distribution through a held-out sample.
- Candidate-support output is an audit artifact and is not a commit candidate.

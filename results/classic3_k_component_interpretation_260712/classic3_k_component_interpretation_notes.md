# Classic3 K=3 versus K=10 component interpretation

- Exact E-CGL BIC-after-refit fits are reused; no model is refitted.
- Supplied labels are used only for post-fit evaluation and component naming.
- Test documents: 779; vocabulary coordinates: 2000.

## K=10 component profile

| component | majority topic | test n | purity | positive centered-Eta tokens |
|---|---|---:|---:|---|
| CISI-1 | CISI | 147 | 1.000 | scientific (+133.9); information (+125.4); science (+100.7) |
| CISI-2 | CISI | 85 | 1.000 | library (+515.7); librarian (+453.2); libraries (+388.2) |
| CISI-3 | CISI | 59 | 1.000 | retrieval (+539.4); retrieve (+345.1); information (+297.3) |
| CRAN-1 | CRAN | 78 | 0.923 | heat (+106.4); thermal (+67.5); temperature (+66.4) |
| CRAN-2 | CRAN | 60 | 1.000 | boundary (+339.1); lam (+289.4); layers (+286.3) |
| CRAN-3 | CRAN | 56 | 1.000 | flow (+230.3); super (+188.6); wing (+151.8) |
| CRAN-4 | CRAN | 55 | 1.000 | mach (+364.6); tunnel (+222.2); pressure (+220.2) |
| CRAN-5 | CRAN | 37 | 1.000 | buck (+434.4); shell (+223.1); shells (+207.1) |
| MED-1 | MED | 123 | 1.000 | inhibitor (+137.2); rat (+109.8); dose (+92.5) |
| MED-2 | MED | 79 | 1.000 | child (+128.9); disease (+115.1); children (+109.1) |

## Interpretation boundary

- K=10 has high component purity but divides each supplied broad topic into multiple components.
- Post-hoc component names do not imply supervised fitting.
- Token contrasts describe relative component scores, not absolute word preferences.
- K=3 remains the externally defined broad-topic benchmark; K=10 is a finer density resolution.

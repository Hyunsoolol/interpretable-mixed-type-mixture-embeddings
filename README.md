# thesis-hp-clustering

Research notes for a sparse directional clustering thesis project based on vMF mixtures and eta-contrast variable selection.

## Current Repository Scope

This repository is currently kept as a document-first research meeting archive.

- R source code is intentionally kept local and excluded from Git for security.
- Public-facing materials are limited to thesis notes, meeting summaries, and selected result summaries.
- The previous README content was archived in `docs/archive/README_previous_260616.md`.

## Folder Layout

- `docs/meetings/`: research meeting notes and current meeting summary.
- `docs/methods/`: model and code-level methodology notes.
- `docs/implementation/`: current implementation status notes.
- `docs/simulations/`: simulation design, diagnostics, and interpretation notes.
- `docs/proposals/`: earlier proposal and research-note drafts.
- `docs/references/`: reference notes and selected papers.
- `legacy/python/`: old Python prototypes kept for historical context.
- `r/`: local-only R scripts, ignored by Git and organized by role.

## Key Documents

- `docs/meetings/thesis-meeting_260624.md`: meeting-ready summary and current decisions.
- `docs/methods/thesis-methods_260624.md`: method definitions, objective functions, estimation, tuning, and refit notes.
- `docs/implementation/thesis-implementation_260624.md`: implementation status and algorithm notes.
- `docs/simulations/thesis-simulation_260615.md`: simulation design, diagnostics, and interpretation.
- `docs/methods/thesis-code-rossi-eta_260615.md`: Rossi/Eta code-level notes.

## Current Research Position

The main claim is not that Eta-group greatly increases ARI. The safer claim is that centered eta group penalization can preserve clustering performance while producing a sparser, more interpretable support inside a vMF mixture framework.

Weak concentration settings remain a diagnostic limitation: recent checks point to path/BIC instability rather than a simple grid-density or BIC constant-term issue.

## Local-Only Materials

R scripts, raw simulation code, and large generated outputs are maintained locally and are not tracked in Git. Local R scripts are grouped under:

- `r/data_prep/`
- `r/methods/`
- `r/realdata/`
- `r/simulation/`

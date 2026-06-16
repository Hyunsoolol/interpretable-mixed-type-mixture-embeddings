# thesis-hp-clustering

Research notes for a sparse directional clustering thesis project based on vMF mixtures and eta-contrast variable selection.

## Current Repository Scope

This repository is currently kept as a document-first research meeting archive.

- R source code is intentionally kept local and excluded from Git for security.
- Public-facing materials are limited to thesis notes, meeting summaries, and selected result summaries.
- The previous README content was archived in `README_previous_260616.md`.

## Key Documents

- `thesis-meeting_260622.md`: meeting-ready summary and current decisions.
- `thesis-implementation_260622.md`: implementation status and algorithm notes.
- `thesis-simulation_260615.md`: simulation design, diagnostics, and interpretation.
- `thesis-code-rossi-eta_260615.md`: Rossi/Eta code-level notes.

## Current Research Position

The main claim is not that Eta penalty greatly increases ARI. The safer claim is that eta-contrast penalization can preserve clustering performance while producing a sparser, more interpretable support inside a vMF mixture framework.

Weak concentration settings remain a diagnostic limitation: recent checks point to path/BIC instability rather than a simple grid-density or BIC constant-term issue.

## Local-Only Materials

R scripts, raw simulation code, and large generated outputs are maintained locally and are not tracked in Git.

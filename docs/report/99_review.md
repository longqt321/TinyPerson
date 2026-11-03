# Review notes

## Checks completed

- Compared primary tiled AP/MR values in draft against raw 150e baseline and DCAF JSON.
- Compared full-frame AP values against raw full-frame JSON.
- Kept validation metrics separate from standalone test metrics.
- Draft identifies single-seed evidence and does not claim statistical significance.
- Excluded QAC results and unverified external citations.

## Open issues

- Raw test JSON evaluates `last.pt`; project guidance prefers `best.pt`.
- Evaluator says standalone implementation, 786 images, dense exclusions 0. Official protocol equivalence unverified.
- No multi-seed statistics or controlled ablations isolating DCAF v2.1 mechanisms.
- No verified primary literature yet; Related Work remains category-level placeholder.
- Venue-specific page limit/template and requested deliverable format not specified; current output is Markdown technical report.
- Figure and generated table scripts not added yet; draft tables checked directly against source JSON and remain reproducible manually, not automatically.

## Outcome

Draft is usable as an evidence-bounded internal report, not submission-ready paper. Stronger claims require resolving checkpoint/evaluator provenance and completing verified literature review.

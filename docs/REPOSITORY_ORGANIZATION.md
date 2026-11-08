# Repository organization and cleanup review

Reviewed 2026-11-08 at the user's request. The useful immediate cleanup is
navigation and organization of one-off utilities. Historical evidence remains
at its recorded paths.

## Current layout

| Location | Purpose |
|---|---|
| `PROJECT_MASTER_PLAN.md`, `DECISION_LOG.md`, `EVIDENCE_LEDGER.md` | Current authoritative plan, decisions and evidence |
| `docs/waveform_research/` | Current research index, aggregate tables and receipts |
| `docs/project_audit/` | Repository, data-admission and camera/sampling receipts |
| `docs/waveform_v2/` | Historical Gate evidence at pinned paths |
| `rppg_lab/` | Reusable signal, data, model and loss modules |
| `scripts/` | Research/preparation/evaluation entry points and some experimental shared helpers |
| `tests/`, `.github/workflows/` | Software contracts and synthetic CI |
| `outputs/waveform_screen/` | Ignored development arrays, weights, full predictions and figures |
| `outputs/two_source_waveform/` | Ignored linear baseline artifacts |
| `outputs/project_reassessment_20261108/admin_tools/` | Archived one-off audit/download/integration utilities |
| `checkpoints/`, existing raw/cache directories | Local research sources and frozen identities; unchanged |

## Completed organization

- Moved eight just-created administrative helper files into the audit's
  `admin_tools/` directory. Contents were preserved. Logs, merge receipts,
  source manifests, dataset-admission records and experiment results remain.
- Added one [current-research index](waveform_research/README.md) linking protocols,
  results and the reusable experimental pipeline.
- Consolidated findings and the selected candidate's preprocessing/reproduction
  contract in the [campaign report](WAVEFORM_DEVELOPMENT_CAMPAIGN_REPORT.md).
- Audited tracked layout and known generated-cache folders. Raw/protected dataset
  contents were outside the cleanup inventory; no reserved subject was enumerated.

No research data, historical results, checkpoints or tracked files were deleted.
Automatic approval review blocked the proposed recursive deletion of Python and
pytest caches without a more specific reason. Those rebuildable caches were
left intact; normal Python/testing use regenerates them anyway.

## Consolidation that should wait

The saved Gate 09 replay pins 224 inherited source/document files. Moving old
reports, changing old loaders, deleting historical outputs/caches, or relocating
frozen files would break that provenance contract. Large raw directories cannot
be classified as disposable from names; do not bulk-delete or scan them.

Four tracked `rppg_lab.egg-info` files are generated package metadata, but belong
to the inherited snapshot and editable-install behavior. Defer removal to an
intentional packaging migration with compatibility checks.

Experimental helpers can later move from command scripts into one shared
development engine after comparisons settle. This is a routine refactor
opportunity rather than a reason to rename recorded sources mid-study. Existing
version-pinned experiments need no historical file deletion to stay usable.

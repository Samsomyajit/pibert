# Artifact Bundles

This branch includes the final figure exports, supplementary figure bundles, and lightweight result summaries for the latest RealPDEBench update.

## Included

- `FIGURE/`: manuscript-ready final figure exports.
- `figures_rpb_final/`: cylinder-real final and supplementary figure bundle.
- `figures_rpb_fsi_real_final_v2/`: FSI-real final and supplementary figure bundle.
- `runs_rpb_final/`: cylinder-real result summaries (`test_metrics.json`, `train_history.json`, `train.log`, `tune_summary.json`).
- `runs_rpb_fsi_real_baselines_v3_fast/`: FSI-real baseline result summaries.
- `runs_rpb_fsi_real_pibert_v3_fast/`: FSI-real PIBERT result summaries.
- `scripts/`: figure regeneration helpers used for the released bundles.

## Excluded

- Manuscript sources and paper-format outputs such as `paper.md`, LaTeX build files, DOCX submissions, and reviewer-response documents.
- Raw checkpoints and tensor dumps such as `best.pt`, `predictions.npz`, and `norm_stats.npz`.

Several raw model artifacts exceed GitHub's 100 MB per-file limit, so this branch keeps the publishable figures and summary results only. If the full raw bundle needs to be distributed, use GitHub Releases, Git LFS, or external object storage.

The figure regeneration scripts expect those raw local artifacts. If you want to rerun them from this repository checkout, restore the omitted `best.pt` and `predictions.npz` files in the same directory structure first.

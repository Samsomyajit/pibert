PIBERT vs Baselines on CFDBench Cylinder — Full Pack
====================================================
See README in earlier message for full usage. Key steps:

1) pip install -U huggingface_hub
2) python src/get_cfdb_cylinder.py --subset bc --out cfdb_cylinder_npz
3) python src/runner.py --config config.json
4) Plot: python helpers/plot_matplotlib.py --npz results/PIBERT/sample_000.npz --field uvmag --save figures/pibert_uvmag.png

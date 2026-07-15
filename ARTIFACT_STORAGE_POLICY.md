# Artifact storage policy

This repository tracks source code, configurations, dependency declarations,
change logs, and lightweight reproducibility reports.

Datasets, OpenAI CLIP weights, feature caches, training checkpoints, and other
large generated artifacts remain on the formal experiment server because they
exceed normal GitHub repository limits. Their paths, identities, and SHA256
manifests must be recorded in experiment reports so that every result remains
auditable without duplicating multi-gigabyte files in Git.

The formal server project is:

`/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI`

Historical exploratory result directories are protected from overwrite:

- `outputs/server_mkan_reproduction/`
- `outputs/server_full_wo_kd/`
- `outputs/server_full_logits_kd/`


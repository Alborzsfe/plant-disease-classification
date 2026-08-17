# GitHub publication guide

## File placement

- Put all executable Python files in `src/`.
- Put the downloaded source data in `data/raw/PlantVillage/`.
- Let `prepare_data.py` create `data/processed/`.
- Let `train.py` create the `.keras` model in `models/` and history artifacts in
  `results/`.
- Keep dataset images, model binaries, and generated result files untracked unless
  you intentionally change `.gitignore`.
- The `notebooks/` directory currently contains only an explanation because no
  notebook was provided. Do not add a notebook that disagrees with `src/train.py`.

## Commands to publish

Create an empty GitHub repository first, then run from this project root:

```bash
git init
git status
git add .
git status
git commit -m "Create reproducible plant disease classification project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/plant-disease-classification.git
git push -u origin main
```

If the directory is already a Git repository, do not run `git init`. If `origin`
already exists, inspect it with `git remote -v` rather than adding it again.

## Pre-publication checklist

- [ ] Verify and cite the exact PlantVillage download source, version, and license.
- [ ] Choose a repository license and add its complete license text.
- [ ] Confirm that every source class folder is correctly mapped by the binary rule.
- [ ] Check for duplicate images and train/validation leakage.
- [ ] Run `python src/check_data.py` and record the real class counts.
- [ ] Train from a clean environment and save the environment versions.
- [ ] Run `python src/evaluate.py`; inspect the confusion matrix and metrics JSON.
- [ ] Replace README result placeholders only with generated, verified metrics.
- [ ] Clearly label validation metrics; ideally add a separate held-out test set.
- [ ] Test inference on several images with `src/predict.py`.
- [ ] Decide where the model artifact will be hosted and add its checksum/link.
- [ ] Confirm `git status` does not list dataset images, secrets, or large model files.
- [ ] Remove personal paths, names, credentials, and notebook outputs if present.
- [ ] Confirm README commands work after cloning into a different directory.

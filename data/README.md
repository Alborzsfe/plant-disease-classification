# Dataset setup

This repository does not include the dataset.

The supplied project identifies the source data as **PlantVillage**, but it does
not record the exact download URL, release, license, or original split source.
Add the verified official source and license here before publishing the project.

Place the downloaded data in this layout:

```text
data/raw/PlantVillage/
├── train/
│   ├── <original PlantVillage class folder>/
│   └── ...
└── val/
    ├── <original PlantVillage class folder>/
    └── ...
```

Then run:

```bash
python src/prepare_data.py
python src/check_data.py
```

`prepare_data.py` preserves the existing project behavior: any source directory
whose name contains `healthy` (case-insensitive) is mapped to `healthy`; every
other source directory is mapped to `diseased`. It copies rather than moves files
and prefixes each filename with its original class directory to reduce collisions.

Important: the script expects pre-existing `train` and `val` source splits. It
does not create a random or stratified split.

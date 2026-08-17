# Model artifacts

Trained model files are intentionally excluded from normal Git tracking.

The repository's current training command writes:

```text
models/plant_disease_mobilenetv2.keras
```

For a public release, use one of these approaches and record the exact model
version and checksum in this file:

- attach the model to a GitHub Release;
- track it with Git LFS; or
- publish it in a model repository such as Hugging Face Hub.

The native `.keras` format is used for newly trained models because it is the
recommended whole-model format for modern Keras. The supplied legacy `.h5`
artifact is not committed automatically and is discussed in `PROJECT_AUDIT.md`.

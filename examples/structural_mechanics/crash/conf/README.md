# Configuration Layout

## Start here: `experiments/`

**This is the main folder you interact with.** Each YAML file is a self-contained experiment config. Run training or inference by selecting one:

```bash
python train.py --config-name=experiments/bumper_geotransolver
python inference.py --config-name=experiments/crash_geotransolver
```

To add a new experiment, copy an existing file in `experiments/` and edit data paths, model, and features.

---

## Component configs (advanced)

The `components/` folder contains configs referenced by experiments. You rarely need to edit them unless customizing models, readers, or training defaults.

| Path                     | Purpose                                      |
|--------------------------|----------------------------------------------|
| `components/model/`      | Model architectures (selected via experiment) |
| `components/datapipe/`   | Dataset and feature configs                  |
| `components/reader/`    | Data format readers (VTP, Zarr)              |
| `components/training/`   | Training hyperparameters                     |
| `components/inference/`  | Inference options                            |

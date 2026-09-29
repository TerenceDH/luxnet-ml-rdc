# Dataset

The full dataset is provided separately through the GitHub release. A small runnable subset is included in `data/sample/` for checking the code.

Each scene contains the following files:

| Folder | Content |
|---|---|
| `A_float` | Three-channel model input: luminaire position, room mask, and window projection |
| `cond_vec` | Room, window, luminaire, and material parameters |
| `B_float` | Horizontal electric illuminance target in lux |
| `meta_json` | Readable scene information |

The image input and illuminance target use a 128 × 128 grid. The condition vector contains:

```text
room height
luminaire mounting height
sill height
window height
bottom, right, top, and left wall reflectance
floor reflectance
ceiling reflectance
window reflectance
```

After downloading the release asset, extract it with:

```bash
python scripts/extract_dataset.py path/to/luxnet_dataset.zip
```

The resulting files can be used directly by `train.py` and `evaluate.py` with the provided split file.


# XRD Pattern Identification

This repository contains code for data processing, clustering, model definition, active learning, and domain shift analysis, as well as trained checkpoint files.
It is designed to efficiently identify phases from perovskite-type XRD patterns with good transferability.

The key methodological components are:
1. **Physics-motivated data augmentation** — simulation of strain, texture,
   crystallite-size broadening, impurity phases, and peak shifts
   (`Data/data_augment/`), adapted from
   [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer).
2. **Clustering** — hierarchical clustering of simulated
   patterns with the Wasserstein distance to build a 177-class taxonomy (`Clustering/`).
3. **Active learning** — BADGE sampling combined with Optuna hyper-parameter
   search to reduce training cost (`Active_Learning/`).
4. **Cross-database transfer learning** — models fine-tuned for Materials Project
   (MP), ICSD, COD, and their overlap (`Model/Checkpoints/`).
5. **Domain-shift analysis** — UMAP + per-dimension Wasserstein distance to
   quantify the shift between source and target domains (`Domain shift/`).

---

## Repository structure

```
.
├── Model/                          # Shared model definition + trained weights
│   ├── basemodel.py                #   XRDNet / XRDDataset / CustomDropout
│   └── Checkpoints/                #   trained checkpoints
├── Sample/                         # Inference demo on real RRUFF spectra
│   ├── test_trained_model.py       #   Predict compounds from .npy spectra
│   └── RRUFF_processed/            #   real CaTiO3 patterns (ready to predict)
├── Domain shift/                   # Domain-shift analysis
│   └── domain_shift_analysis.py
├── Clustering/                     # Hierarchical clustering -> label taxonomy
│   ├── xrd_clustering.py           #   Clustering entry point
│   ├── pure_peak.py                #   Pure-peak helper module
│   └── compound_cluster_summary.csv  #   177-class label map (pre-computed)
├── Data/                           # Data preparation pipeline
│   ├── tabulate_Perovskite.py      #   CIF standardization entry point
│   ├── tabulate_cifs_all/          #   CIF standardization + de-duplication
│   ├── augment.py                  #   Data-augmentation entry point
│   ├── data_augment/               #   Augmentation package (6 physical effects)
│   └── build_xrd_dataset.py        #   Merge / split / label the final datasets
├── comp_label_map/                 # Compound -> label maps for each model
│   ├── icsd_comp_labels.csv        #   ICSD (5203 classes)
│   ├── cod_comp_labels.csv         #   COD  (2610 classes)
│   └── o84_comp_labels.csv         #   overlap (84 classes)
└── Active_Learning/
    └── active_learning_optuna_3.8.py
```

---

## Requirements

Python 3.9+ (developed with 3.11).

The installation can be done quickly with the following statement.

```bash
pip install -r requirements.txt
```

---

## Quick start: inference on real samples

The simplest way to verify the environment is to run inference on the three
real RRUFF spectra shipped in `Sample/RRUFF_processed/`.

```bash
cd Sample
python test_trained_model.py
```

By default this evaluates the **MP model** (177 classes). To switch models,
edit `DATASET_NAME` in `test_trained_model.py` to one of `"mp"`, `"icsd"`,
`"cod"`, or `"overlap"`.

Outputs are written to `Sample/experiments_report/real_test/<model>/`:

| File | Content |
|------|---------|
| `predicted_compounds.json` | Top-5 predicted compounds per spectrum |
| `predicted_compounds.csv`  | Same result in CSV format |

---

## Models

All checkpoints share the same XRDNet architecture and hyper-parameters. The
canonical hyper-parameters are stored in `Model/Checkpoints/basemodel_177.pth`
under the `best_params` key; the ICSD/COD checkpoints contain only weights, so
the hyper-parameters are always read from `basemodel_177.pth`.

| Checkpoint | Classes | Description |
|-----------|--------:|-------------|
| `basemodel_177.pth` | 177 | MP base model (also the hyper-parameter source) |
| `icsd_best_5203.pth` | 5203 | ICSD transfer model |
| `cod_best_2610.pth` | 2610 | COD transfer model |
| `best_model_overlap84.pth` | 84 | MP/COD/ICSD overlap model |

The corresponding label maps live in `comp_label_map/` (for icsd / cod /
overlap) and `Clustering/compound_cluster_summary.csv` (for mp).

---

## Data preparation pipeline

The pipeline turns raw CIF files into labeled, augmented `.npy` datasets. The
three stages run in order:

### 1. Standardize CIF files

```bash
cd Data
python tabulate_Perovskite.py
```

`tabulate_Perovskite.py` is a thin wrapper over the `tabulate_cifs_all` package:

```python
import tabulate_cifs_all as tc
tc.main('./icsd_perov', 'icsd_perov_all')   # input CIF dir -> output dir
```

Edit the input/output directories for each database (`mp_perovskite_raw`,
`icsd_perov_raw`, `cod_perovskite_raw`). The routine normalizes the CIF files
and removes duplicates by grouping structures by (reduced formula, space group).

### 2. Augment the spectra

```bash
cd Data
python augment.py
```

`augment.py` instantiates `data_augment.SpectraGenerator` to simulate spectra
with controlled physical artifacts (see `data_augment/__init__.py` for the
constructor arguments):

```python
xrd_obj = data_augment.SpectraGenerator(
    './icsd_perov_all', 'aug_icsd_all', num_spectra=50,
    max_texture=0.5, min_domain_size=5.0, max_domain_size=30.0,
    max_strain=0.03, max_shift=0.5, impur_amt=70.0,
    min_angle=10.0, max_angle=80.0, batch_size=100, separate=True,
)
xrd_obj.generate_and_save()
```

The `data_augment` package implements six independent effects:
`uniform_shifts`, `strain_shifts`, `peak_broadening`, `intensity_changes`,
`impurity_peaks`, and `mixed`.

### 3. Build the labeled dataset

```bash
cd Data
python build_xrd_dataset.py --input-dir aug_mp --output-dir val_test \
    --split-counts 80 10 10 --label-source csv \
    --label-csv ../Clustering/compound_cluster_summary.csv
```

`build_xrd_dataset.py` merges the per-compound `.npy` files into aligned
samples/labels, optionally splitting each file into train/val/test subsets.
Key options:

- `--split-counts TRAIN VAL TEST` — per-file split; omit to extract everything.
- `--label-source sorted|csv` — label by sorted name or by a CSV map.
- `--label-format onehot|index` — one-hot vectors or integer indices.

---

## Clustering (label taxonomy)

The 177-class label system is produced by hierarchical clustering:

```bash
cd Clustering
python xrd_clustering.py --cif_dir ../mp_perov_cif6 --outdir results_by_tau
```

`xrd_clustering.py` simulates one pattern per CIF, computes the pairwise
Wasserstein distance matrix, and applies agglomerative clustering. It writes
`compound_cluster_summary.csv` (the 177-class `Compound -> cluster_id` map used
everywhere else) and `cluster_quality_details.csv` (per-cluster quality
metrics). Pass `--recompute-dist` to regenerate the (large) distance matrix.

---

## Active learning

```bash
cd Active_Learning
python active_learning_optuna_3.8.py --weight_search --select_way class \
    --rounds 20 --seed 42 --save_dir ./exp_class
```

The script performs BADGE-based active learning over the labeled dataset, with
an optional Optuna hyper-parameter search (`--weight_search`). It loads
`../data_set/X_{train,val,test}.npy` and `../val_test/*.npy`, which are the
outputs of the data-preparation pipeline above. After a search, re-run without
`--weight_search` and point `--param_path` at the saved best hyper-parameters.

---

## Domain-shift analysis

```bash
cd "Domain shift"
python domain_shift_analysis.py
```

This script quantifies the domain shift between the MP source domain and a
COD/ICSD target domain. Set `TARGET_DOMAIN` and the `MP_DIR` / `TARGET_DIR`
data directories in the `CONFIG` section before running. It writes, to the
current working directory:

- `domain_shift_scores_{cod|icsd}.csv` — per-class shift scores (core result);
- `model_accuracy_{cod|icsd}.csv` — overall / MP / target-domain accuracy;
- `model_accuracy_by_class_{cod|icsd}.csv` — per-class target-domain accuracy;
- `domain_shift_a_global_umap.png/.pdf` — global UMAP scatter plot.

The model backbone is imported from `Model/basemodel.py` and the weights from
`Model/Checkpoints/basemodel_177.pth`.

---

## Notes

The raw CIF files and intermediate datasets are not included in this
repository; adjust the data directories (marked in each script's config / CLI)
to your local paths before running the corresponding stage.

---

## Cite

If you find this code useful for your research, please consider citing it. The associated paper is currently under review; the DOI will be added here upon publication. 

---

## Contact

For bugs or questions, please contact 2024222030032@stu.scu.edu.cn.
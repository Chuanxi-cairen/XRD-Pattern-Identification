#!/usr/bin/env python3
"""
Build merged NumPy datasets from per-compound augmented .npy files.

Supported modes
---------------
1. Full extraction mode:
   Load every sample from every compound file and save one sample file and one
   label file.

2. Per-file split mode:
   Split every compound file independently into train/validation/test subsets,
   then merge the corresponding subsets across all compounds.

Label sources
-------------
1. sorted:
   Sort compound names with Python's default sorted() and assign labels
   0, 1, 2, ... in that order.

2. csv:
   Read compound-to-label mappings from a CSV file, for example
   compound_cluster_summary.csv with columns Compound and cluster_id.

The script always keeps samples and labels in the same order and writes mapping
and manifest CSV files for auditing.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompoundFile:
    """One compound and its source .npy file."""

    compound_name: str
    path: Path


@dataclass
class CompoundManifestRow:
    """Audit information for one compound file."""

    compound_name: str
    class_index: int
    source_file: str
    total_available: int
    selected_count: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-compound .npy files, optionally split each file into "
            "train/validation/test subsets, and generate aligned labels."
        )
    )

    # Input and output paths.
    parser.add_argument(
        "--input-dir",
        default="aug_mp",
        help="Directory containing per-compound NumPy files.",
    )
    parser.add_argument(
        "--output-dir",
        default="val_test",
        help="Directory in which generated datasets are saved.",
    )
    parser.add_argument(
        "--file-suffix",
        default="_aug.npy",
        help=(
            "Only files ending with this suffix are loaded. The suffix is "
            "removed to obtain the compound name. Default: _aug.npy"
        ),
    )

    # Optional per-file split.
    parser.add_argument(
        "--split-counts",
        nargs=3,
        type=int,
        metavar=("TRAIN", "VAL", "TEST"),
        help=(
            "Number of samples taken from every compound file for train, "
            "validation, and test sets. Omit this option to extract all samples."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        choices=("random", "sequential"),
        default="random",
        help=(
            "How samples inside each compound file are selected in split mode. "
            "Default: random."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for per-file splitting and optional output shuffling.",
    )
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help=(
            "Synchronously shuffle merged samples and labels before saving. "
            "By default, compounds remain grouped in deterministic name order."
        ),
    )

    # Label source and format.
    parser.add_argument(
        "--label-source",
        choices=("sorted", "csv"),
        default="sorted",
        help=(
            "sorted: assign labels after sorting compound names; "
            "csv: use labels from a CSV mapping file. Default: sorted."
        ),
    )
    parser.add_argument(
        "--label-csv",
        default="compound_cluster_summary.csv",
        help="CSV mapping file used when --label-source csv.",
    )
    parser.add_argument(
        "--compound-column",
        default="Compound",
        help="Compound-name column in the label CSV. Default: Compound.",
    )
    parser.add_argument(
        "--id-column",
        default="cluster_id",
        help="Integer label column in the label CSV. Default: cluster_id.",
    )
    parser.add_argument(
        "--label-format",
        choices=("onehot", "index"),
        default="onehot",
        help="Save labels as one-hot vectors or integer indices. Default: onehot.",
    )

    # Output file names for full extraction mode.
    parser.add_argument(
        "--sample-name",
        default="samples.npy",
        help="Sample filename used when --split-counts is omitted.",
    )
    parser.add_argument(
        "--label-name",
        default="labels.npy",
        help="Label filename used when --split-counts is omitted.",
    )

    # Output file names for split mode.
    parser.add_argument(
        "--train-sample-name",
        default="train_samples.npy",
        help="Training-sample filename in split mode.",
    )
    parser.add_argument(
        "--train-label-name",
        default="train_labels.npy",
        help="Training-label filename in split mode.",
    )
    parser.add_argument(
        "--val-sample-name",
        default="val_samples.npy",
        help="Validation-sample filename in split mode.",
    )
    parser.add_argument(
        "--val-label-name",
        default="val_labels.npy",
        help="Validation-label filename in split mode.",
    )
    parser.add_argument(
        "--test-sample-name",
        default="test_samples.npy",
        help="Test-sample filename in split mode.",
    )
    parser.add_argument(
        "--test-label-name",
        default="test_labels.npy",
        help="Test-label filename in split mode.",
    )

    # Audit files.
    parser.add_argument(
        "--mapping-name",
        default="compound_label_mapping.csv",
        help="Filename for the compound-to-label mapping CSV.",
    )
    parser.add_argument(
        "--manifest-name",
        default="dataset_manifest.csv",
        help="Filename for the per-compound extraction manifest CSV.",
    )
    parser.add_argument(
        "--config-name",
        default="dataset_build_config.json",
        help="Filename for the saved run configuration JSON.",
    )

    return parser.parse_args()


def ensure_suffix(filename: str, suffix: str) -> str:
    """Append a required suffix when it is missing."""
    return filename if filename.lower().endswith(suffix.lower()) else filename + suffix


def validate_output_names(args: argparse.Namespace) -> None:
    """Reject duplicate output names that would overwrite one another."""
    if args.split_counts is None:
        names = [
            ensure_suffix(args.sample_name, ".npy"),
            ensure_suffix(args.label_name, ".npy"),
            ensure_suffix(args.mapping_name, ".csv"),
            ensure_suffix(args.manifest_name, ".csv"),
            ensure_suffix(args.config_name, ".json"),
        ]
    else:
        names = [
            ensure_suffix(args.train_sample_name, ".npy"),
            ensure_suffix(args.train_label_name, ".npy"),
            ensure_suffix(args.val_sample_name, ".npy"),
            ensure_suffix(args.val_label_name, ".npy"),
            ensure_suffix(args.test_sample_name, ".npy"),
            ensure_suffix(args.test_label_name, ".npy"),
            ensure_suffix(args.mapping_name, ".csv"),
            ensure_suffix(args.manifest_name, ".csv"),
            ensure_suffix(args.config_name, ".json"),
        ]

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Output filenames must be unique. Duplicate names: "
            + ", ".join(duplicates)
        )


def discover_compound_files(input_dir: Path, file_suffix: str) -> List[CompoundFile]:
    """Discover matching files and return them in compound-name order."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not file_suffix:
        raise ValueError("--file-suffix cannot be empty.")

    discovered: List[CompoundFile] = []
    seen: Dict[str, Path] = {}

    for path in input_dir.iterdir():
        if not path.is_file() or not path.name.endswith(file_suffix):
            continue

        compound_name = path.name[: -len(file_suffix)]
        if not compound_name:
            raise ValueError(f"Cannot extract compound name from file: {path.name}")
        if compound_name in seen:
            raise ValueError(
                "Duplicate compound name extracted from multiple files: "
                f"{compound_name!r}\n  - {seen[compound_name]}\n  - {path}"
            )

        seen[compound_name] = path
        discovered.append(CompoundFile(compound_name=compound_name, path=path))

    if not discovered:
        raise FileNotFoundError(
            f"No files ending with {file_suffix!r} were found in {input_dir}."
        )

    # This is the same default Python string ordering used by sorted(compound_names).
    discovered.sort(key=lambda item: item.compound_name)
    return discovered


def normalize_csv_labels(
    csv_path: Path,
    compound_column: str,
    id_column: str,
) -> Dict[str, int]:
    """Read and strictly validate compound-to-label mappings from CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Label CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing_columns = [
        column
        for column in (compound_column, id_column)
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Label CSV is missing required columns {missing_columns}. "
            f"Available columns: {list(frame.columns)}"
        )

    selected = frame[[compound_column, id_column]].copy()
    if selected[compound_column].isna().any():
        bad_rows = selected.index[selected[compound_column].isna()].tolist()[:10]
        raise ValueError(f"Compound column contains missing values at rows: {bad_rows}")
    if selected[id_column].isna().any():
        bad_rows = selected.index[selected[id_column].isna()].tolist()[:10]
        raise ValueError(f"Label column contains missing values at rows: {bad_rows}")

    selected[compound_column] = selected[compound_column].astype(str).str.strip()
    if (selected[compound_column] == "").any():
        bad_rows = selected.index[selected[compound_column] == ""].tolist()[:10]
        raise ValueError(f"Compound column contains empty names at rows: {bad_rows}")

    numeric_labels = pd.to_numeric(selected[id_column], errors="coerce")
    if numeric_labels.isna().any():
        bad_values = selected.loc[numeric_labels.isna(), id_column].head(10).tolist()
        raise ValueError(
            "Label column must contain integer values. Invalid examples: "
            f"{bad_values}"
        )

    numeric_array = numeric_labels.to_numpy(dtype=float)
    rounded = np.rint(numeric_array)
    if not np.allclose(numeric_array, rounded):
        bad_mask = ~np.isclose(numeric_array, rounded)
        bad_values = numeric_array[bad_mask][:10].tolist()
        raise ValueError(
            "Label column contains non-integer numeric values. Examples: "
            f"{bad_values}"
        )

    integer_labels = rounded.astype(np.int64)
    if np.any(integer_labels < 0):
        bad_values = integer_labels[integer_labels < 0][:10].tolist()
        raise ValueError(f"Labels must be non-negative. Invalid examples: {bad_values}")
    selected[id_column] = integer_labels

    # Duplicate rows with the same compound and same label are harmless. Conflicting
    # labels for the same compound are not.
    conflicts = (
        selected.groupby(compound_column, sort=False)[id_column]
        .nunique(dropna=False)
    )
    conflicting_names = conflicts[conflicts > 1].index.tolist()
    if conflicting_names:
        examples = conflicting_names[:10]
        raise ValueError(
            "The label CSV assigns multiple different labels to the same compound. "
            f"Examples: {examples}"
        )

    deduplicated = selected.drop_duplicates(subset=[compound_column], keep="first")
    return {
        str(row[compound_column]): int(row[id_column])
        for _, row in deduplicated.iterrows()
    }


def build_compound_labels(
    compounds: Sequence[CompoundFile],
    label_source: str,
    label_csv: Path,
    compound_column: str,
    id_column: str,
) -> Tuple[Dict[str, int], int]:
    """Create the compound-to-class mapping and determine label-vector width."""
    compound_names = [item.compound_name for item in compounds]

    if label_source == "sorted":
        # compounds are already in sorted compound-name order.
        mapping = {name: index for index, name in enumerate(compound_names)}
        return mapping, len(compound_names)

    csv_mapping = normalize_csv_labels(label_csv, compound_column, id_column)

    missing_from_csv = [name for name in compound_names if name not in csv_mapping]
    if missing_from_csv:
        preview = "\n".join(f"  - {name}" for name in missing_from_csv[:20])
        more = "" if len(missing_from_csv) <= 20 else f"\n  ... and {len(missing_from_csv) - 20} more"
        raise KeyError(
            "Some input compounds are missing from the label CSV:\n"
            + preview
            + more
        )

    extra_in_csv = sorted(set(csv_mapping) - set(compound_names))
    if extra_in_csv:
        print(
            f"Warning: {len(extra_in_csv)} compounds exist in the label CSV but "
            "have no matching input file. They will be ignored."
        )

    mapping = {name: int(csv_mapping[name]) for name in compound_names}
    unique_labels = sorted(set(mapping.values()))
    n_classes = max(unique_labels) + 1

    expected = list(range(n_classes))
    if unique_labels != expected:
        missing_ids = sorted(set(expected) - set(unique_labels))
        print(
            "Warning: CSV labels are not contiguous from 0. One-hot labels will "
            f"have width max_label + 1 = {n_classes}, with unused columns: {missing_ids}."
        )

    return mapping, n_classes


def validate_split_counts(split_counts: Optional[Sequence[int]]) -> Optional[Tuple[int, int, int]]:
    """Validate and normalize TRAIN/VAL/TEST counts."""
    if split_counts is None:
        return None

    train_count, val_count, test_count = (int(value) for value in split_counts)
    if min(train_count, val_count, test_count) < 0:
        raise ValueError("Split counts must be non-negative integers.")
    if train_count + val_count + test_count <= 0:
        raise ValueError("At least one split count must be greater than zero.")
    return train_count, val_count, test_count


def inspect_array(
    path: Path,
    expected_feature_shape: Optional[Tuple[int, ...]],
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Load one array and validate its sample and feature dimensions."""
    array = np.load(path, allow_pickle=False)
    if array.ndim < 2:
        raise ValueError(
            f"Expected at least two dimensions (samples, features...), but "
            f"{path.name} has shape {array.shape}."
        )
    if array.shape[0] <= 0:
        raise ValueError(f"Source file has no samples: {path}")

    feature_shape = tuple(int(value) for value in array.shape[1:])
    if expected_feature_shape is not None and feature_shape != expected_feature_shape:
        raise ValueError(
            "All source files must have identical dimensions after the sample axis.\n"
            f"Expected: {expected_feature_shape}\n"
            f"Found in {path.name}: {feature_shape}"
        )

    return array, feature_shape


def make_label_block(
    class_index: int,
    count: int,
    n_classes: int,
    label_format: str,
) -> np.ndarray:
    """Create labels aligned with one selected sample block."""
    if label_format == "index":
        return np.full(count, class_index, dtype=np.int64)

    labels = np.zeros((count, n_classes), dtype=np.float32)
    if count > 0:
        labels[:, class_index] = 1.0
    return labels


def concatenate_nonempty(arrays: Sequence[np.ndarray], description: str) -> np.ndarray:
    """Concatenate a list and give a clear error for an empty requested split."""
    if not arrays:
        raise ValueError(f"No arrays were collected for {description}.")
    return np.concatenate(arrays, axis=0)


def synchronized_shuffle(
    samples: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle samples and labels with the same permutation."""
    if len(samples) != len(labels):
        raise ValueError(
            f"Cannot shuffle misaligned arrays: {len(samples)} samples and "
            f"{len(labels)} labels."
        )
    order = rng.permutation(len(samples))
    return samples[order], labels[order]


def save_npy(output_dir: Path, filename: str, array: np.ndarray) -> Path:
    """Save a NumPy array and return its path."""
    path = output_dir / ensure_suffix(filename, ".npy")
    np.save(path, array)
    return path


def build_full_dataset(
    compounds: Sequence[CompoundFile],
    compound_to_label: Mapping[str, int],
    n_classes: int,
    label_format: str,
    shuffle_output: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[CompoundManifestRow], Tuple[int, ...]]:
    """Load every sample from every file and merge them."""
    sample_blocks: List[np.ndarray] = []
    label_blocks: List[np.ndarray] = []
    manifest: List[CompoundManifestRow] = []
    expected_feature_shape: Optional[Tuple[int, ...]] = None

    for item in compounds:
        data, expected_feature_shape = inspect_array(item.path, expected_feature_shape)
        count = int(data.shape[0])
        class_index = int(compound_to_label[item.compound_name])

        sample_blocks.append(data)
        label_blocks.append(
            make_label_block(class_index, count, n_classes, label_format)
        )
        manifest.append(
            CompoundManifestRow(
                compound_name=item.compound_name,
                class_index=class_index,
                source_file=str(item.path.resolve()),
                total_available=count,
                selected_count=count,
            )
        )

    samples = concatenate_nonempty(sample_blocks, "the full dataset")
    labels = concatenate_nonempty(label_blocks, "the full label set")

    if len(samples) != len(labels):
        raise RuntimeError(
            f"Internal alignment error: {len(samples)} samples but {len(labels)} labels."
        )

    if shuffle_output:
        samples, labels = synchronized_shuffle(
            samples,
            labels,
            np.random.default_rng(seed + 1),
        )

    assert expected_feature_shape is not None
    return samples, labels, manifest, expected_feature_shape


def build_split_datasets(
    compounds: Sequence[CompoundFile],
    compound_to_label: Mapping[str, int],
    n_classes: int,
    label_format: str,
    split_counts: Tuple[int, int, int],
    split_strategy: str,
    shuffle_output: bool,
    seed: int,
) -> Tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    List[CompoundManifestRow],
    Tuple[int, ...],
]:
    """Split every source file independently and merge matching subsets."""
    train_count, val_count, test_count = split_counts
    required_count = train_count + val_count + test_count

    sample_blocks: Dict[str, List[np.ndarray]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    label_blocks: Dict[str, List[np.ndarray]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    manifest: List[CompoundManifestRow] = []
    expected_feature_shape: Optional[Tuple[int, ...]] = None
    split_rng = np.random.default_rng(seed)

    for item in compounds:
        data, expected_feature_shape = inspect_array(item.path, expected_feature_shape)
        available = int(data.shape[0])
        if available < required_count:
            raise ValueError(
                f"Insufficient samples in {item.path.name}: available={available}, "
                f"required={required_count} ({train_count}+{val_count}+{test_count})."
            )

        if split_strategy == "random":
            selected_indices = split_rng.permutation(available)[:required_count]
        else:
            selected_indices = np.arange(required_count, dtype=np.int64)

        train_end = train_count
        val_end = train_count + val_count
        train_indices = selected_indices[:train_end]
        val_indices = selected_indices[train_end:val_end]
        test_indices = selected_indices[val_end:required_count]

        split_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
        split_sizes = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }
        class_index = int(compound_to_label[item.compound_name])

        for split_name in ("train", "val", "test"):
            count = split_sizes[split_name]
            if count == 0:
                continue
            sample_blocks[split_name].append(data[split_indices[split_name]])
            label_blocks[split_name].append(
                make_label_block(class_index, count, n_classes, label_format)
            )

        manifest.append(
            CompoundManifestRow(
                compound_name=item.compound_name,
                class_index=class_index,
                source_file=str(item.path.resolve()),
                total_available=available,
                selected_count=required_count,
                train_count=train_count,
                val_count=val_count,
                test_count=test_count,
            )
        )

    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    output_shuffle_rng = np.random.default_rng(seed + 1)

    for split_name, count in (
        ("train", train_count),
        ("val", val_count),
        ("test", test_count),
    ):
        if count == 0:
            # A zero-sized split is valid but is intentionally not saved.
            continue

        samples = concatenate_nonempty(
            sample_blocks[split_name], f"the {split_name} sample set"
        )
        labels = concatenate_nonempty(
            label_blocks[split_name], f"the {split_name} label set"
        )
        if len(samples) != len(labels):
            raise RuntimeError(
                f"Internal alignment error in {split_name}: {len(samples)} samples "
                f"but {len(labels)} labels."
            )

        if shuffle_output:
            samples, labels = synchronized_shuffle(samples, labels, output_shuffle_rng)
        datasets[split_name] = (samples, labels)

    assert expected_feature_shape is not None
    return datasets, manifest, expected_feature_shape


def save_mapping_csv(
    output_dir: Path,
    filename: str,
    compounds: Sequence[CompoundFile],
    compound_to_label: Mapping[str, int],
) -> Path:
    """Save one row per input compound in deterministic compound-name order."""
    rows = [
        {
            "compound_name": item.compound_name,
            "class_index": int(compound_to_label[item.compound_name]),
            "source_file": item.path.name,
        }
        for item in compounds
    ]
    path = output_dir / ensure_suffix(filename, ".csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_manifest_csv(
    output_dir: Path,
    filename: str,
    rows: Sequence[CompoundManifestRow],
    split_mode: bool,
) -> Path:
    """Save extraction counts for every compound."""
    frame = pd.DataFrame([asdict(row) for row in rows])
    if split_mode:
        columns = [
            "compound_name",
            "class_index",
            "source_file",
            "total_available",
            "selected_count",
            "train_count",
            "val_count",
            "test_count",
        ]
    else:
        columns = [
            "compound_name",
            "class_index",
            "source_file",
            "total_available",
            "selected_count",
        ]
    path = output_dir / ensure_suffix(filename, ".csv")
    frame.loc[:, columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_config_json(
    output_dir: Path,
    filename: str,
    args: argparse.Namespace,
    n_compounds: int,
    n_classes: int,
    feature_shape: Tuple[int, ...],
) -> Path:
    """Save the resolved run configuration for reproducibility."""
    config = vars(args).copy()
    config.update(
        {
            "resolved_input_dir": str(Path(args.input_dir).resolve()),
            "resolved_output_dir": str(output_dir.resolve()),
            "n_compounds": int(n_compounds),
            "n_classes": int(n_classes),
            "feature_shape": list(feature_shape),
        }
    )
    path = output_dir / ensure_suffix(filename, ".json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return path


def print_saved_array(description: str, path: Path, array: np.ndarray) -> None:
    print(f"Saved {description}: {path} | shape={array.shape} | dtype={array.dtype}")


def main() -> None:
    args = parse_args()
    validate_output_names(args)
    split_counts = validate_split_counts(args.split_counts)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compounds = discover_compound_files(input_dir, args.file_suffix)
    print(f"Discovered {len(compounds)} compound files in: {input_dir.resolve()}")

    compound_to_label, n_classes = build_compound_labels(
        compounds=compounds,
        label_source=args.label_source,
        label_csv=Path(args.label_csv),
        compound_column=args.compound_column,
        id_column=args.id_column,
    )
    print(
        f"Label source: {args.label_source}; compounds={len(compounds)}; "
        f"label-vector width/classes={n_classes}; format={args.label_format}"
    )

    mapping_path = save_mapping_csv(
        output_dir,
        args.mapping_name,
        compounds,
        compound_to_label,
    )

    if split_counts is None:
        samples, labels, manifest, feature_shape = build_full_dataset(
            compounds=compounds,
            compound_to_label=compound_to_label,
            n_classes=n_classes,
            label_format=args.label_format,
            shuffle_output=args.shuffle_output,
            seed=args.seed,
        )

        sample_path = save_npy(output_dir, args.sample_name, samples)
        label_path = save_npy(output_dir, args.label_name, labels)
        print_saved_array("samples", sample_path, samples)
        print_saved_array("labels", label_path, labels)
    else:
        datasets, manifest, feature_shape = build_split_datasets(
            compounds=compounds,
            compound_to_label=compound_to_label,
            n_classes=n_classes,
            label_format=args.label_format,
            split_counts=split_counts,
            split_strategy=args.split_strategy,
            shuffle_output=args.shuffle_output,
            seed=args.seed,
        )

        output_names = {
            "train": (args.train_sample_name, args.train_label_name),
            "val": (args.val_sample_name, args.val_label_name),
            "test": (args.test_sample_name, args.test_label_name),
        }
        for split_name in ("train", "val", "test"):
            if split_name not in datasets:
                print(f"Skipping {split_name}: requested count is 0 for every compound.")
                continue
            samples, labels = datasets[split_name]
            sample_name, label_name = output_names[split_name]
            sample_path = save_npy(output_dir, sample_name, samples)
            label_path = save_npy(output_dir, label_name, labels)
            print_saved_array(f"{split_name} samples", sample_path, samples)
            print_saved_array(f"{split_name} labels", label_path, labels)

    manifest_path = save_manifest_csv(
        output_dir,
        args.manifest_name,
        manifest,
        split_mode=split_counts is not None,
    )
    config_path = save_config_json(
        output_dir,
        args.config_name,
        args,
        n_compounds=len(compounds),
        n_classes=n_classes,
        feature_shape=feature_shape,
    )

    print(f"Saved compound-label mapping: {mapping_path}")
    print(f"Saved dataset manifest: {manifest_path}")
    print(f"Saved run configuration: {config_path}")
    print("Dataset construction completed successfully.")


if __name__ == "__main__":
    main()

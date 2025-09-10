import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _can_stratify(labels: pd.Series) -> bool:
    """Return True if labels have at least 2 samples per class for a split."""
    value_counts = labels.value_counts()
    return (value_counts >= 2).all()


def _split_one_group(
    group_df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a single group's rows into train/val/test using stratification when possible.

    The split is performed in two steps to achieve exact ratios:
      1) train vs temp
      2) temp -> val vs test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    if len(group_df) == 0:
        return group_df.copy(), group_df.copy(), group_df.copy()

    # First split: train vs temp
    temp_ratio = 1.0 - train_ratio
    stratify_labels = None
    if _can_stratify(group_df[label_col]):
        stratify_labels = group_df[label_col]

    train_df, temp_df = train_test_split(
        group_df,
        test_size=temp_ratio,
        stratify=stratify_labels,
        random_state=random_state,
        shuffle=True,
    )

    if len(temp_df) == 0:
        return train_df, temp_df, temp_df

    # Second split: temp -> val vs test
    # Proportion of val within temp: val_ratio / (val_ratio + test_ratio)
    val_within_temp = val_ratio / (val_ratio + test_ratio)
    stratify_labels_temp = None
    if _can_stratify(temp_df[label_col]):
        stratify_labels_temp = temp_df[label_col]

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_within_temp,
        stratify=stratify_labels_temp,
        random_state=random_state,
        shuffle=True,
    )

    return train_df, val_df, test_df


def split_by_dog_id(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    random_state: int,
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset per Dog ID, preserving class distribution where feasible."""
    rng = np.random.RandomState(random_state)
    all_train: List[pd.DataFrame] = []
    all_val: List[pd.DataFrame] = []
    all_test: List[pd.DataFrame] = []

    # Shuffle group order for fairness
    unique_ids: List = df[id_col].dropna().unique().tolist()
    rng.shuffle(unique_ids)

    # Print all behavior labels
    print(f"All behavior labels: {df[label_col].unique()}")

    grouped: Dict[object, pd.DataFrame] = {dog_id: g for dog_id, g in df.groupby(id_col)}

    for dog_id in unique_ids:
        group_df = grouped.get(dog_id, pd.DataFrame(columns=df.columns))
        train_df, val_df, test_df = _split_one_group(
            group_df=group_df,
            label_col=label_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )
        all_train.append(train_df)
        all_val.append(val_df)
        all_test.append(test_df)

    return (
        pd.concat(all_train, axis=0, ignore_index=True),
        pd.concat(all_val, axis=0, ignore_index=True),
        pd.concat(all_test, axis=0, ignore_index=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a CSV dataset into train/val/test (64/16/20) per Dog ID, "
            "preserving class distribution where possible."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write output CSVs (defaults to input file directory)",
    )
    parser.add_argument(
        "--id-col",
        default="DogID",
        help="Column name for Dog ID (default: DogID)",
    )
    parser.add_argument(
        "--label-col",
        default="Behavior_1",
        help="Column name for class label (default: Behavior_1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    required_cols = [args.id_col, args.label_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}. Present columns: {list(df.columns)}"
        )

    train_df, val_df, test_df = split_by_dog_id(
        df=df,
        id_col=args.id_col,
        label_col=args.label_col,
        random_state=args.seed,
        train_ratio=0.64,
        val_ratio=0.16,
        test_ratio=0.20,
    )

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    train_path = os.path.join(output_dir, f"{base_name}_train.csv")
    val_path = os.path.join(output_dir, f"{base_name}_val.csv")
    test_path = os.path.join(output_dir, f"{base_name}_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Simple summary
    print("Saved splits:")
    print(f"  Train: {len(train_df)} rows -> {train_path}")
    print(f"  Val:   {len(val_df)} rows -> {val_path}")
    print(f"  Test:  {len(test_df)} rows -> {test_path}")


if __name__ == "__main__":
    main()



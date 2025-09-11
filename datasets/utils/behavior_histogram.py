import argparse
import os
import sys
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def resolve_default_input_path() -> Optional[str]:
    """
    Try to resolve a sensible default input CSV path relative to this script
    (../DogMoveData.csv). Returns None if it does not exist.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, "..", "DogMoveData.csv"))
    return candidate if os.path.isfile(candidate) else None


def parse_args() -> argparse.Namespace:
    default_input = resolve_default_input_path()

    parser = argparse.ArgumentParser(
        description=(
            "Plot a histogram (bar chart) of frequency distribution for dog behavior classes."
        )
    )
    parser.add_argument(
        "--input",
        default=default_input,
        help=(
            "Path to input CSV file. Defaults to ../DogMoveData.csv relative to this script "
            "if present."
        ),
    )
    parser.add_argument(
        "--label-col",
        default="Behavior_1",
        help="Column name containing behavior class labels (default: Behavior_1)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to save the histogram image (PNG). Defaults to <input>_behavior_hist.png."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the image",
    )
    parser.add_argument(
        "--sort",
        choices=["count", "label"],
        default="count",
        help="Sort bars by 'count' (desc) or by 'label' (asc). Default: count",
    )
    return parser.parse_args()


def derive_output_path(input_path: str, explicit_output: Optional[str]) -> str:
    if explicit_output:
        return explicit_output
    base_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(base_dir, f"{base_name}_behavior_hist.png")


def main() -> None:
    args = parse_args()

    input_path = args.input
    if not input_path:
        print("--input is required and no default CSV was found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(2)

    label_col = args.label_col
    if label_col not in df.columns:
        print(
            f"Label column '{label_col}' not found. Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(3)

    counts = df[label_col].value_counts(dropna=False)

    if args.sort == "label":
        counts = counts.sort_index()
    else:
        counts = counts.sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(counts.index.astype(str), counts.values, color="#4C78A8")
    plt.title("Dog Behavior Class Frequency")
    plt.xlabel("Behavior Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = derive_output_path(input_path, args.output)
    try:
        plt.savefig(output_path, dpi=150)
        print(f"Saved histogram to '{output_path}'.")
    except Exception as e:
        print(f"Failed to save image: {e}", file=sys.stderr)
        # Still allow showing if requested

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()




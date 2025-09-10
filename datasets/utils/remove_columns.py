import argparse
import csv
import os
import sys
import tempfile
from typing import Iterable, List


def _normalize_column_list(columns: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for item in columns:
        # allow comma-separated or repeated flags
        parts = [p.strip() for p in item.split(",")]
        normalized.extend([p for p in parts if p])
    # preserve order, remove duplicates while keeping first occurrence
    seen = set()
    unique_ordered: List[str] = []
    for col in normalized:
        if col not in seen:
            seen.add(col)
            unique_ordered.append(col)
    return unique_ordered


def remove_columns_from_csv(
    input_path: str,
    output_path: str,
    columns_to_remove: List[str],
) -> int:
    """Remove specified columns from a CSV. Returns number of data rows written."""
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)

        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row or is empty.")

        available = reader.fieldnames
        to_remove = [c for c in columns_to_remove if c in available]
        missing = [c for c in columns_to_remove if c not in available]

        if not to_remove:
            raise KeyError(
                f"None of the specified columns were found. Available columns: {available}"
            )

        remaining = [c for c in available if c not in to_remove]

        with open(output_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=remaining)
            writer.writeheader()

            rows_written = 0
            for row in reader:
                # Write only remaining columns
                writer.writerow({k: row.get(k, "") for k in remaining})
                rows_written += 1

    if missing:
        print(
            f"Warning: columns not found and thus not removed: {missing}", file=sys.stderr
        )

    return rows_written


def derive_output_path(input_path: str, explicit_output: str | None, inplace: bool) -> str:
    if inplace:
        fd, temp_path = tempfile.mkstemp(
            prefix="cols_removed_",
            suffix=os.path.splitext(input_path)[1],
            dir=os.path.dirname(input_path) or None,
        )
        os.close(fd)
        return temp_path

    if explicit_output:
        return explicit_output

    stem, ext = os.path.splitext(input_path)
    return f"{stem}_cols_removed{ext or '.csv'}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove specified columns from a CSV file."
    )
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write the modified CSV (default: <input>_cols_removed.csv)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input CSV in place",
    )
    parser.add_argument(
        "--columns",
        "-c",
        nargs="+",
        required=True,
        help="Column names to remove (space or comma separated).",
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    columns = _normalize_column_list(args.columns)
    if not columns:
        print("No valid column names provided.", file=sys.stderr)
        sys.exit(1)

    try:
        output_path = derive_output_path(input_path, args.output, args.inplace)
        rows_written = remove_columns_from_csv(
            input_path=input_path,
            output_path=output_path,
            columns_to_remove=columns,
        )

        if args.inplace:
            os.replace(output_path, input_path)
            final_path = input_path
        else:
            final_path = output_path

        print(
            f"Removed columns {columns}. Wrote {rows_written} rows to '{final_path}'."
            + (" (overwrote input)" if args.inplace else "")
        )
    except KeyError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()



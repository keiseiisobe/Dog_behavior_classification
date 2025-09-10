import argparse
import csv
import os
import sys
import tempfile


def filter_csv_rows(
    input_path: str,
    output_path: str,
    column_name: str = "Behavior_1",
    undefined_value: str = "<undefined>",
) -> int:
    """Read a CSV and write only rows whose column != undefined_value.

    Returns the number of rows written to the output file.
    """
    rows_written = 0

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f_in, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)

        if column_name not in reader.fieldnames if reader.fieldnames else []:
            raise KeyError(
                f"Column '{column_name}' not found. Available columns: {reader.fieldnames}"
            )

        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            value = row.get(column_name)
            if value != undefined_value:
                writer.writerow(row)
                rows_written += 1

    return rows_written


def derive_output_path(input_path: str, explicit_output: str | None, inplace: bool) -> str:
    if inplace:
        # We'll write to a temp file first and then atomically replace input.
        # The caller will handle the replacement.
        fd, temp_path = tempfile.mkstemp(
            prefix="cleaned_", suffix=os.path.splitext(input_path)[1], dir=os.path.dirname(input_path) or None
        )
        os.close(fd)
        return temp_path

    if explicit_output:
        return explicit_output

    stem, ext = os.path.splitext(input_path)
    return f"{stem}_cleaned{ext or '.csv'}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove rows from a CSV where a specified column equals '<undefined>'."
        )
    )
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write the cleaned CSV (default: <input>_cleaned.csv)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Edit the CSV in place (overwrites the input file)",
    )
    parser.add_argument(
        "--column",
        default="Behavior_1",
        help="Column name to check (default: 'Behavior_1')",
    )
    parser.add_argument(
        "--value",
        default="<undefined>",
        help="Value considered undefined (default: '<undefined>')",
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        output_path = derive_output_path(input_path, args.output, args.inplace)
        rows_written = filter_csv_rows(
            input_path=input_path,
            output_path=output_path,
            column_name=args.column,
            undefined_value=args.value,
        )

        if args.inplace:
            # Atomically replace the input file
            os.replace(output_path, input_path)
            final_path = input_path
        else:
            final_path = output_path

        print(
            f"Wrote {rows_written} rows to '{final_path}'."
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



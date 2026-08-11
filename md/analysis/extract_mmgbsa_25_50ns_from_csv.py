#!/usr/bin/env python3

"""
Extract 25-50 ns MM/GBSA Delta-energy statistics from independent MD replicates.

Expected directory structure under --analysis-root:

analysis_root/
├── mmgbsa_replicates/
│   ├── aptC_rep1/FINAL_RESULTS.csv
│   ├── aptC_rep2/FINAL_RESULTS.csv
│   ├── aptC_rep3/FINAL_RESULTS.csv
│   ├── APT3_rep1/FINAL_RESULTS.csv
│   ├── ...
│   └── APT4_rep3/FINAL_RESULTS.csv
└── results/

The script extracts frames 2501-5001 from the Delta Energy Terms section,
corresponding to the 25-50 ns post-equilibration analysis window used in
the manuscript. Each replicate contributes 501 sampled frames.

Replicate-level means are calculated first. Final mean and sample SD are
then calculated across the three independent trajectories (n=3).
"""

from pathlib import Path
import argparse
import csv
import statistics


SYSTEMS = ["aptC", "APT3", "APT4"]

WANTED_TERMS = [
    "VDWAALS",
    "EEL",
    "EGB",
    "ESURF",
    "GGAS",
    "GSOLV",
    "TOTAL",
]


def read_delta(path: Path):
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    start = None

    for i, line in enumerate(lines):
        if "delta energy terms" in line.strip().lower():
            start = i + 1
            break

    if start is None:
        raise RuntimeError(
            f"Delta Energy Terms section not found: {path}"
        )

    while (
        start < len(lines)
        and not lines[start].strip().startswith("Frame #")
    ):
        start += 1

    if start >= len(lines):
        raise RuntimeError(
            f"Delta header not found: {path}"
        )

    header = next(csv.reader([lines[start]]))
    idx = {
        name: j
        for j, name in enumerate(header)
    }

    required = ["Frame #"] + WANTED_TERMS
    missing = [
        name
        for name in required
        if name not in idx
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns {missing}: {path}"
        )

    rows = []

    for line in lines[start + 1:]:
        if not line.strip():
            if rows:
                break
            continue

        try:
            row = next(csv.reader([line]))
            frame = int(
                float(row[idx["Frame #"]])
            )
        except Exception:
            if rows:
                break
            continue

        if 2501 <= frame <= 5001:
            rows.append(
                (
                    frame,
                    {
                        term: float(row[idx[term]])
                        for term in WANTED_TERMS
                    },
                )
            )

    if len(rows) != 501:
        raise RuntimeError(
            f"Expected 501 sampled frames, "
            f"got {len(rows)}: {path}"
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract replicate-level and n=3 summary "
            "MM/GBSA statistics from the 25-50 ns window."
        )
    )

    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing "
            "mmgbsa_replicates/"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for output TSV files. "
            "Default: <analysis-root>/results"
        ),
    )

    args = parser.parse_args()

    root = args.analysis_root.expanduser().resolve()
    src = root / "mmgbsa_replicates"

    if args.output_dir is None:
        output_dir = root / "results"
    else:
        output_dir = (
            args.output_dir
            .expanduser()
            .resolve()
        )

    if not src.is_dir():
        raise FileNotFoundError(
            f"MM/GBSA replicate directory not found: {src}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_rep = (
        output_dir
        / "MMGBSA_25_50ns_replicates.tsv"
    )

    out_sum = (
        output_dir
        / "MMGBSA_25_50ns_summary.tsv"
    )

    rep_rows = []

    by_system = {
        system: {
            term: []
            for term in WANTED_TERMS
        }
        for system in SYSTEMS
    }

    for system in SYSTEMS:
        for replicate in (1, 2, 3):

            path = (
                src
                / f"{system}_rep{replicate}"
                / "FINAL_RESULTS.csv"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Input file not found: {path}"
                )

            rows = read_delta(path)

            means = {
                term: statistics.fmean(
                    frame_values[term]
                    for _, frame_values in rows
                )
                for term in WANTED_TERMS
            }

            rep_rows.append(
                [
                    system,
                    f"rep{replicate}",
                    str(len(rows)),
                ]
                + [
                    f"{means[term]:.4f}"
                    for term in WANTED_TERMS
                ]
            )

            for term in WANTED_TERMS:
                by_system[system][term].append(
                    means[term]
                )

    with out_rep.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle,
            delimiter="\t",
        )

        writer.writerow(
            [
                "system",
                "replicate",
                "n_frames",
            ]
            + WANTED_TERMS
        )

        writer.writerows(rep_rows)

    with out_sum.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle,
            delimiter="\t",
        )

        writer.writerow(
            [
                "system",
                "n_replicates",
            ]
            + [
                f"{term}_mean"
                for term in WANTED_TERMS
            ]
            + [
                f"{term}_SD"
                for term in WANTED_TERMS
            ]
        )

        for system in SYSTEMS:

            means = [
                statistics.fmean(
                    by_system[system][term]
                )
                for term in WANTED_TERMS
            ]

            sds = [
                statistics.stdev(
                    by_system[system][term]
                )
                for term in WANTED_TERMS
            ]

            writer.writerow(
                [system, 3]
                + [
                    f"{value:.4f}"
                    for value in means
                ]
                + [
                    f"{value:.4f}"
                    for value in sds
                ]
            )

    print(
        "===== 25-50 ns replicate means "
        "(Delta section, 501 frames each) ====="
    )
    print(out_rep.read_text())

    print("===== n=3 summary =====")
    print(out_sum.read_text())

    print(f"SAVED: {out_rep}")
    print(f"SAVED: {out_sum}")


if __name__ == "__main__":
    main()

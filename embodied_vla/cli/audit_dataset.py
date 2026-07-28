from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_vla.data import audit_expert_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit an expert trajectory dataset.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = audit_expert_dataset(
        args.dataset,
        write_statistics=not args.no_write,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

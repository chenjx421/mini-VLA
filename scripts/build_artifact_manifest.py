from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_ROOTS = (
    Path("checkpoints"),
    Path("datasets/so_arm_pick_place_v2_120_dr"),
    Path("datasets/dagger_v1_seed30000_beta050"),
    Path("datasets/dagger_v2_seed35000_beta020"),
    Path("docs/assets"),
    Path("results"),
)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(project_root: Path, output_path: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    absolute_output = output_path.resolve()
    artifacts: list[dict[str, object]] = []

    for relative_root in DEFAULT_ROOTS:
        artifact_root = project_root / relative_root
        if not artifact_root.exists():
            continue
        candidates = (candidate for candidate in artifact_root.rglob("*") if candidate.is_file())
        for path in sorted(candidates):
            if path.resolve() == absolute_output:
                continue
            relative_path = path.relative_to(project_root).as_posix()
            artifacts.append(
                {
                    "path": relative_path,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    return {
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(item["bytes"]) for item in artifacts),
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hash promoted datasets and result artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/artifact_manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    manifest = build_manifest(project_root, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {manifest['artifact_count']} artifacts "
        f"({manifest['total_bytes']} bytes) to {output_path}"
    )


if __name__ == "__main__":
    main()

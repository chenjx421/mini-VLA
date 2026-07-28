from __future__ import annotations

from pathlib import Path

from scripts.build_artifact_manifest import build_manifest


def test_manifest_is_sorted_and_excludes_its_own_output(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    result_dir = tmp_path / "results"
    checkpoint_dir.mkdir()
    result_dir.mkdir()
    (checkpoint_dir / "z.pt").write_bytes(b"weights-z")
    (checkpoint_dir / "a.pt").write_bytes(b"weights-a")
    output_path = result_dir / "artifact_manifest.json"
    output_path.write_text("old manifest", encoding="utf-8")

    manifest = build_manifest(tmp_path, output_path)

    paths = [artifact["path"] for artifact in manifest["artifacts"]]
    assert paths == ["checkpoints/a.pt", "checkpoints/z.pt"]
    assert manifest["artifact_count"] == 2
    assert manifest["total_bytes"] == len(b"weights-a") + len(b"weights-z")
    assert all(len(artifact["sha256"]) == 64 for artifact in manifest["artifacts"])

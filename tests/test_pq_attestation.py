from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_jp_judge.pq_attestation import (
    ATTESTATION_FILE_NAME,
    ISSUER,
    create_attestation,
    generate_key_pair,
    verify_attestation,
)


def _make_results(output_dir: Path) -> None:
    (output_dir / "score_table.json").write_text(json.dumps([{"model": "test", "quality": 0.81}]))
    (output_dir / "summary.json").write_text(json.dumps({"quality": 0.81, "safety": 0.95}))


@pytest.mark.parametrize("algorithm", ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"])
def test_attestation_round_trip(tmp_path: Path, algorithm: str) -> None:
    _make_results(tmp_path)
    public_key, private_key = generate_key_pair(algorithm)

    attestation = create_attestation(tmp_path, algorithm=algorithm, key_pair=(public_key, private_key))
    assert attestation["issuer"] == ISSUER
    assert attestation["algorithm"] == algorithm
    assert set(attestation["files"]) == {"score_table.json", "summary.json"}
    assert (tmp_path / ATTESTATION_FILE_NAME).exists()

    assert verify_attestation(tmp_path).valid
    assert verify_attestation(tmp_path, expected_public_key=public_key).valid


def test_detects_tampered_results(tmp_path: Path) -> None:
    _make_results(tmp_path)
    create_attestation(tmp_path)

    (tmp_path / "summary.json").write_text(json.dumps({"quality": 0.99, "safety": 0.95}))

    assert not verify_attestation(tmp_path).valid


def test_rejects_wrong_signer(tmp_path: Path) -> None:
    _make_results(tmp_path)
    create_attestation(tmp_path)

    other_public_key, _ = generate_key_pair()
    assert not verify_attestation(tmp_path, expected_public_key=other_public_key).valid


def test_requires_result_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        create_attestation(tmp_path)

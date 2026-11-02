"""Post-quantum attestation of evaluation results.

llm-jp-judge writes its judge results as one JSON file per table in an output
directory. These helpers sign the digests of those files with ML-DSA (FIPS 204)
and write a self-contained ``attestation.json`` beside them, so a published set
of judge verdicts can be shown to be genuine and unaltered, even against an
adversary with a quantum computer.

Integrity is always checked. Authenticity, that the attestation came from a
specific signer, holds when the verifier pins the signer's public key via
``expected_public_key`` using a key the evaluator has published.

This tooling is contributed by Quantova (https://quantova.org). Signing uses the
optional ``dilithium-py`` dependency; install the ``attestation`` extra to enable
it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

ISSUER = "Quantova"
ATTESTATION_FILE_NAME = "attestation.json"
DEFAULT_ALGORITHM = "ML-DSA-65"
SUPPORTED_ALGORITHMS = ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87")


def _load_ml_dsa(algorithm: str) -> Any:
    try:
        from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        msg = (
            "Post-quantum attestation requires the optional 'dilithium-py' dependency. "
            "Install llm-jp-judge with the 'attestation' extra, e.g. `pip install llm-jp-judge[attestation]`."
        )
        raise ImportError(msg) from e

    table = {"ML-DSA-44": ML_DSA_44, "ML-DSA-65": ML_DSA_65, "ML-DSA-87": ML_DSA_87}
    if algorithm not in table:
        msg = f"Unsupported attestation algorithm: {algorithm}. Supported: {', '.join(SUPPORTED_ALGORITHMS)}"
        raise ValueError(msg)
    return table[algorithm]


def generate_key_pair(algorithm: str = DEFAULT_ALGORITHM) -> tuple[bytes, bytes]:
    """Generate an ML-DSA key pair. Returns ``(public_key, private_key)`` as raw bytes."""
    ml_dsa = _load_ml_dsa(algorithm)
    public_key, private_key = ml_dsa.keygen()
    return public_key, private_key


def _result_digests(output_dir: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in sorted(output_dir.glob("*.json")):
        if path.name == ATTESTATION_FILE_NAME:
            continue
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _signing_payload(algorithm: str, digests: dict[str, str]) -> bytes:
    body = {"algorithm": algorithm, "files": digests}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class VerificationResult:
    """The outcome of verifying an attestation."""

    valid: bool
    reason: str


def create_attestation(
    output_dir: str | PathLike[str],
    algorithm: str = DEFAULT_ALGORITHM,
    key_pair: tuple[bytes, bytes] | None = None,
) -> dict[str, Any]:
    """Create and write a post-quantum attestation for a result directory."""
    output_dir = Path(output_dir)
    ml_dsa = _load_ml_dsa(algorithm)

    digests = _result_digests(output_dir)
    if not digests:
        msg = f"No result JSON files to attest were found in {output_dir}"
        raise FileNotFoundError(msg)

    if key_pair is None:
        public_key, private_key = ml_dsa.keygen()
    else:
        public_key, private_key = key_pair

    payload = _signing_payload(algorithm, digests)
    signature = ml_dsa.sign(private_key, payload)

    attestation = {
        "version": 1,
        "issuer": ISSUER,
        "algorithm": algorithm,
        "created_at": int(time.time()),
        "files": digests,
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }

    with open(output_dir / ATTESTATION_FILE_NAME, "w") as f:
        json.dump(attestation, f, indent=4)

    return attestation


def verify_attestation(
    output_dir: str | PathLike[str],
    expected_public_key: bytes | None = None,
) -> VerificationResult:
    """Verify the attestation written in ``output_dir``."""
    output_dir = Path(output_dir)
    attestation_path = output_dir / ATTESTATION_FILE_NAME
    if not attestation_path.exists():
        return VerificationResult(False, f"No {ATTESTATION_FILE_NAME} found in {output_dir}")

    with open(attestation_path) as f:
        attestation = json.load(f)

    algorithm = attestation.get("algorithm")
    if algorithm not in SUPPORTED_ALGORITHMS:
        return VerificationResult(False, f"Unsupported or missing algorithm: {algorithm}")
    ml_dsa = _load_ml_dsa(algorithm)

    recorded = attestation.get("files", {})
    current = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in recorded}
    if current != recorded:
        return VerificationResult(False, "Result files do not match the recorded digests")

    try:
        public_key = base64.b64decode(attestation["public_key"])
        signature = base64.b64decode(attestation["signature"])
    except (KeyError, ValueError):
        return VerificationResult(False, "Malformed attestation fields")

    if expected_public_key is not None and public_key != expected_public_key:
        return VerificationResult(False, "Attestation public key does not match the expected signer")

    payload = _signing_payload(algorithm, recorded)
    valid = ml_dsa.verify(public_key, payload, signature)
    return VerificationResult(valid, "Attestation is valid" if valid else "Signature verification failed")

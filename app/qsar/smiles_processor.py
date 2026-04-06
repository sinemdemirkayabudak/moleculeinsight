"""Utilities for loading and processing EGFR inhibitor SMILES structures."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_egfr_compounds() -> list[dict[str, Any]]:
    """Load EGFR inhibitor compounds from JSON data file.

    Returns:
        List of compound dictionaries with properties and SMILES.

    Raises:
        FileNotFoundError: If data file not found.
        json.JSONDecodeError: If JSON is malformed.
    """
    data_path: Path = Path(__file__).parent.parent / "data" / "egfr_inhibitors.json"

    if not data_path.exists():
        raise FileNotFoundError(f"EGFR inhibitors data not found: {data_path}")

    try:
        with open(data_path) as f:
            data: dict[str, Any] = json.load(f)
        logger.info(f"Loaded {len(data['compounds'])} EGFR inhibitors from {data_path}")
        return data["compounds"]
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise


def get_all_smiles(compounds: list[dict[str, Any]]) -> dict[str, str]:
    """Extract name -> SMILES mapping.

    Args:
        compounds: List of compound dictionaries.

    Returns:
        Dictionary mapping compound names to SMILES strings.
    """
    return {compound["name"]: compound["smiles"] for compound in compounds}


def get_cached_bioactivity(
    compounds: list[dict[str, Any]], compound_name: str
) -> list[dict[str, Any]] | None:
    """Get cached bioactivity data for a compound.

    Returns pre-loaded bioactivity data from JSON file to avoid API calls.
    Data is sanitized to ensure all values are strings (prevents PyArrow type errors).
    Searches by either compound name or SMILES string.

    Args:
        compounds: List of compound dictionaries from JSON.
        compound_name: Name of compound or SMILES string to retrieve bioactivity for.

    Returns:
        List of sanitized bioactivity records (all values as strings) or None if not found/cached.
    """
    for compound in compounds:
        # Check by name
        if compound.get("name", "").lower() == compound_name.lower():
            bioactivity = compound.get("bioactivity")
            # Sanitize to ensure all values are strings
            return _sanitize_bioactivity(bioactivity) if bioactivity else None
        # Check by SMILES
        if compound.get("smiles", "") == compound_name:
            bioactivity = compound.get("bioactivity")
            # Sanitize to ensure all values are strings
            return _sanitize_bioactivity(bioactivity) if bioactivity else None
    return None


def _sanitize_bioactivity(bioactivity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure all bioactivity values are strings to prevent PyArrow conversion errors.

    Args:
        bioactivity: List of bioactivity records from JSON.

    Returns:
        List of bioactivity records with all string values.
    """
    if not bioactivity:
        return []

    sanitized = []
    for record in bioactivity:
        sanitized_record = {
            "target_chembl_id": str(record.get("target_chembl_id", "N/A")),
            "target_name": str(record.get("target_name", "N/A")),
            "activity_type": str(record.get("activity_type", "N/A")),
            "value": str(record.get("value", "N/A")),
            "units": str(record.get("units", "")),
            "assay_description": str(record.get("assay_description", "")),
            "pubmed_id": str(record.get("pubmed_id", "")) if record.get("pubmed_id") else "",
            "selection_rationale": str(record.get("selection_rationale", ""))
            if record.get("selection_rationale")
            else "",
        }
        sanitized.append(sanitized_record)

    return sanitized

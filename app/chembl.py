import pandas as pd
import streamlit as st

from app.config import CHEMBL_BASE_URL, logger
from app.pubchem import get_pubchem_metadata
from app.utils import get_response_json


@st.cache_data(ttl=86400)
def get_chembl_target_id(target_name: str) -> str | None:
    """
    Fetch ChEMBL target ID by target name.

    Queries ChEMBL target.json endpoint to resolve target name to ID.
    Returns the first (most relevant) match. Cached for 24 hours.

    Parameters
    ----------
    target_name : str
        Target protein ChEMBL preferred name
        (e.g., "Epidermal growth factor receptor")

    Returns
    -------
    str or None
        ChEMBL target ID (e.g., "CHEMBL203") or None if not found

    Examples
    --------
    >>> target_id = get_chembl_target_id("Epidermal growth factor receptor")
    >>> print(target_id)
    CHEMBL203
    """

    try:
        logger.info(f"Fetching ChEMBL target ID for: {target_name}")

        url = f"{CHEMBL_BASE_URL}/target.json"
        params = {
            "pref_name__iexact": target_name,
            "limit": 1,
        }

        data = get_response_json(url, params)

        if not data:
            logger.warning(f"No response from ChEMBL target endpoint for {target_name}")
            return None

        targets = data.get("targets", [])

        if not targets:
            logger.warning(f"No target found for name: {target_name}")
            return None

        target_id = targets[0].get("target_chembl_id")
        logger.info(f"Resolved {target_name} → {target_id}")

        return target_id

    except Exception as e:
        logger.exception(f"Error fetching target ID for {target_name}: {e}")
        return None


@st.cache_data(ttl=86400)
def get_chembl_molecule(inchikey: str) -> dict:
    """
    Resolve ChEMBL molecule entry using InChIKey.
    Returns molecule metadata including ChEMBL ID.
    Cached for 1 hour to avoid redundant API calls.
    """

    try:
        logger.info(f"Searching ChEMBL molecule for InChIKey: {inchikey}")
        url = f"{CHEMBL_BASE_URL}/molecule.json"
        params = {"molecule_structures__standard_inchi_key": inchikey}

        data = get_response_json(url, params)

        if not data:
            logger.warning("No response from ChEMBL molecule endpoint")
            return {"success": False, "message": "No ChEMBL match found"}

        molecules = data.get("molecules", [])

        if not molecules:
            logger.warning(f"No ChEMBL match found for InChIKey: {inchikey}")
            return {"success": False, "message": "No ChEMBL match found"}

        # Take the highest-confidence match returned by ChEMBL
        molecule = molecules[0]

        chembl_id = molecule.get("molecule_chembl_id")

        logger.info(f"ChEMBL molecule found: {chembl_id}")

        return {
            "success": True,
            "chembl_id": chembl_id,
            "pref_name": molecule.get("pref_name"),
            "molecule_type": molecule.get("molecule_type"),
            "max_phase": molecule.get("max_phase"),
        }

    except Exception as e:
        logger.exception(f"ChEMBL molecule lookup error: {e}")
        return {"success": False, "error": str(e)}


@st.cache_data(ttl=86400)
def get_chembl_bioactivity(
    molecule_chembl_id: str | None = None,
    target_chembl_id: str | None = None,
    standard_type: str | None = None,
    standard_relation: str | None = None,
    standard_units: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """
    Retrieve bioactivity data from ChEMBL.

    Queries activity.json endpoint for either a specific molecule or target.
    Supports optional filtering by bioactivity measurement type, units, etc.
    Cached for 24 hours to avoid redundant API calls.

    Parameters
    ----------
    molecule_chembl_id : str, optional
        ChEMBL molecule ID to fetch activities for (e.g., "CHEMBL25")
    target_chembl_id : str, optional
        ChEMBL target ID to fetch activities for (e.g., "CHEMBL203")
    standard_type : str, optional
        Bioactivity measurement type (e.g., "IC50"). If None, returns all types.
    standard_relation : str, optional
        Relationship operator (e.g., "=" for exact values). Only used with target queries.
    standard_units : str, optional
        Unit of measurement (e.g., "nM"). Only used with target queries.
    limit : int
        Number of records per request (default 20 for molecules, can be higher for targets)
    offset : int
        Pagination offset for result set (default 0). Only used with target queries.

    Returns
    -------
    dict
        Dictionary with:
        - 'success' (bool): Query success status
        - 'data' (pd.DataFrame): Activity data with columns:
            - smiles, standard_value, assay_id, reference
        - 'count' (int): Number of records
        - 'activities' (list): For molecule queries, list of cleaned activity records
        - 'error' (str): Error message if failed

    Raises
    ------
    ValueError
        If neither or both molecule_chembl_id and target_chembl_id are provided

    Examples
    --------
    >>> # Molecule-based query (all activities for a compound)
    >>> result = get_chembl_bioactivity(molecule_chembl_id="CHEMBL25", limit=20)

    >>> # Target-based query (all IC50 values for a target)
    >>> result = get_chembl_bioactivity(
    ...     target_chembl_id="CHEMBL203",
    ...     standard_type="IC50",
    ...     standard_units="nM",
    ...     limit=10000
    ... )
    """

    # Validate input: exactly one ID must be provided
    if (molecule_chembl_id is None and target_chembl_id is None) or (
        molecule_chembl_id is not None and target_chembl_id is not None
    ):
        error_msg = "Provide either molecule_chembl_id or target_chembl_id, not both"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    try:
        url = f"{CHEMBL_BASE_URL}/activity.json"
        params = {"limit": limit}

        # Set query parameter based on ID type
        if molecule_chembl_id:
            params["molecule_chembl_id"] = molecule_chembl_id
            logger.info(f"Fetching ChEMBL bioactivity for molecule {molecule_chembl_id}")
        else:
            params["target_chembl_id"] = target_chembl_id
            logger.info(
                f"Fetching {standard_type or 'all'} bioactivity data for target {target_chembl_id}"
            )

            # Add measurement filters for target queries
            if standard_type:
                params["standard_type"] = standard_type
            if standard_relation:
                params["standard_relation"] = standard_relation
            if standard_units:
                params["standard_units"] = standard_units
            if offset > 0:
                params["offset"] = offset

        data = get_response_json(url, params)

        activities = data.get("activities", []) if data else []

        if not activities:
            query_target = molecule_chembl_id or target_chembl_id
            logger.warning(f"No bioactivity records found for {query_target}")
            return {
                "success": False,
                "count": 0,
                "error": "No bioactivity records found",
            }

        # For molecule queries: return cleaned list of activities
        if molecule_chembl_id:
            cleaned = []
            for act in activities:
                cleaned.append(
                    {
                        "target_chembl_id": act.get("target_chembl_id"),
                        "target_name": act.get("target_pref_name"),
                        "standard_type": act.get("standard_type"),
                        "standard_value": act.get("standard_value"),
                        "standard_units": act.get("standard_units"),
                        "assay_description": act.get("assay_description"),
                    }
                )

            logger.info(
                f"Retrieved {len(cleaned)} bioactivity records for molecule {molecule_chembl_id}"
            )

            return {
                "success": True,
                "count": len(cleaned),
                "activities": cleaned,
            }

        # For target queries: return DataFrame with SMILES + values
        else:
            records = []
            for act in activities:
                smiles = act.get("canonical_smiles")
                value = act.get("standard_value")
                assay_id = act.get("assay_chembl_id")
                reference = act.get("target_pref_name")

                # Only include records with SMILES and measurement value
                if smiles and value is not None:
                    records.append(
                        {
                            "smiles": smiles,
                            "standard_value": value,
                            "assay_id": assay_id,
                            "reference": reference,
                        }
                    )

            df = pd.DataFrame(records) if records else pd.DataFrame()

            logger.info(
                f"Retrieved {len(df)} valid records for target {target_chembl_id} "
                f"(total returned: {len(activities)})"
            )

            return {
                "success": True,
                "data": df,
                "count": len(df),
                "total_returned": len(activities),
            }

    except Exception as e:
        logger.exception(f"Error fetching bioactivity data: {e}")
        return {"success": False, "error": str(e)}


def get_compound_bioactivity_from_mol(mol, limit: int = 20):
    try:
        logger.info("Starting compound bioactivity pipeline")

        pubchem_data = get_pubchem_metadata(mol)

        if not pubchem_data.get("success"):
            logger.warning("PubChem resolution failed")
            return {"success": False, "stage": "pubchem"}

        inchikey = pubchem_data.get("inchikey")

        logger.info(f"Resolved InChIKey: {inchikey}")

        chembl_data = get_chembl_molecule(inchikey)

        if not chembl_data.get("success"):
            logger.warning("ChEMBL molecule lookup failed")
            return {"success": False, "stage": "chembl_lookup"}

        chembl_id = chembl_data["chembl_id"]

        bioactivity = get_chembl_bioactivity(molecule_chembl_id=chembl_id, limit=limit)

        logger.info("Bioactivity pipeline completed")

        return {
            "success": True,
            "pubchem": pubchem_data,
            "chembl": chembl_data,
            "bioactivity": bioactivity,
        }

    except Exception as e:
        logger.exception(f"Compound bioactivity pipeline error: {e}")
        return {"success": False, "error": str(e)}

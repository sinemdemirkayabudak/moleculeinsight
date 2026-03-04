import streamlit as st

from app.config import CHEMBL_BASE_URL, logger
from app.pubchem import get_pubchem_metadata
from app.utils import get_response_json


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
def get_chembl_bioactivity(chembl_id: str, limit: int = 20) -> dict:
    """
    Retrieve bioactivity data for a ChEMBL compound.
    Cached for 1 hour to avoid redundant API calls.
    """

    try:
        logger.info(f"Fetching ChEMBL bioactivity for {chembl_id}")
        url = f"{CHEMBL_BASE_URL}/activity.json"
        params = {"molecule_chembl_id": chembl_id, "limit": limit}

        data = get_response_json(url, params)

        activities = data.get("activities", []) if data else []

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

        logger.info(f"Retrieved {len(cleaned)} bioactivity records for {chembl_id}")

        return {"success": True, "count": len(cleaned), "activities": cleaned}

    except Exception as e:
        logger.exception(f"ChEMBL bioactivity retrieval error: {e}")
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

        bioactivity = get_chembl_bioactivity(chembl_id, limit=limit)

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

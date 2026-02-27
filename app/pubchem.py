import re

import pubchempy as pcp
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Mol

from app.config import logger


def get_clean_common_name(synonyms: list[str]) -> str:
    if not synonyms:
        return "Unknown"

    blacklist_patterns = [
        r"\d{2,7}-\d{2}-\d",  # CAS numbers
        r"RefChem",  # RefChem IDs
        r"PubChem",  # PubChem IDs
        r"CID:\d+",  # PubChem CID IDs
        r"^\d+$",  # Pure numeric IDs
    ]

    for syn in synonyms:
        # Skip bad IDs
        if any(re.search(p, syn) for p in blacklist_patterns):
            continue

        # Remove stereo descriptors + punctuation
        clean_name = re.sub(r"\(.*?\)", "", syn)
        clean_name = re.sub(r"[,\-+]", " ", clean_name)

        # Normalize spaces
        clean_name = " ".join(clean_name.split())

        # Remove trailing punctuation ⭐ (THIS FIXES YOUR PROBLEM)
        clean_name = clean_name.strip("-,. ")

        return clean_name.title()

    return "Unknown"


def _get_pubchem_metadata(mol: Mol) -> dict:
    try:
        if mol is None:
            return {"iupac": "Unknown", "common": "Unknown", "success": False}

        smiles = Chem.MolToSmiles(mol, canonical=True)

        compounds = pcp.get_compounds(smiles, namespace="smiles")

        if not compounds:
            return {
                "iupac": "Unknown",
                "common": "Unknown",
                "success": False,
                "query_smiles": smiles,
            }
        # There might be multiple possible matches
        # Take the first (best-ranked, highest confidence) result
        compound = compounds[0]

        iupac = compound.iupac_name or "Unknown"
        synonyms = compound.synonyms or []
        common = get_clean_common_name(synonyms)

        return {
            # 🔹 OLD KEYS (unchanged → no disruption)
            "iupac": iupac,
            "common": common,
            # 🔹 NEW SAFE FIELDS
            "success": True,
            "query_smiles": smiles,
            "cid": compound.cid,
            "inchikey": compound.inchikey,
            "inchi": compound.inchi,
            "molecular_formula": compound.molecular_formula,
            "molecular_weight": compound.molecular_weight,
            "canonical_smiles": compound.canonical_smiles,
            "isomeric_smiles": compound.isomeric_smiles,
            "synonyms": synonyms[:10] if synonyms else [],
        }

    except Exception as e:
        logger.exception(f"Error retrieving PubChem metadata: {e}")
        st.error(str(e))
        return {"iupac": "Error", "common": "Error", "success": False}


@st.cache_data(ttl=86400)
def get_pubchem_metadata(mol: Mol) -> dict[str, str]:
    return _get_pubchem_metadata(mol)

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
            return {"iupac": "Unknown", "common": "Unknown", "cid": None, "inchikey": None, "success": False}

        smiles = Chem.MolToSmiles(mol, canonical=True)

        compounds = pcp.get_compounds(smiles, namespace="smiles")

        if not compounds:
            return {
                "iupac": "Unknown",
                "common": "Unknown",
                "cid": None,
                "inchikey": None,
                "success": False,
            }
        # There might be multiple possible matches
        # Take the first (best-ranked, highest confidence) result
        compound = compounds[0]

        iupac = compound.iupac_name or "Unknown"
        synonyms = compound.synonyms or []
        common = get_clean_common_name(synonyms)

        return {
            "iupac": iupac,
            "common": common,
            "cid": compound.cid,
            "inchikey": compound.inchikey,
            "success": True,
        }

    except Exception as e:
        logger.warning(f"PubChem metadata retrieval error: {e}")
        return {"iupac": "Unknown", "common": "Unknown", "cid": None, "inchikey": None, "success": False}


@st.cache_data(ttl=86400)
@st.cache_data(ttl=86400)
def get_pubchem_metadata(mol: Mol) -> dict[str, str]:
    return _get_pubchem_metadata(mol)

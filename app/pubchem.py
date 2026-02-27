import re

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Mol

from app.config import PUBCHEM_CID_URL, PUBCHEM_PROP_URL, PUBCHEM_SYN_URL, logger
from app.utils import get_response_json


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


def _get_pubchem_metadata(mol: Mol) -> dict[str, str]:
    try:
        if mol is None:
            return {"iupac": "Unknown", "common": "Unknown"}

        # ⭐ Convert RDKit Mol → SMILES STRING
        smiles = Chem.MolToSmiles(mol)

        # Step 1: SMILES → CID lookup
        cid_url = PUBCHEM_CID_URL.format(smiles)
        cid_data = get_response_json(cid_url)

        if not cid_data:
            return {"iupac": "Unknown", "common": "Unknown"}

        cid_list = cid_data.get("IdentifierList", {}).get("CID", [])

        if not cid_list or len(cid_list) == 0:
            return {"iupac": "Unknown", "common": "Unknown"}

        cid = cid_list[0]

        # PubChem CIDs start from 1
        if cid <= 0:
            return {"iupac": "Unknown", "common": "Unknown"}

        # Step 2: Get IUPAC name
        prop_url = PUBCHEM_PROP_URL.format(cid)
        prop_data = get_response_json(prop_url)

        iupac = "Unknown"

        if prop_data:
            iupac = (
                prop_data.get("PropertyTable", {})
                .get("Properties", [{}])[0]
                .get("IUPACName", "Unknown")
            )

        # Step 3: Get synonyms
        syn_url = PUBCHEM_SYN_URL.format(cid)
        syn_data = get_response_json(syn_url)

        synonyms = []

        if syn_data:
            synonyms = (
                syn_data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
            )

        common = get_clean_common_name(synonyms)

        return {"iupac": iupac, "common": common}

    except Exception as e:
        logger.exception(f"Error retrieving PubChem metadata: {e}")
        st.error(str(e))
        return {"iupac": "Error", "common": "Error"}


@st.cache_data(ttl=86400)
def get_pubchem_metadata(mol: Mol) -> dict[str, str]:
    return _get_pubchem_metadata(mol)

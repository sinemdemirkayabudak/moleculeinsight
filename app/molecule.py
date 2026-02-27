import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, Mol

from app.config import logger


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_molecule(smiles: str) -> Mol | None:
    """Convert SMILES string to RDKit Mol object with caching."""
    try:
        if not smiles or not smiles.strip():
            logger.warning("Empty SMILES string provided")
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            logger.info(f"Created molecule from SMILES: {smiles}")
        return mol
    except Exception as e:
        logger.error(f"Failed to create molecule from SMILES '{smiles}': {e}")
        return None


def get_rdkit_properties(mol: Mol) -> dict[str, float] | None:
    try:
        properties = {
            "mw": Descriptors.MolWt(mol),  # molecular weight
            "logP": Crippen.MolLogP(mol),  # logP (octanol/water)
            "tpsa": Descriptors.TPSA(mol),  # topological polar surface area
            "hbd": Lipinski.NumHDonors(mol),  # hydrogen bond donors
            "hba": Lipinski.NumHAcceptors(mol),  # hydrogen bond acceptors
            "rotb": Lipinski.NumRotatableBonds(mol),  # rotatable bonds
        }
        return properties

    except Exception as e:
        logger.error(f"Property calculation failed: {e}")
        st.error(f"Property calculation failed: {e}")
        return None


def lipinski_rules(properties: dict[str, float]) -> dict[str, bool]:
    try:
        mw = properties["mw"]
        logp = properties["logP"]
        hbd = properties["hbd"]
        hba = properties["hba"]

        rules = {
            "MW <= 500": mw <= 500,
            "LogP <= 5": logp <= 5,
            "HBD <= 5": hbd <= 5,
            "HBA <= 10": hba <= 10,
        }
        violations = sum(not passed for passed in rules.values())
        logger.info(f"Lipinski rules evaluated: {violations} violation(s)")
        return rules
    except KeyError as e:
        logger.error(f"Missing property key in lipinski_rules: {e}")
        raise
    except Exception as e:
        logger.error(f"Lipinski rules calculation failed: {e}")
        raise

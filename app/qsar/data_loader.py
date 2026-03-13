"""
Fetch EGFR IC50 bioactivity data from ChEMBL.

This module queries the ChEMBL API for IC50 measurements against EGFR
(target_chembl_id = CHEMBL203) to build bulk datasets for QSAR modeling.
"""

from typing import Any

import pandas as pd
import streamlit as st

from app.chembl import get_chembl_bioactivity, get_chembl_target_id
from app.config import logger

STANDARD_TYPE = "IC50"
STANDARD_RELATION = "="
STANDARD_UNITS = "nM"
TARGET_NAME = "Epidermal growth factor receptor"


@st.cache_data(ttl=86400)
def get_egfr_ic50_data(limit: int = 10000, offset: int = 0) -> dict[str, Any] | None:
    """
    Fetch raw EGFR IC50 bioactivity data from ChEMBL.

    Dynamically resolves EGFR target ID by target name, then queries
    activity.json endpoint for all compounds tested against EGFR with IC50
    measurements. Data is in nM units (not converted to pIC50 yet).
    Supports pagination via limit/offset for bulk dataset collection.

    Parameters
    ----------
    limit : int
        Number of records per API call (default 10000)
    offset : int
        Pagination offset (default 0). Use with limit to fetch different pages.
        Example: offset=0, limit=10000 (records 0-9999)
                 offset=10000, limit=10000 (records 10000-19999)

    Returns
    -------
    dict or None
        Dictionary with:
        - 'success' (bool): Query success status
        - 'data' (pd.DataFrame): Raw IC50 data with columns:
            - smiles: SMILES string
            - standard_value: IC50 in nM
            - assay_id: ChEMBL assay ID
            - reference: Target name
        - 'count' (int): Number of records in this batch
        - 'total_returned' (int): Total records returned by API
        - 'error' (str): Error message if failed

    Examples
    --------
    >>> # Fetch first 10000 records
    >>> result = get_egfr_ic50_data(limit=10000, offset=0)

    >>> # Fetch next batch (records 10000-19999)
    >>> result = get_egfr_ic50_data(limit=10000, offset=10000)
    """

    try:
        # Dynamically fetch EGFR target ID
        target_id = get_chembl_target_id(TARGET_NAME)
        if not target_id:
            error_msg = f"Could not resolve target ID for {TARGET_NAME}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
            }

        logger.info(
            f"Fetching EGFR IC50 data from ChEMBL (target_id={target_id}, "
            f"limit={limit}, offset={offset})"
        )

        # Use centralized ChEMBL bioactivity function with target query
        result = get_chembl_bioactivity(
            target_chembl_id=target_id,
            standard_type=STANDARD_TYPE,
            standard_relation=STANDARD_RELATION,
            standard_units=STANDARD_UNITS,
            limit=limit,
            offset=offset,
        )

        if not result.get("success"):
            logger.warning(f"Failed to fetch EGFR IC50 data: {result.get('error')}")
            return result

        return {
            "success": True,
            "data": result.get("data"),
            "count": result.get("count"),
            "total_returned": result.get("total_returned"),
        }

    except Exception as e:
        logger.exception(f"Error fetching EGFR IC50 data: {e}")
        return {"success": False, "error": str(e)}


@st.cache_data(ttl=86400)
def load_egfr_dataset(limit: int = 10000) -> pd.DataFrame | None:
    """
    Load EGFR IC50 dataset (wrapper for convenience).

    Cached for 24 hours to avoid redundant API calls and DataFrame operations.

    Parameters
    ----------
    limit : int
        Maximum number of records to fetch in a single API call (default 10000)

    Returns
    -------
    pd.DataFrame or None
        Raw IC50 data or None if failed
    """
    result = get_egfr_ic50_data(limit=limit)
    if result and result.get("success"):
        return result.get("data")
    return None

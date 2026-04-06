# User Interface for Streamlit app - handles rendering of components,
# user interactions, and display logic.
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.figure import Figure
from rdkit.Chem import Draw

from app.chembl import get_compound_bioactivity_from_mol
from app.config import logger
from app.molecule import get_molecule, get_rdkit_properties, lipinski_rules
from app.pubchem import get_pubchem_metadata
from app.qsar.features import compute_morgan_fingerprints, compute_rdkit_descriptors
from app.qsar.predict import QSARPredictor
from app.qsar.smiles_processor import (
    get_all_smiles,
    get_cached_bioactivity,
    load_egfr_compounds,
)
from app.scaffold_sar import (
    add_scaffolds_to_dataframe,
    detect_activity_cliffs,
    fetch_missing_ic50_values,
    get_ic50_summary_stats,
    load_sample_ic50_data,
    summarize_scaffolds,
)
from app.similarity_search import create_structure_image, prepare_csv_export, run_similarity_search
from app.utils import safe_execute
from app.validators import validate_smiles
from app.virtual_screening import run_virtual_screening_pipeline


def display_results_table(results_df: pd.DataFrame) -> None:
    """Display the results dataframe with structure images.

    Parameters:
        results_df (pd.DataFrame): Results dataframe with columns: Query Molecule, Query SMILES,
                                  Reference Molecule, Reference SMILES, Similarity Score, Structures

    Returns:
        None. Renders Streamlit components directly.
    """
    st.subheader("Results")

    st.caption(
        "💡 Tip: Click on an image to see a zoomed-in view of the query and reference molecules"
    )

    display_df = results_df.drop(columns=["Reference SMILES", "Query SMILES"], errors="ignore")
    st.dataframe(
        display_df,
        column_config={"Structures": st.column_config.ImageColumn("Structures", width="medium")},
        width="stretch",
    )
    csv = prepare_csv_export(results_df)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="similarity_results.csv",
        mime="text/csv",
        width="content",
    )


def display_ranking_plots(plots_dict: dict[str, Figure]) -> None:
    """Display ranking plots with query dropdown selector.

    Parameters:
        plots_dict (dict): Dictionary mapping query_name -> matplotlib figure object

    Returns:
        None. Renders Streamlit components directly. Returns early if plots_dict is empty.
    """
    if plots_dict is None or len(plots_dict) == 0:
        return

    st.subheader("Similarity Ranking Plots")
    unique_queries = list(plots_dict.keys())

    selected_query = st.selectbox(
        "Select query molecule to view ranking plot:",
        unique_queries,
        key="query_plot_select",
    )

    plot_df = plots_dict.get(selected_query)

    if plot_df is not None and not plot_df.empty:
        df = plot_df.reset_index().rename(columns={"index": "ref_name"})
        df = df.sort_values("Similarity", ascending=False)

        base = alt.Chart(df).encode(y=alt.Y("ref_name:N", sort="-x", title=None))

        max_sim = df["Similarity"].max()
        x_max = max_sim

        bars = base.mark_bar().encode(
            x=alt.X(
                "Similarity:Q", title="Tanimoto Similarity", scale=alt.Scale(domain=[0, x_max])
            ),
            tooltip=[
                alt.Tooltip("ref_name:N", title="Reference Molecule"),
                alt.Tooltip("Similarity:Q", title="Tanimoto Similarity", format=".3f"),
            ],
        )

        # 👇 TEXT anchored at end of bars (reliable)
        text = base.mark_text(align="right", dx=30, color="white").encode(
            x="Similarity:Q", text=alt.Text("Similarity:Q", format=".3f")
        )

        chart = (bars + text).properties(
            height=max(300, len(df) * 25),
        )

        st.altair_chart(chart, width="stretch")

    else:
        st.warning("No data available for selected query.")


def generate_structure_images(
    results_df: pd.DataFrame, query_smiles_map: dict[str, str]
) -> list[str | None]:
    """Generate base64-encoded structure images for each result.

    Creates side-by-side comparison images for each query-reference pair.
    Images are cached internally for performance on repeated molecules.

    Parameters:
        results_df (pd.DataFrame): Results with columns: query_name, ref_name, smiles
        query_smiles_map (dict): Mapping of query_name -> SMILES string

    Returns:
        list: Base64-encoded PNG data URLs (or None) for each result row
    """
    structure_images = []
    with st.spinner("Generating structure images..."):
        for query_name, ref_name, ref_smiles in zip(
            results_df["query_name"], results_df["ref_name"], results_df["smiles"], strict=True
        ):
            query_smiles = query_smiles_map.get(query_name)

            # Cached function - reuses previously generated images
            img_base64 = create_structure_image(query_name, query_smiles, ref_name, ref_smiles)
            # Convert to data URL for ImageColumn
            img_url = f"data:image/png;base64,{img_base64}" if img_base64 else None
            structure_images.append(img_url)

    return structure_images


def prepare_display_dataframe(
    results_df: pd.DataFrame, query_smiles_map: dict[str, str], structure_images: list[str | None]
) -> pd.DataFrame:
    """Transform results DataFrame for UI display.

    Adds structure images, query SMILES, and renames columns for better readability.

    Parameters:
        results_df (pd.DataFrame): Raw results from similarity search
        query_smiles_map (dict): Mapping of query_name -> SMILES string
        structure_images (list): Base64-encoded structure images for each row

    Returns:
        pd.DataFrame: Formatted DataFrame ready for display
    """
    # Create copy to avoid mutating input DataFrame
    results_df = results_df.copy()

    # Add structures column
    results_df["structures"] = structure_images

    # Add Query SMILES column for CSV export
    results_df["query_smiles"] = results_df["query_name"].map(query_smiles_map)

    # Rename columns for better display
    results_df = results_df.rename(
        columns={
            "query_name": "Query Molecule",
            "query_smiles": "Query SMILES",
            "ref_name": "Reference Molecule",
            "smiles": "Reference SMILES",
            "similarity": "Similarity Score",
            "structures": "Structures",
        }
    )

    return results_df


def process_similarity_results(results_df: pd.DataFrame, query_df: pd.DataFrame) -> pd.DataFrame:
    """Process raw similarity search results into displayable format.

    Orchestrates image generation and DataFrame preparation.

    Parameters:
        results_df (pd.DataFrame): Raw results from run_similarity_search()
        query_df (pd.DataFrame): Query molecules DataFrame with 'name' and 'smiles'

    Returns:
        pd.DataFrame: Formatted DataFrame ready for display
    """
    # Create lookup for query SMILES
    query_df_display = query_df.rename(columns={"name": "query_name"})
    query_smiles_map = dict(
        zip(query_df_display["query_name"], query_df_display["smiles"], strict=True)
    )

    # Generate structure images for each result (cached for performance)
    structure_images = generate_structure_images(results_df, query_smiles_map)

    # Prepare display DataFrame with images and renamed columns
    display_df = prepare_display_dataframe(results_df, query_smiles_map, structure_images)

    return display_df


def render_single_molecule() -> None:
    """Render single molecule analysis interface.

    Allows users to input SMILES or select example molecules.
    Displays molecular properties, Lipinski rule-of-5 compliance, and bioactivity data from ChEMBL.

    Returns:
        None. Renders Streamlit components directly.
    """
    st.subheader("Single Molecule Analysis")

    st.markdown(
        "Analyze molecular properties, visualize structures, and explore known bioactivity for a single compound."
    )

    st.markdown("")

    # Clear cached similarity search results when switching pages
    if st.session_state.get("last_page") == "Similarity Search":
        st.session_state.pop("results_df", None)
        st.session_state.pop("query_plots", None)
        st.session_state.pop("sample_query_path", None)
        st.session_state.pop("sample_ref_path", None)
    st.session_state.last_page = "Single Molecule"

    col1, col2 = st.columns([3, 1])
    with col1:
        smiles = st.text_input(
            "Enter SMILES",
            placeholder="e.g., Nc1ncnc2cc(O)c(O)cc12",
            label_visibility="visible",
            help="SMILES: Simplified Molecular Input Line Entry System",
            key="smiles_input_main",
        )

    with col2:
        # Load EGFR inhibitors from JSON data file
        try:
            egfr_compounds = load_egfr_compounds()
            egfr_smiles_map = get_all_smiles(egfr_compounds)
        except Exception:
            egfr_smiles_map = {}

        # Example molecules dropdown: use only validated EGFR inhibitors
        examples = {"None": ""}
        examples.update(egfr_smiles_map)  # Add only validated EGFR inhibitors from JSON

        def update_smiles_from_example():
            """Callback to update SMILES when example is selected."""
            selected = st.session_state.get("example_select", "None")
            if selected != "None" and selected in examples:
                st.session_state["smiles_input_main"] = examples[selected]

        st.selectbox(
            "Load Example Molecule",
            options=["None"] + sorted([k for k in examples.keys() if k != "None"]),
            index=0,
            key="example_select",
            on_change=update_smiles_from_example,
        )

    if smiles:
        smiles = smiles.strip()
        is_valid, error_msg = validate_smiles(smiles)
        if not is_valid:
            st.error(error_msg)
            return

        mol = get_molecule(smiles)

        if not mol:
            st.error("Invalid SMILES string")
            return

        properties = get_rdkit_properties(mol)
        if not properties:
            return

        rules = safe_execute(lipinski_rules, properties)
        if not rules:
            st.error("Lipinski rules calculation failed")
            return

        mw = properties["mw"]
        logp = properties["logP"]
        tpsa = properties["tpsa"]
        hbd = properties["hbd"]
        hba = properties["hba"]
        rotb = properties["rotb"]

        violations = sum(not passed for passed in rules.values())

        tab1, tab2, tab3 = st.tabs(["Properties", "Lipinski Rules", "Bioactivity"])

        with tab1:
            col1, col2 = st.columns(2, gap="small")

            with col1:
                with st.container():
                    st.subheader("Molecular Structure")
                    img = Draw.MolToImage(mol, size=(400, 300))
                    st.image(img, width="content")

            with col2:
                with st.container():
                    meta = get_pubchem_metadata(mol)
                    prop = {
                        "IUPAC Name": meta.get("iupac", "N/A"),
                        "Common Name": meta.get("common", "N/A"),
                        "CID": meta.get("cid", "N/A"),
                        "InChIKey": meta.get("inchikey", "N/A"),
                        "Molecular Weight (MW)": f"{mw:.2f}",
                        "LogP (octanol-water)": f"{logp:.2f}",
                        "Topological Polar Surface Area (TPSA)": f"{tpsa:.2f}",
                        "H-bond Donors (HBD)": str(hbd),
                        "H-bond Acceptors (HBA)": str(hba),
                        "Rotatable Bonds": str(rotb),
                    }
                    prop_df = pd.DataFrame(list(prop.items()), columns=["Property", "Value"])
                    # Explicitly set all columns to string dtype to prevent PyArrow errors
                    prop_df = prop_df.astype(str)
                    st.dataframe(prop_df, width="stretch", hide_index=True)

        with tab2:
            st.subheader("Lipinski Rule-of-5 Compliance")
            df = pd.DataFrame(
                [
                    {"Rule": rule, "Status": "✔ Passed" if passed else "✘ Violated"}
                    for rule, passed in rules.items()
                ]
            )
            # Explicitly set all columns to string dtype to prevent PyArrow errors
            df = df.astype(str)
            st.dataframe(df, width="stretch", hide_index=True)

            st.write(f"**Total violations:** {violations}")

        with tab3:
            st.subheader("Bioactivity Evidence (ChEMBL)")

            # Try to get cached bioactivity data for known EGFR inhibitors
            try:
                egfr_compounds = load_egfr_compounds()
                cached_bioactivity = get_cached_bioactivity(egfr_compounds, smiles)

                if cached_bioactivity:
                    st.info(
                        f"✓ Using cached bioactivity data: {len(cached_bioactivity)} records (no API call needed)"
                    )

                    # Display all cached bioactivity records
                    # First, ensure all values are strings to avoid PyArrow type conflicts
                    sanitized_bioactivity = []
                    for record in cached_bioactivity:
                        sanitized_record = {
                            "target_chembl_id": str(record.get("target_chembl_id", "N/A")),
                            "target_name": str(record.get("target_name", "N/A")),
                            "activity_type": str(record.get("activity_type", "N/A")),
                            "value": str(record.get("value", "N/A")),
                            "units": str(record.get("units", "")),
                            "assay_description": str(record.get("assay_description", "")),
                        }
                        sanitized_bioactivity.append(sanitized_record)

                    bioactivity_df = pd.DataFrame(sanitized_bioactivity)

                    # Filter columns to display (hide optional/internal fields)
                    display_columns = [
                        "target_chembl_id",
                        "target_name",
                        "activity_type",
                        "value",
                        "units",
                        "assay_description",
                    ]
                    bioactivity_df = bioactivity_df[
                        [col for col in display_columns if col in bioactivity_df.columns]
                    ]

                    bioactivity_df = bioactivity_df.rename(
                        columns={
                            "target_chembl_id": "Target ID",
                            "target_name": "Target Name",
                            "activity_type": "Activity Type",
                            "value": "Value",
                            "units": "Units",
                            "assay_description": "Assay Description",
                        }
                    )

                    st.dataframe(bioactivity_df, width="stretch")
                    st.caption(
                        f"Showing all {len(cached_bioactivity)} cached bioactivity records for this compound"
                    )

                    # Show selection rationale for each record
                    with st.expander("Why were these records selected?"):
                        st.write(
                            "These cached bioactivity records were **strategically selected** based on scientific significance, not just top 2 by potency."
                        )
                        for i, record in enumerate(cached_bioactivity, 1):
                            activity_type = record.get("activity_type", "Unknown")
                            value = record.get("value")
                            units = record.get("units", "")
                            target = record.get("target_name", "N/A")
                            rationale = record.get(
                                "selection_rationale", "No explanation available"
                            )

                            st.write(
                                f"**Record {i}: {activity_type}** ({value} {units}) vs {target}"
                            )
                            st.info(rationale)

                    # Option to fetch all bioactivity evidence from API
                    with st.expander(
                        "View all bioactivity evidence (via ChEMBL API)", expanded=False
                    ):
                        st.write(
                            "Fetch comprehensive bioactivity data from ChEMBL. This may include records for different targets, assays, and activity types."
                        )

                        api_limit = st.slider(
                            "Number of records to retrieve from API",
                            min_value=10,
                            max_value=500,
                            value=100,
                            step=10,
                            key="bioactivity_api_limit",
                            help="Maximum number of bioactivity records to fetch from ChEMBL",
                        )

                        if st.button("🔄 Fetch all bioactivity data", key="fetch_all_bioactivity"):
                            # Create a cache key based on SMILES and API limit
                            # This ensures different record limits have separate caches
                            cache_key = f"bioactivity_cache_{smiles}_{api_limit}"

                            # Check if we have cached results from a previous fetch with this limit
                            if cache_key in st.session_state:
                                st.info(
                                    f"📦 Using previously fetched results from cache (instant) - {api_limit} records"
                                )
                                activities = st.session_state[cache_key]
                            else:
                                with st.spinner(
                                    "Fetching bioactivity data from ChEMBL API (this may take 30-60 seconds)..."
                                ):
                                    bioactivity_data = get_compound_bioactivity_from_mol(
                                        mol, limit=api_limit
                                    )

                                    if bioactivity_data.get("success"):
                                        activities = bioactivity_data.get("bioactivity", {}).get(
                                            "activities", []
                                        )
                                        if activities:
                                            # Cache successful results
                                            st.session_state[cache_key] = activities
                                    else:
                                        activities = None
                                        error_stage = bioactivity_data.get("stage", "unknown")
                                        error_msg = bioactivity_data.get("error", "Unknown error")

                                        logger.warning(
                                            f"Failed to retrieve bioactivity data from ChEMBL API\nError stage: {error_stage}\nDetails: {error_msg}"
                                        )
                                        st.error(
                                            "Failed to retrieve bioactivity data from ChEMBL API"
                                        )

                            # Display results if we have them
                            if activities:
                                st.success(
                                    f"✓ Retrieved {len(activities)} bioactivity records from ChEMBL API"
                                )

                                api_df = pd.DataFrame(activities)

                                api_df = api_df.rename(
                                    columns={
                                        "target_chembl_id": "Target ID",
                                        "target_name": "Target Name",
                                        "standard_type": "Activity Type",
                                        "standard_value": "Value",
                                        "standard_units": "Units",
                                        "assay_description": "Assay Description",
                                    }
                                )

                                st.divider()
                                st.write("**All bioactivity evidence from ChEMBL (API results):**")
                                st.dataframe(api_df, width="stretch")

                                # Summary statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Records", len(activities))
                                with col2:
                                    unique_targets = len(
                                        set(
                                            r.get("target_name")
                                            for r in activities
                                            if r.get("target_name")
                                        )
                                    )
                                    st.metric("Unique Targets", unique_targets)
                                with col3:
                                    unique_types = len(
                                        set(
                                            r.get("standard_type")
                                            for r in activities
                                            if r.get("standard_type")
                                        )
                                    )
                                    st.metric("Activity Types", unique_types)
                            elif cache_key not in st.session_state:
                                st.info("No bioactivity records found for this compound")
                else:
                    # Fall back to API if not cached
                    st.info("Fetching live bioactivity data from ChEMBL...")
                    bioactivity_limit = st.slider(
                        "Bioactivity Records Limit",
                        min_value=10,
                        max_value=500,
                        value=100,
                        step=10,
                        help="Maximum number of bioactivity records to retrieve from ChEMBL",
                    )

                    bioactivity_data = get_compound_bioactivity_from_mol(
                        mol, limit=bioactivity_limit
                    )
                    if bioactivity_data.get("success"):
                        activities = bioactivity_data.get("bioactivity", {}).get("activities", [])

                        if activities:
                            df = pd.DataFrame(activities)

                            df = df.rename(
                                columns={
                                    "target_chembl_id": "Target ID",
                                    "target_name": "Target Name",
                                    "standard_type": "Activity Type",
                                    "standard_value": "Value",
                                    "standard_units": "Units",
                                    "assay_description": "Assay Description",
                                }
                            )

                            st.dataframe(df, width="stretch")
                            st.caption(
                                f"Retrieved {len(activities)} bioactivity records from ChEMBL"
                            )
                        else:
                            st.info("No bioactivity records found for this compound")

                    else:
                        error_stage = bioactivity_data.get("stage", "unknown")
                        error_msg = bioactivity_data.get("error", "Unknown error")

                        logger.warning(
                            f"Failed to retrieve bioactivity data from ChEMBL API\nError stage: {error_stage}\nDetails: {error_msg}"
                        )
                        st.error("Failed to retrieve bioactivity data from ChEMBL API")

            except Exception as e:
                logger.warning(f"Error checking cached bioactivity: {e}")
                # Fall back to API on any error
                bioactivity_limit = st.slider(
                    "Bioactivity Records Limit",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Maximum number of bioactivity records to retrieve from ChEMBL",
                )

                bioactivity_data = get_compound_bioactivity_from_mol(mol, limit=bioactivity_limit)
                if bioactivity_data.get("success"):
                    activities = bioactivity_data.get("bioactivity", {}).get("activities", [])

                    if activities:
                        df = pd.DataFrame(activities)

                        df = df.rename(
                            columns={
                                "target_chembl_id": "Target ID",
                                "target_name": "Target Name",
                                "standard_type": "Activity Type",
                                "standard_value": "Value",
                                "standard_units": "Units",
                                "assay_description": "Assay Description",
                            }
                        )

                        st.dataframe(df, width="stretch")
                        st.caption(f"Retrieved {len(activities)} bioactivity records from ChEMBL")
                    else:
                        st.info("No bioactivity records found for this compound")

                    df = df.rename(
                        columns={
                            "target_chembl_id": "Target ID",
                            "target_name": "Target Name",
                            "standard_type": "Activity Type",
                            "standard_value": "Value",
                            "standard_units": "Units",
                            "assay_description": "Assay Description",
                        }
                    )

                    st.dataframe(df, width="stretch", height="content")

                else:
                    st.error("Bioactivity data retrieval failed")


def render_similarity_search() -> None:
    """Render similarity search interface.

    Allows users to upload CSV files or load sample data with query and reference molecules.
    Performs Morgan fingerprint similarity search and displays results with optional ranking plots.

    Returns:
        None. Renders Streamlit components directly and saves results to session state.
    """
    # Clear cached results when switching from a different page
    if st.session_state.get("last_page") != "Similarity Search":
        st.session_state.pop("results_df", None)
        st.session_state.pop("query_plots", None)
    st.session_state.last_page = "Similarity Search"

    # ========== DESCRIPTION ==========
    st.subheader("Molecular Similarity Search")

    st.markdown("""
    The tool computes **Morgan fingerprints** and calculates **Tanimoto similarity scores** 
    (the standard metric for binary fingerprint comparison in chemoinformatics).
    Results are ranked from highest to lowest similarity.
    """)

    # Show file format examples
    with st.expander("View CSV Format Examples"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Query Molecules Example")
            query_csv = """
            smiles,name
            CCO,Ethanol
            CCN,Ethylamine
            CCC,Propane
            CC(=O)O,AceticAcid
            c1ccccc1,Benzene
            """
            st.code(query_csv, language="csv")
            with st.expander("Preview"):
                query_example = pd.DataFrame(
                    {
                        "smiles": ["CCO", "CCN", "CCC", "CC(=O)O", "c1ccccc1"],
                        "name": ["Ethanol", "Ethylamine", "Propane", "AceticAcid", "Benzene"],
                    }
                )
                st.dataframe(query_example, hide_index=True, width="stretch")

        with col2:
            st.subheader("Reference Library Example")
            ref_csv = """
            smiles,name
            CCO,Ethanol_ref
            CCOC,EthylMethylEther
            CCCC,Butane
            CC(=O)OC,MethylAcetate
            c1ccccc1O,Phenol
            CC(=O)N,Acetamide
            CCCN,Propylamine
            COC,DimethylEther
            CO,Methanol
            """

            st.code(ref_csv, language="csv")
            with st.expander("Preview"):
                ref_example = pd.DataFrame(
                    {
                        "smiles": [
                            "CCO",
                            "CCOC",
                            "CCCC",
                            "CC(=O)OC",
                            "c1ccccc1O",
                            "CC(=O)N",
                            "CCCN",
                            "COC",
                            "CO",
                        ],
                        "name": [
                            "Ethanol_ref",
                            "EthylMethylEther",
                            "Butane",
                            "MethylAcetate",
                            "Phenol",
                            "Acetamide",
                            "Propylamine",
                            "DimethylEther",
                            "Methanol",
                        ],
                    }
                )
                st.dataframe(ref_example, hide_index=True, width="stretch")

    st.markdown("")

    # ========== INPUTS AND CONTROLS ==========
    st.markdown("### Settings")

    # Load sample data button
    if st.button("Load Sample Data", help="Load example molecules for quick testing"):
        sample_query = "app/data/sample/query_molecules.csv"
        sample_ref = "app/data/sample/reference_library.csv"
        st.session_state.sample_query_path = sample_query
        st.session_state.sample_ref_path = sample_ref
        st.success("Sample data loaded!")

    st.markdown("**Or upload your CSV files:**")

    # File uploads
    col1, col2 = st.columns(2, gap="medium")

    query_file = col1.file_uploader("Query molecules (CSV)", key="query_file")
    ref_file = col1.file_uploader("Reference library (CSV)", key="ref_file")

    st.markdown("")

    # Parameter controls
    radius = col2.number_input(
        "Fingerprint Radius",
        min_value=0,
        max_value=5,
        value=2,
        help="0=atoms, 1=neighbors, 2=ECFP4 (standard), 3+=extended",
    )

    top_n = col2.number_input("Top N Results", min_value=1, max_value=100, value=20)

    show_plots = col2.checkbox("Show Similarity Ranking Plots", value=True)

    if "sample_query_path" in st.session_state and "sample_ref_path" in st.session_state:
        # Load from sample files
        query_df = pd.read_csv(st.session_state.sample_query_path)
        ref_df = pd.read_csv(st.session_state.sample_ref_path)
        data_source = "sample"
    elif query_file and ref_file:
        # Load from uploaded files directly (no disk I/O)
        # Streamlit file objects can be passed directly to pandas
        query_df = pd.read_csv(query_file)
        ref_df = pd.read_csv(ref_file)
        data_source = "uploaded"
    else:
        query_df = None
        ref_df = None
        data_source = None

    run_search = st.button("Run Similarity Search", type="primary", width="content")

    # ========== RESULTS ==========

    # Status indicator
    if data_source == "sample":
        st.info("ℹ️ Using sample data")
    elif data_source == "uploaded":
        st.info("ℹ️ Using uploaded data")

    # Run similarity search
    if run_search and query_df is not None and ref_df is not None:
        try:
            with st.spinner("Running similarity search..."):
                # Run similarity search directly with DataFrames (in-memory, no disk I/O)
                figures, results_df = run_similarity_search(
                    query_file=query_df,
                    reference_file=ref_df,
                    radius=radius,
                    top_n=top_n,
                    show_plots=show_plots,
                )

            # Display results
            st.success("Similarity search completed!")

            # Process results and prepare for display
            display_df = process_similarity_results(results_df, query_df)

            # Store in session state to persist across interactions
            st.session_state.results_df = display_df
            st.session_state.query_plots = figures  # Store pre-cached plots

            # Display results and plots
            display_results_table(display_df)
            st.markdown("")
            display_ranking_plots(figures)

        except Exception as e:
            st.error(f"Error during similarity search: {e}")

    # Display cached results if available (e.g., when dropdown is interacted with)
    elif "results_df" in st.session_state and not run_search:
        results_df = st.session_state.results_df
        plots_dict = st.session_state.query_plots if "query_plots" in st.session_state else {}

        # Display results and plots
        st.success("Similarity search completed!")
        display_results_table(results_df)
        st.divider()
        display_ranking_plots(plots_dict)
    elif run_search and query_df is None:
        st.warning("⚠️ Please load sample data or upload both query and reference files to proceed")


def render_qsar_dashboard() -> None:
    """Render QSAR model prediction dashboard with visualizations.

    Displays model performance plots, allows users to make predictions on new molecules
    with SMILES input, shows SHAP feature importance explanations, and provides feature
    interpretation reference.

    Returns:
        None. Renders Streamlit components directly.
    """
    if st.session_state.get("last_page") != "QSAR Model":
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("smiles_input_value", None)
        st.session_state.pop("quick_example_select", None)
    st.session_state.last_page = "QSAR Model"

    st.subheader("EGFR pIC50 Prediction Model")

    st.markdown("""
    XGBoost QSAR model trained on ChEMBL bioactivity data to predict EGFR binding affinity (pIC50) from molecular structures.
    """)

    # Create 2 tabs
    tab1, tab2 = st.tabs(["Make Predictions", "Model Performance"])

    # ========== TAB 1: MAKE PREDICTIONS ==========
    with tab1:
        # Mark that we're in tab1 - this will prevent tab2 code from affecting tab1
        st.session_state.active_qsar_tab = "make_predictions"

        st.markdown("""
        Enter a **SMILES string** to predict its binding affinity (pIC50) to EGFR.
        The model will show the predicted value and binding strength interpretation.
        """)

        # Initialize session state for SMILES input
        if "smiles_input_value" not in st.session_state:
            st.session_state.smiles_input_value = ""

        st.markdown("")
        # SMILES input with example molecules
        col1, col2 = st.columns([3, 1])

        with col1:
            smiles_input = st.text_input(
                "Enter SMILES",
                value=st.session_state.smiles_input_value,
                placeholder="e.g., Nc1ncnc2cc(O)c(O)cc12",
                help="SMILES: Simplified Molecular Input Line Entry System",
                label_visibility="visible",
            )

            # Update session state with text input changes
            if smiles_input:
                st.session_state.smiles_input_value = smiles_input

        with col2:
            # Load EGFR inhibitors from JSON data file
            try:
                egfr_compounds = load_egfr_compounds()
                example_molecules = {c["name"]: c["smiles"] for c in egfr_compounds}
            except Exception:
                example_molecules = {}

            def update_smiles_from_example():
                """Callback to update SMILES when example is selected."""
                selected = st.session_state.get("quick_example_select", "None")
                if selected != "None" and selected in example_molecules:
                    st.session_state.smiles_input_value = example_molecules[selected]

            st.selectbox(
                "Load Example Molecule",
                options=["None"] + sorted([k for k in example_molecules.keys() if k != "None"]),
                label_visibility="visible",
                key="quick_example_select",
                on_change=update_smiles_from_example,
            )

        # Make prediction when SMILES is entered
        if smiles_input:
            smiles_input = smiles_input.strip()

            # Validate SMILES
            is_valid, error_msg = validate_smiles(smiles_input)
            if not is_valid:
                st.error(f"❌ Invalid SMILES: {error_msg}")
            else:
                try:
                    with st.spinner("Making prediction..."):
                        # Get molecule object for visualization
                        mol = get_molecule(smiles_input)
                        if not mol:
                            st.error("Invalid SMILES string")
                        else:
                            # Compute features (Morgan + RDKit)
                            morgan_result = compute_morgan_fingerprints([smiles_input])
                            rdkit_result = compute_rdkit_descriptors([smiles_input])

                            if not (morgan_result["success"] and rdkit_result["success"]):
                                st.error("Failed to compute molecular features")
                            else:
                                X_morgan = morgan_result["X"]
                                X_rdkit = rdkit_result["X"]
                                X_combined = np.hstack([X_morgan, X_rdkit])

                                # Load model and make prediction
                                model_path = (
                                    Path(__file__).parent
                                    / "qsar"
                                    / "saved_models"
                                    / "egfr_xgb_model.pkl"
                                )
                                if not model_path.exists():
                                    st.error(
                                        "❌ Model file not found. Please train the model first."
                                    )
                                else:
                                    model = joblib.load(model_path)
                                    y_pred = QSARPredictor.predict(model, X_combined)[0]

                                    # Calculate prediction confidence interval (95%)
                                    # Load residual standard error from model metadata
                                    residual_std = 0.65  # Default estimate
                                    ci_margin = 1.274  # 1.96 × 0.65 (default)

                                    # Try to load actual metrics from metadata
                                    metadata_path = (
                                        Path(__file__).parent
                                        / "qsar"
                                        / "saved_models"
                                        / "egfr_metadata.json"
                                    )
                                    if metadata_path.exists():
                                        try:
                                            with open(metadata_path) as f:
                                                metadata = json.load(f)
                                                if "uncertainty_metrics" in metadata:
                                                    residual_std = metadata[
                                                        "uncertainty_metrics"
                                                    ].get("residual_std", 0.65)
                                                    ci_margin = metadata["uncertainty_metrics"].get(
                                                        "ci_95_margin", 1.96 * residual_std
                                                    )
                                        except Exception:
                                            pass

                                    y_pred_lower = y_pred - ci_margin
                                    y_pred_upper = y_pred + ci_margin

                                    # Display prediction result
                                    st.success("✅ Prediction Complete!")

                                    # Show molecule structure and prediction
                                    col1, col2 = st.columns([1, 2], gap="medium")

                                    with col1:
                                        st.markdown("**Molecular Structure**")
                                        img = Draw.MolToImage(mol, size=(400, 300))
                                        st.image(img, width="content")

                                    with col2:
                                        st.markdown("**Prediction Result**")

                                        # Display prediction with confidence interval
                                        st.metric(
                                            "Predicted pIC50",
                                            f"{y_pred:.2f}",
                                            help="Higher values = stronger binding affinity",
                                        )

                                        # Show confidence interval
                                        st.info(
                                            f"95% Confidence Interval: **{y_pred_lower:.2f} – {y_pred_upper:.2f}** pIC50\n\n"
                                            f"*Uncertainty range based on model training residuals. "
                                            f"±{ci_margin:.2f} pIC50 units*"
                                        )

                                        # Interpretation guide
                                        st.markdown("**Interpretation (EGFR binding affinity):**")
                                        if y_pred < 3.0:
                                            st.info(
                                                "⚪ Not measurable / Essentially inactive (pIC50 < 3)"
                                            )
                                        elif y_pred < 4.0:
                                            st.info("⚪ Inactive (pIC50 3–4, IC50 > 100 µM)")
                                        elif y_pred < 5.0:
                                            st.info("🟡 Weak (pIC50 4–5, IC50 10–100 µM)")
                                        elif y_pred < 6.0:
                                            st.info("🟠 Moderate (pIC50 5–6, IC50 1–10 µM)")
                                        elif y_pred < 7.0:
                                            st.info(
                                                "🟢 Active/Lead-like (pIC50 6–7, IC50 100nM–1µM)"
                                            )
                                        elif y_pred < 8.0:
                                            st.info("🟢 Strong (pIC50 7–8, IC50 10–100 nM)")
                                        else:
                                            st.info("🟢 Excellent (pIC50 > 8, IC50 < 10 nM)")

                                    # Display feature importance plot (advanced visualization)
                                    st.markdown("---")

                                    try:
                                        # Load pre-computed feature annotations from comprehensive file
                                        morgan_annotations = {}
                                        annotations_path = (
                                            Path(__file__).parent
                                            / "qsar"
                                            / "saved_models"
                                            / "egfr_feature_annotations.json"
                                        )
                                        if annotations_path.exists():
                                            try:
                                                with open(annotations_path) as f:
                                                    anno_data = json.load(f)
                                                    # Extract Morgan bits from comprehensive file
                                                    if "morgan_bits" in anno_data:
                                                        morgan_annotations = {
                                                            int(k): v
                                                            for k, v in anno_data[
                                                                "morgan_bits"
                                                            ].items()
                                                        }
                                                    else:
                                                        # Fallback for old format (direct dict)
                                                        morgan_annotations = {
                                                            int(k): v for k, v in anno_data.items()
                                                        }
                                            except Exception:
                                                pass

                                        # Get global feature importance from model
                                        global_importance = model.feature_importances_

                                        # Weight by feature presence in this molecule
                                        # Features that are 1 (present) get full importance, 0 (absent) get 0
                                        X_sample = (
                                            X_combined.flatten()
                                            if X_combined.ndim > 1
                                            else X_combined
                                        )
                                        weighted_importance = global_importance * np.clip(
                                            X_sample, 0, 1
                                        )

                                        # RDKit descriptors mapping
                                        rdkit_descriptors = {
                                            2048: "MW (Molecular Weight)",
                                            2049: "LogP (Lipophilicity)",
                                            2050: "TPSA (Polar Surface)",
                                            2051: "HBD (H-Bond Donors)",
                                            2052: "HBA (H-Bond Acceptors)",
                                            2053: "RotBonds (Rotatable)",
                                            2054: "AromaticRings",
                                            2055: "RingCount",
                                        }

                                        # Get top 10 features (by weighted importance for THIS molecule)
                                        top_indices = np.argsort(weighted_importance)[-10:][::-1]

                                        # Build labels and values with Morgan bit annotations
                                        top_features = []
                                        top_values = []
                                        feature_types = []

                                        for idx in top_indices:
                                            top_values.append(weighted_importance[idx])
                                            if idx < 2048:
                                                # Use pre-computed annotation if available
                                                if idx in morgan_annotations:
                                                    substructure = morgan_annotations[idx]
                                                    if len(substructure) <= 12:
                                                        feature_label = (
                                                            f"Morgan_Bit{idx:04d}_{substructure}"
                                                        )
                                                    else:
                                                        truncated = substructure[:8] + "*"
                                                        feature_label = (
                                                            f"Morgan_Bit{idx:04d}_{truncated}"
                                                        )
                                                else:
                                                    feature_label = f"Morgan_Bit{idx:04d}"
                                                top_features.append(feature_label)
                                                feature_types.append("Morgan")
                                            elif idx in rdkit_descriptors:
                                                top_features.append(rdkit_descriptors[idx])
                                                feature_types.append("RDKit")
                                            else:
                                                top_features.append(f"RDKit_{idx - 2048}")
                                                feature_types.append("RDKit")

                                        # Create interactive Plotly bar chart
                                        # Build dataframe for Plotly
                                        fi_df = pd.DataFrame(
                                            {
                                                "feature": top_features,
                                                "importance": top_values,
                                                "type": feature_types,
                                            }
                                        )

                                        # Sort by importance (ascending for Plotly horizontal bars)
                                        fi_df_sorted = fi_df.sort_values(
                                            "importance", ascending=True
                                        )

                                        # Create interactive Plotly bar chart
                                        fig = px.bar(
                                            fi_df_sorted,
                                            y="feature",
                                            x="importance",
                                            orientation="h",
                                            color="type",
                                            color_discrete_map={
                                                "Morgan": "#64B5F6",
                                                "RDKit": "#81C784",
                                            },
                                            hover_data={"importance": ":.5f", "type": True},
                                            title="Top 10 Most Important Features for This EGFR Binding Prediction",
                                        )

                                        fig.update_layout(
                                            xaxis_title="Feature Importance (Weighted by Presence in Molecule)",
                                            yaxis_title="",
                                            height=500,
                                            showlegend=True,
                                            legend=dict(
                                                title="Feature Type",
                                                yanchor="bottom",
                                                y=0.01,
                                                xanchor="right",
                                                x=0.99,
                                            ),
                                            yaxis=dict(
                                                autorange=True, categoryorder="total ascending"
                                            ),
                                            hovermode="closest",
                                        )

                                        fig.update_xaxes(
                                            showgrid=True, gridwidth=1, gridcolor="lightgray"
                                        )

                                        st.plotly_chart(fig, width="stretch")

                                        st.caption(
                                            "Scores show per-molecule feature importance (model importance weighted by feature presence). Morgan bits include SMILES substructure annotations where available."
                                        )
                                    except Exception as e:
                                        st.warning(
                                            f"Could not display feature importance: {str(e)[:150]}"
                                        )

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

    # ========== TAB 2: MODEL PERFORMANCE ==========
    with tab2:
        # Mark that we're in tab2 and clear prediction results
        st.session_state.active_qsar_tab = "model_performance"
        st.session_state.pop("prediction_result", None)

        st.markdown("""
        This XGBoost QSAR model predicts **EGFR binding affinity (pIC50)** from molecular structures.
        Trained on ChEMBL bioactivity data with cross-validated performance metrics.
        """)

        # Load performance data from JSON
        performance_data_path = (
            Path(__file__).parent / "qsar" / "visualizations" / "performance_data.json"
        )

        if not performance_data_path.exists():
            st.error(
                "Performance data not found. Please run `python qsar/visualizations.py` to generate it."
            )
            return

        # Load JSON data
        with open(performance_data_path) as f:
            data = json.load(f)

        summary = data["model_summary"]

        # ========== METRICS ROW ==========
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="Test R²",
                value=f"{summary['test_r2']:.3f}",
                help="R² score on held-out test set",
            )

        with col2:
            st.metric(
                label="CV R²",
                value=f"{summary['cv_r2_mean']:.3f}",
                help="5-fold cross-validation performance",
            )

        with col3:
            st.metric(
                label="CV Std",
                value=f"±{summary['cv_r2_std']:.3f}",
                help="5-fold cross-validation standard deviation",
            )

        with col4:
            st.metric(
                label="MAE",
                value=f"{summary['test_mae']:.3f}",
                help="Average prediction error in pIC50 units",
            )

        with col5:
            st.metric(
                label="Overfitting Gap",
                value=f"{summary['overfitting_gap']:.3f}",
                help="Difference between train and test R²",
            )

        st.markdown("---")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                "Training Samples",
                f"{summary['n_train_samples']:,}",
                help="Number of compounds used to train the model",
            )

        with col2:
            st.metric(
                "Test Samples",
                f"{summary['n_test_samples']:,}",
                help="Number of compounds used to evaluate the model",
            )

        with col3:
            st.metric(
                "Total Features",
                f"{summary['n_features']}",
                help="Total number of features used in the model",
            )

        with col4:
            st.metric(
                "Morgan FP",
                f"{summary['features_breakdown']['morgan_fp']}",
                help="Number of Morgan fingerprint features",
            )

        with col5:
            st.metric(
                "RDKit Descriptors",
                f"{summary['features_breakdown']['rdkit_descriptors']}",
                help="Number of RDKit descriptor features",
            )

        st.divider()

        # ========== INTERACTIVE PLOTS ==========
        st.markdown("### Performance Visualizations")

        # Plot selection dropdown
        plot_options = {
            "feature_importance": "Feature Importance (Top 20 - SHAP Based)",
            "shap_heatmap": "SHAP Feature Contribution Heatmap",
            "predictions_vs_actual": "Predictions vs Actual pIC50",
            "residuals": "Residuals Analysis",
            "error_distribution": "Prediction Error Distribution",
        }

        selected_plot = st.selectbox(
            "Select a visualization:",
            options=list(plot_options.keys()),
            format_func=lambda x: plot_options[x],
            index=0,  # Default to feature importance plot
        )

        # ========== RENDER SELECTED PLOT ==========

        if selected_plot == "predictions_vs_actual":
            st.markdown("#### Predictions vs Actual pIC50")

            pva_data = data["predictions_vs_actual"]

            # Create Plotly scatter plot
            fig = go.Figure()

            # Scatter points
            fig.add_trace(
                go.Scatter(
                    x=pva_data["actual"],
                    y=pva_data["predicted"],
                    mode="markers",
                    marker=dict(
                        size=6, color="#1976D2", opacity=0.6, line=dict(width=0.5, color="black")
                    ),
                    name="Predictions",
                    hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
                )
            )

            # Perfect prediction line
            perfect_line = pva_data["perfect_line"]
            fig.add_trace(
                go.Scatter(
                    x=[perfect_line["min"], perfect_line["max"]],
                    y=[perfect_line["min"], perfect_line["max"]],
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name="Perfect Prediction",
                    hoverinfo="skip",
                )
            )

            fig.update_layout(
                xaxis_title="Actual pIC50",
                yaxis_title="Predicted pIC50",
                height=500,
                hovermode="closest",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )

            st.plotly_chart(fig, width="stretch")

            st.caption(
                f"**R² Score: {pva_data['r2_score']:.4f}** | Points closer to the diagonal line indicate better predictions"
            )

        elif selected_plot == "residuals":
            st.markdown("#### Residuals Analysis")

            res_data = data["residuals"]

            fig = go.Figure()

            # Residuals scatter
            fig.add_trace(
                go.Scatter(
                    x=res_data["predicted"],
                    y=res_data["residuals"],
                    mode="markers",
                    marker=dict(
                        size=6, color="#2E7D32", opacity=0.6, line=dict(width=0.5, color="black")
                    ),
                    name="Residuals",
                    hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>",
                )
            )

            # Zero line
            fig.add_trace(
                go.Scatter(
                    x=[min(res_data["predicted"]), max(res_data["predicted"])],
                    y=[0, 0],
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name="Perfect Prediction (y=0)",
                    hoverinfo="skip",
                )
            )

            fig.update_layout(
                xaxis_title="Predicted pIC50",
                yaxis_title="Residual (Actual - Predicted)",
                height=500,
                hovermode="closest",
            )

            st.plotly_chart(fig, width="stretch")

            st.caption("**Ideal pattern:** Random scatter around zero line (no systematic bias)")

        elif selected_plot == "feature_importance":
            st.markdown("#### Feature Importance (Top 20 - SHAP Based)")

            fi_data = data["feature_importance"]
            fi_df = pd.DataFrame(fi_data["features"])

            fi_df_sorted = fi_df.sort_values(
                "importance", ascending=True
            )  # ← True because plotly will flip it

            # Create horizontal bar chart with color coding
            fig = px.bar(
                fi_df_sorted,
                y="feature",
                x="importance",
                orientation="h",
                color="type",
                color_discrete_map={"Morgan": "#64B5F6", "RDKit": "#81C784"},
                hover_data={"importance": ":.5f", "type": True},
            )

            gridline_config = fi_data.get("gridlines", {})
            if gridline_config.get("show", False):
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=gridline_config.get("width", 1),
                )

            fig.update_layout(
                xaxis_title="SHAP Mean |Value| (Prediction Impact)",
                yaxis_title="",
                height=600,
                showlegend=True,
                legend=dict(
                    title="Feature Type", yanchor="bottom", y=0.01, xanchor="right", x=0.99
                ),
                yaxis=dict(
                    autorange=True, categoryorder="total ascending"
                ),  # to ensure proper sort
            )

            st.plotly_chart(fig, width="stretch")

            st.info(f"""
            **Method:** {fi_data["method"]}  
            {fi_data["description"]}
            
            - **Morgan Bits:** Circular substructure patterns (SMILES notation after →)
            - **RDKit:** Physicochemical properties (MW, LogP, TPSA, etc.)
            """)

        elif selected_plot == "error_distribution":
            st.markdown("#### Prediction Error Distribution")

            err_data = data["error_distribution"]

            fig = go.Figure()

            bin_edges = err_data["histogram"]["bin_edges"]
            counts = err_data["histogram"]["counts"]

            # Calculate bin centers for x-axis
            bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))]
            bin_width = err_data["histogram"]["bin_width"]

            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=counts,
                    width=bin_width * 0.95,  # 95% of bin width to show borders
                    marker=dict(
                        color=err_data.get("color", "#FFB74D"),
                        line=dict(
                            color="rgba(255, 255, 255, 0.3)",  # ← Visible bin borders
                            width=1,
                        ),
                    ),
                    name="Error Distribution",
                    hovertemplate="Error: %{x:.3f}<br>Count: %{y}<extra></extra>",
                )
            )

            # Add mean line
            fig.add_vline(
                x=err_data["mean"],
                line_dash="dash",
                line_color="red",
                annotation_text=f"MAE: {err_data['mean']:.3f}",
                annotation_position="top right",
            )

            # Add median line
            fig.add_vline(
                x=err_data["median"],
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Median: {err_data['median']:.3f}",
                annotation_position="top left",
            )

            fig.update_layout(
                xaxis_title="Absolute Error (pIC50)",
                yaxis_title="Frequency (# samples)",
                height=400,
                showlegend=False,
                bargap=0.05,
            )

            st.plotly_chart(fig, width="stretch")

            # Error statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", f"{err_data['mean']:.3f}")
            col2.metric("Median Error", f"{err_data['median']:.3f}")
            col3.metric("Std Dev", f"{err_data['std']:.3f}")

        elif selected_plot == "shap_heatmap":
            st.markdown("#### SHAP Feature Contribution Heatmap")

            if data["shap_heatmap"] is not None:
                shap_data = data["shap_heatmap"]

                n_samples = shap_data["n_samples"]

                # Get all sample labels
                sample_labels_all = shap_data.get(
                    "sample_labels", [f"Sample_{i + 1}" for i in range(n_samples)]
                )

                # Define which sample NUMBERS you want to show (1-based)
                desired_sample_numbers = [1, 10, 20, 30, 40, 50]

                # Filter to only samples that exist
                desired_sample_numbers = [s for s in desired_sample_numbers if s <= n_samples]

                # Convert to 0-based indices
                tick_indices = [s - 1 for s in desired_sample_numbers]

                # Get the category labels at these positions
                tick_vals_categories = [sample_labels_all[i] for i in tick_indices]

                # Display labels (just the numbers)
                tick_labels = [str(s) for s in desired_sample_numbers]

                # Create heatmap
                fig = px.imshow(
                    shap_data["shap_matrix"],
                    labels=dict(x="Samples", y="Features", color="SHAP Value"),
                    x=sample_labels_all,
                    y=shap_data["feature_names"],
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                    color_continuous_midpoint=0,
                )

                fig.update_layout(
                    height=600,
                    xaxis=dict(
                        tickangle=0,
                        side="bottom",
                        title="Test Set Samples",
                        tickmode="array",
                        tickvals=tick_vals_categories,  # ["Sample_1", "Sample_10", "Sample_20", "Sample_30", "Sample_40", "Sample_50"]
                        ticktext=tick_labels,  # ["1", "10", "20", "30", "40", "50"]
                    ),
                    yaxis=dict(title="Top Features (by SHAP importance)", autorange="reversed"),
                    coloraxis_colorbar=dict(
                        title=dict(text="SHAP Value", side="right"),
                        tickmode="linear",
                        tick0=-max(abs(v) for row in shap_data["shap_matrix"] for v in row),
                        dtick=0.2,
                    ),
                )

                st.plotly_chart(fig, width="stretch")

                st.caption(f"""
                **Heatmap showing SHAP values for {shap_data["n_samples"]} test samples across top {shap_data["n_features"]} features**
                - **X-axis:** Individual test samples (each column is one molecule)
                - **Y-axis:** Features ranked by importance (highest at top)
                - **Red:** Positive contribution (increases pIC50 prediction)
                - **Blue:** Negative contribution (decreases pIC50 prediction)
                - Base value (average prediction): {shap_data["base_value"]:.3f}
                """)
            else:
                st.warning("SHAP heatmap data not available")

        st.markdown("")
        st.markdown("")
        # Load training metadata
        metadata_path = Path(__file__).parent / "qsar" / "saved_models" / "egfr_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

                with st.expander("Model Overview", expanded=False):
                    st.markdown("""
                    **What is EGFR?**
                    
                    EGFR (Epidermal Growth Factor Receptor) is a protein on cell surfaces that plays a crucial role in cell growth and survival. 
                    It's a major target for cancer therapeutics, particularly in lung cancer treatment. Binding of molecules to EGFR 
                    can block its signaling and inhibit tumor growth.
                    
                    **What is pIC50 & IC50?**
                    
                    - **IC50** (Half Maximal Inhibitory Concentration) = the concentration of a drug required to inhibit 50% of EGFR activity
                    - **pIC50** = -log₁₀(IC50), a logarithmic transformation that converts IC50 to a more convenient scale
                        - Higher pIC50 = Stronger binding (lower IC50)
                        - Example: pIC50 = 6 means IC50 = 1 µM (1 micromolar)
                    
                    **Model Used**
                    
                    This model is built with **XGBoost** (Extreme Gradient Boosting), a powerful ensemble learning algorithm that combines 
                    multiple decision trees to make predictions. XGBoost was selected because it:
                    - Handles non-linear relationships well
                    - Provides accurate predictions on unseen data
                    - Ranks features by importance (useful for understanding what matters in binding)
                    """)

                with st.expander("Training & Evaluation Setup", expanded=False):
                    total_samples = metadata["training_data"]["cleaned_molecules"]
                    train_samples = int(total_samples * 0.8)
                    test_samples = total_samples - train_samples
                    raw_mols = metadata["training_data"]["raw_molecules"]
                    cleaned_mols = metadata["training_data"]["cleaned_molecules"]

                    # Data Quality Overview
                    st.markdown("**Data Quality & Preparation**")
                    quality_col1, quality_col2, quality_col3 = st.columns(3)
                    with quality_col1:
                        st.metric(
                            "Raw Molecules", f"{raw_mols:,}", help="Initial molecules from ChEMBL"
                        )
                    with quality_col2:
                        st.metric(
                            "After Cleaning",
                            f"{cleaned_mols:,}",
                            help="Valid molecules for training",
                        )
                    with quality_col3:
                        st.metric(
                            "Data Retention",
                            metadata["training_data"]["retention_rate"],
                            help="Percentage kept after quality filtering (duplicates, invalid data removed)",
                        )

                    st.markdown("---")

                    # Dataset Composition
                    st.markdown("**Feature Engineering**")
                    feat_col1, feat_col2 = st.columns(2)
                    with feat_col1:
                        st.markdown("""
                        **Total Features:** 2,056
                        - Morgan Fingerprints: 2,048 bits
                        - RDKit Descriptors: 8 properties
                        """)
                    with feat_col2:
                        st.markdown("""
                        **Feature Types:**
                        - Substructure patterns
                        - Molecular properties
                        - Physicochemical characteristics
                        """)

                    st.markdown("---")

                    # Train/Test Split
                    st.markdown("**Model Evaluation Strategy**")
                    split_col1, split_col2, split_col3 = st.columns(3)
                    with split_col1:
                        st.metric(
                            "Training Set",
                            f"{train_samples:,}",
                            help=f"{metadata['train_test_split'].split('/')[0]}% of data",
                        )
                    with split_col2:
                        st.metric(
                            "Test Set",
                            f"{test_samples:,}",
                            help=f"{metadata['train_test_split'].split('/')[1]}% of data",
                        )
                    with split_col3:
                        st.metric(
                            "Validation",
                            "5-fold CV",
                            help="Cross-validation on training set: split training data into 5 folds, train on 4, evaluate on 1, repeat 5 times",
                        )

                    st.markdown("---")

                    # Model Performance
                    st.markdown("**Model Performance & Prediction Uncertainty**")
                    perf_col1, perf_col2 = st.columns(2)
                    with perf_col1:
                        st.metric("Model RMSE", "0.67 pIC50", help="Typical prediction error")
                    with perf_col2:
                        st.metric("CI Margin", "±1.27", help="95% confidence interval")

                with st.expander("EGFR Binding Affinity Scale", expanded=False):
                    st.markdown("""
                    **pIC50 Interpretation for EGFR (based on real drug discovery data):**
                    """)
                    scale_data = pd.DataFrame(
                        {
                            "pIC50 Range": ["< 3", "3–4", "4–5", "5–6", "6–7", "7–8", "> 8"],
                            "IC50 (µM equivalent)": [
                                "> 1000 nM",
                                "100–1000 nM",
                                "10–100 nM",
                                "1–10 nM",
                                "100–1000 pM",
                                "10–100 pM",
                                "< 10 pM",
                            ],
                            "Category": [
                                "Not measurable",
                                "Inactive",
                                "Weak",
                                "Moderate",
                                "Active/Lead-like",
                                "Strong",
                                "Excellent",
                            ],
                        }
                    )
                    st.dataframe(scale_data, width="stretch", hide_index=True)


def display_virtual_screening_results(results_df: pd.DataFrame) -> None:
    """Display virtual screening results with formatting, column configuration, and download button.

    Parameters:
        results_df (pd.DataFrame): Results DataFrame with screening results

    Returns:
        None. Renders Streamlit components directly.
    """
    st.subheader("Screened Molecules ")
    st.caption("Results are ranked by Predicted Activity")
    # Format results display
    display_df = results_df.copy()

    # Format numeric columns
    display_df["Predicted Activity (pIC50)"] = display_df["Predicted Activity (pIC50)"].round(2)
    display_df["QED Score"] = display_df["QED Score"].round(3)
    display_df["MW"] = display_df["MW"].round(1)
    display_df["LogP"] = display_df["LogP"].round(2)

    st.dataframe(
        display_df,
        column_config={
            "Predicted Activity (pIC50)": st.column_config.NumberColumn(
                "Predicted Activity",
                format="%.2f",
                help="QSAR-predicted pIC50 (higher = stronger binder)",
            ),
            "QED Score": st.column_config.NumberColumn(
                "QED", format="%.3f", help="Drug-likeness (0-1, higher is better)"
            ),
            "Lipinski Violations": st.column_config.NumberColumn(
                "Violations", help="Lipinski rule-of-5 violations"
            ),
            "MW": st.column_config.NumberColumn("Molecular Weight", format="%.1f", help="g/mol"),
            "LogP": st.column_config.NumberColumn("LogP", format="%.2f", help="Lipophilicity"),
        },
        width="stretch",
    )

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="virtual_screening_results.csv",
        mime="text/csv",
        width="content",
    )


def render_virtual_screening() -> None:
    """Render virtual screening interface.

    Allows users to upload CSV files with SMILES strings for batch processing.
    Performs feature computation, QSAR prediction, drug-likeness filtering,
    and returns ranked results by predicted activity.

    Returns:
        None. Renders Streamlit components directly.
    """
    # Clear cached results when switching pages
    if st.session_state.get("last_page") != "Virtual Screening":
        st.session_state.pop("screening_results_df", None)
        st.session_state.pop("screening_summary", None)
    st.session_state.last_page = "Virtual Screening"

    # ========== DESCRIPTION ==========

    st.subheader("Virtual Screening Pipeline (Batch QSAR Predictions)")

    st.markdown("""
    Upload a CSV file with SMILES strings to screen molecules against the EGFR QSAR model
    
    **Pipeline Steps:**
    1. Validate SMILES strings and compute molecular features
    2. Generate QSAR predictions (pIC50 binding affinity)
    3. Calculate drug-likeness metrics (QED score, Lipinski violations)
    4. Filter molecules by drug-likeness criteria (≤1 Lipinski violation)
    5. Rank results by predicted activity (descending)
    """)

    # Explain QED and Lipinski importance
    with st.expander("Why These Metrics Matter for Drug Discovery"):
        st.markdown("""
        ### QED Score (Quantitative Estimate of Drug-likeness)
        
        **What is it?** QED is a composite drug-likeness score ranging from 0-1 that combines multiple molecular properties 
        to predict how likely a compound is to become an approved drug. Higher scores indicate better drug-like properties.
        
        - **Components:** Molecular weight, LogP, H-bond donors/acceptors, rotatable bonds, aromatic/aliphatic rings
        - **Why it matters:** Compounds with high QED scores are more likely to have good absorption, distribution, and 
        overall bioavailability - critical for therapeutic efficacy
        
        ### Lipinski Violations
        
        **What is it?** Lipinski's Rule of Five defines molecular property thresholds for drug candidates:
        - Molecular Weight ≤ 500 Da
        - LogP (octanol-water partition) ≤ 5
        - H-bond Donors ≤ 5
        - H-bond Acceptors ≤ 10
        
        Violations occur when a compound exceeds these thresholds.
        
        **Why allow ≤1 violation?** While 0 violations is ideal, we allow up to 1 because:
        - Many successful drugs have 1 violation (e.g., aspirin has 0, but many antibiotics have 1)
        - Being too restrictive eliminates potentially active compounds
        - A single violation often doesn't severely impact bioavailability
        - This balance maximizes clinical potential while maintaining drug-likeness
        
        ### Ranking by Predicted Activity
        
        **Why?** After filtering for drug-likeness, we rank by predicted pIC50 (binding affinity) because:
        - Higher pIC50 = stronger binding to EGFR = better therapeutic potential
        - A drug must be both drug-like (passes filters) AND potent (high predicted activity)
        - This ranking shows the most promising candidates at the top
        
        ### MW and LogP on Results Table
        
        **Why include these?** These two metrics are critical indicators of:
        - **MW:** Molecular complexity and target penetration ability
        - **LogP:** Lipophilicity (how well it crosses membranes) and protein binding
        
        Together, they allow you to:
        1. Quickly assess drug-likeness without checking individual violation details
        2. Identify scaffolds for optimization (if MW too high, remove bulky groups; if LogP too high, add polar atoms)
        3. Make informed decisions about lead compound selection
        """)

    # Show file format example
    with st.expander("View CSV Format Example"):
        example_csv = """
        molecule_id,smiles
        mol1,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
        mol2,CCN(CC)CCOC(=O)C1=CC=CC=C1Cl
        mol3,CC(C)NCC(O)COc1ccccc1
        """
        st.code(example_csv, language="csv")
        with st.expander("Preview"):
            example_df = pd.DataFrame(
                {
                    "molecule_id": ["mol1", "mol2", "mol3"],
                    "smiles": [
                        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                        "CCN(CC)CCOC(=O)C1=CC=CC=C1Cl",
                        "CC(C)NCC(O)COc1ccccc1",
                    ],
                }
            )
            st.dataframe(example_df, hide_index=True, width="stretch")
        st.caption("Required columns: molecule_id (identifier), smiles (SMILES string)")

    st.markdown("")

    # ========== INPUTS AND CONTROLS ==========
    st.markdown("### Upload Data")

    # Load sample data button
    if st.button("Load Sample Data", help="Load example molecules for quick testing"):
        st.session_state.sample_smiles_path = "app/data/sample/screening_sample.csv"
        st.success("Sample data loaded!")

    st.markdown("**Or upload your CSV file:**")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="screening_file",
    )

    st.markdown("")

    # Determine data source
    csv_df = None
    data_source = None

    if "sample_smiles_path" in st.session_state:
        try:
            csv_df = pd.read_csv(st.session_state.sample_smiles_path)
            data_source = "sample"
        except Exception as e:
            st.error(f"Failed to load sample data: {e}")

    elif uploaded_file:
        try:
            csv_df = pd.read_csv(uploaded_file)
            data_source = "uploaded"
        except Exception as e:
            st.error(f"Failed to read CSV file: {e}")

    run_screening = st.button("Run Virtual Screening", type="primary", width="content")

    # ========== RESULTS ==========

    # Status indicator
    if data_source == "sample":
        st.info("ℹ️ Using sample data")
    elif data_source == "uploaded":
        st.info("ℹ️ Using uploaded data")

    # Run virtual screening
    if run_screening and csv_df is not None:
        try:
            with st.spinner("Running virtual screening pipeline..."):
                result = run_virtual_screening_pipeline(csv_df)

            if not result["success"]:
                st.error(f"❌ Screening failed: {result.get('error', 'Unknown error')}")
            else:
                # Display filtering summary
                st.success("✅ Virtual screening completed!")

                # Create summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Uploaded",
                        result["total_uploaded"],
                        help="Total molecules in input CSV",
                    )

                with col2:
                    st.metric(
                        "Invalid SMILES",
                        result["invalid_smiles"],
                        help="Molecules with invalid SMILES strings",
                    )

                with col3:
                    st.metric(
                        "Lipinski Filtered",
                        result["lipinski_filtered"],
                        help="Molecules with >1 Lipinski violation",
                    )

                with col4:
                    st.metric(
                        "Final Screened",
                        result["final_screened"],
                        help="Molecules passing all filters",
                    )

                # Store results in session state
                st.session_state.screening_results_df = result["results"]
                st.session_state.screening_summary = {
                    "total": result["total_uploaded"],
                    "invalid": result["invalid_smiles"],
                    "filtered": result["lipinski_filtered"],
                    "final": result["final_screened"],
                }

                st.divider()

                # Display results table
                if len(result["results"]) > 0:
                    display_virtual_screening_results(result["results"])

                else:
                    st.warning(
                        "No molecules passed the filtering criteria. Try adjusting input data."
                    )

        except Exception as e:
            st.error(f"Error during virtual screening: {e}")
            logger.exception(f"Virtual screening error: {e}")

    # Display cached results if available
    elif "screening_results_df" in st.session_state and not run_screening:
        results_df = st.session_state.screening_results_df
        summary = st.session_state.get("screening_summary", {})

        st.success("✅ Virtual screening completed!")

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Uploaded", summary.get("total", 0))
        with col2:
            st.metric("Invalid SMILES", summary.get("invalid", 0))
        with col3:
            st.metric("Lipinski Filtered", summary.get("filtered", 0))
        with col4:
            st.metric("Final Screened", summary.get("final", 0))

        st.divider()

        # Display results table
        if len(results_df) > 0:
            display_virtual_screening_results(results_df)

    elif run_screening and csv_df is None:
        st.warning("⚠️ Please load sample data or upload a CSV file to proceed")


def render_scaffold_sar() -> None:
    """Render Scaffold & SAR Explorer interface.

    Returns:
        None. Renders Streamlit components directly.
    """
    # Clear scaffold data when navigating to this page from another page
    if st.session_state.get("last_page") != "Scaffold & SAR Explorer":
        st.session_state.pop("scaffold_data", None)

    st.session_state.last_page = "Scaffold & SAR Explorer"

    st.subheader("Scaffold & SAR Explorer")

    st.markdown(
        """
    Analyze chemical scaffolds, identify structure-activity relationships (SAR),
    and detect activity cliffs (similar molecules with large potency differences).
    """
    )

    # Scaffolds, SAR, and Activity Cliffs explanation
    with st.expander("Scaffolds, SAR & Activity Cliffs: Why They Matter", expanded=False):
        st.markdown(
            """
            **What is a Molecular Scaffold?**
            
            A scaffold is the **core framework** of a molecule—the essential ring systems and linkers that define its structure. 
            When you remove all side chains and decorative groups, what remains is the scaffold.
            
            Example: All EGFR inhibitors share similar core scaffolds:
            - **Anilinoquinazoline core**: Ring systems that fit the EGFR binding pocket
            - **Linker regions**: Connect the core to substituents
            - **Substituent positions**: Where small chemical groups attach (Cl, F, Br, OH, OCH3, etc.)
            
            Molecules with the **same scaffold** typically talk to the **same biological target** and show related binding modes.
            
            ---
            
            **Why Know Scaffolds?**
            
            1. **Predict Drug Behavior**: Scaffold determines target selectivity and mechanism of action
            2. **Optimize Efficiently**: Change substituents on proven scaffolds rather than starting from scratch
            3. **Avoid Redundancy**: Don't synthesize compounds with identical scaffolds that will behave identically
            4. **Transfer Knowledge**: If you understand one scaffold, you can anticipate how modifications affect others
            
            ---
            
            **What is Structure-Activity Relationship (SAR)?**
            
            SAR is the **quantitative relationship** between molecular structure and biological activity. 
            It answers: *"How do small chemical changes affect binding affinity?"*
            
            Example SAR insights:
            - Adding a **fluorine** atom → 5× stronger binding
            - Extending a **methyl linker** → 2× weaker binding  
            - Switching **chlorine for bromine** → No change (similar size/polarity)
            
            **Why SAR Matters?**
            - Guides rational drug optimization
            - Predicts which modifications will improve potency
            - Identifies "activity plateaus" (changes that don't help)
            - Enables lean drug discovery workflows
            
            ---
            
            **What are Activity Cliffs?**
            
            Activity cliffs are **exceptional SAR patterns** where:
            - Two molecules have **very similar structures** (>0.85 Tanimoto fingerprint similarity)
            - But show **dramatically different** binding affinities (>10× potency ratio, ideally >100×)
            
            **Example Cliff Pair:**
            ```
            Compound A: CC(C)Nc1c(Cl)cc(cc1Nc2c(I)cc(N3CCOCC3)cc2)NC(=O)C
            IC50 = 0.8 nM (extremely potent)
            
            Compound B: CC(C)Nc1c(Cl)cc(cc1Nc2c(I)cc(N3CCOCC3)cc2)NC(=O)CC  ← Only added one CH2 to amide!
            IC50 = 98.5 nM (122× weaker!)
            ```
            
            That single ethyl group **destroyed binding** despite 99% structural similarity.
            
            ---
            
            **Why Activity Cliffs Matter?**
            
            1. **Defy Intuition**: Most chemistry principles say similar molecules = similar activity. Cliffs break this rule!
            2. **Hidden High-Value Targets**: Cliffs often indicate critical binding hotspots
            3. **Risk Detection**: A small synthetic misstep could lose 100× potency
            4. **Lead Optimization Gold**: Finding cliffs guides toward truly optimal compounds
            5. **Asset Differentiation**: Small changes on cliffs = BIG business advantage
            
            ---
            
            **Why Average IC50 is NOT Enough**
            
            ❌ **The Problem with Averages:**
            ```
            Scaffold A: 10 compounds with IC50 = [1, 1, 1, 1, 1, 100, 100, 100, 100, 100] nM
            Average IC50 = 50.5 nM (misleading!)
            
            Reality: 5 are excellent (1 nM), 5 are terrible (100 nM)
            The average hides the bimodal distribution!
            ```
            
            ✅ **Why Cliffs & SAR Analysis Reveal Truth:**
            - **Shows the distribution**: Which modifications work vs. fail
            - **Identifies bottlenecks**: Where the binding pocket is sensitive
            - **Enables optimization**: "If this small change causes 100× loss, what change causes 100× gain?"
            - **De-risks decisions**: Avoid compounds similar to known weak binders
            - **Focuses screening**: Prioritize compounds that might hit cliff pairs
            
            **The Real Challenge in Drug Discovery:**
            It's not finding *one* good compound. It's finding the **right modifications** to get from good (1 μM) to great (1 nM). 
            Activity cliffs highlight exactly where those critical modifications lie.
            
            ---
            
            **Key Takeaway**
            
            🎯 By analyzing scaffolds and identifying activity cliffs, you transform drug discovery from **trial-and-error** 
            into **targeted optimization**. This tool helps you see the hidden patterns that make the difference between a failed candidate and a blockbuster drug.
            """
        )

    # About sample dataset
    with st.expander("About Our Sample Dataset", expanded=False):
        st.markdown(
            """
            **Dataset Composition:**
            - **18 Real EGFR Inhibitors**: FDA-approved drugs (Erlotinib, Gefitinib, Afatinib, Dacomitinib, Neratinib, etc.)
            - **~75 Structural Analogs**: Real scaffolds with estimated IC50 values based on chemistry principles
            - **~30 Activity Cliff Pairs**: Designed molecules showing >0.85 similarity but >100× potency differences
            
            **Naming Strategy:**
            - `{DrugName}_CHEMBL{ID}_Reference` - Real compounds from ChEMBL
            - `{DrugName}_Analog_{Modification}` - Analogs e.g., "Erlotinib_Analog_Naphthyl"
            - `CLIFF_{Letter}_Strong/Weak_{Description}` - Cliff pairs e.g., "CLIFF_A_Strong_acetamide"
            
            **Why This Dataset?**
            - Real compounds provide credibility and educational value
            - Analogs show realistic SAR patterns (estimated IC50 from structure-activity principles)
            - Cliff pairs ensure reproducible detection examples—activity cliffs are rare (~5-10% naturally)
            - Single EGFR target maintains chemical coherence
            
            **Why Include Theoretical Molecules & Hypothetical IC50 Values?**
            
            Activity cliffs are **rare phenomena** in real bioactivity databases (typically <10% of comparable pairs). 
            To effectively demonstrate this tool's cliff detection algorithms, we synthetically designed pairs with:
            
            1. **Guaranteed High Similarity** (>0.85 Tanimoto) - Real analogs often have lower similarity
            2. **Dramatic Potency Ratios** (>100×) - Ensures clear before/after SAR visualization
            3. **Medicinal Chemistry Logic** - Changes follow real drug discovery principles:
               - Small substituent removal (Cl, I, Br) → Big affinity drop
               - Ring system modification (lactam opening) → Binding disruption
               - Linker expansion (acetamide → ethylamide) → Steric clash
            
            **How IC50 Values Are Estimated:**
            - Reference drugs use published ChEMBL affinities (real data)
            - Analogs estimated via: Chemical similarity scoring + Known SAR trends + Medicinal chemistry heuristics
            - Not experimental measurements, but chemically plausible based on:
              - Binding pocket models
              - Ligand efficiency trends
              - Literature precedent for similar scaffolds
            
            **Real-World Analogy:**
            Just as ML datasets use synthetic examples to train on rare phenomena (MNIST for handwriting, 
            GAN data for edge cases), we use theoretical compounds to create reproducible cliff examples. 
            This lets you validate detection algorithms on guaranteed cliffs before testing real data.
            
            **Key Features:**
            - 123 total compounds suitable for SAR visualization
            - All SMILES validated with RDKit
            - Demonstrates how small scaffolding changes cause dramatic binding effects
            
            ⚠️ *Note: Analog IC50 values are estimated; cliff pairs are theoretically designed. For real drug discovery, use ChEMBL/PubChem/BindingDB.*
            """
        )

    # Display CSV format guide on main page
    with st.expander("CSV Format Guide", expanded=False):
        st.markdown(
            """
            **Required columns:**
            - `smiles` - SMILES string representation of the molecule
            
            **Optional columns:**
            - `molecule_id` - Unique identifier (auto-generated if missing)
            - `standard_value` - IC50 binding affinity in nM (auto-fetched from sample data if missing)
            
            **Example:**
            """
        )
        example_csv = """
        molecule_id,smiles,standard_value
        MOL_001,CC(=O)OC1=CC=CC=C1C(=O)O,50.0
        MOL_002,CC(=O)OC1=CC=CC=C1C(=O)N,500.0
        MOL_003,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,10.0
        """

        st.code(example_csv, language="csv")
        with st.expander("Preview"):
            # Create example dataframe
            example_data = {
                "molecule_id": ["MOL_001", "MOL_002", "MOL_003"],
                "smiles": [
                    "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "CC(=O)OC1=CC=CC=C1C(=O)N",
                    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                ],
                "standard_value": [50.0, 500.0, 10.0],
            }
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, hide_index=True, width="stretch")

    # Data loading
    st.markdown("### Upload Data")

    if st.button("Load Sample Data", help="Load example IC50 data for quick testing"):
        with st.spinner("Loading sample IC50 data..."):
            df = load_sample_ic50_data()

        if df is None or df.empty:
            st.error("Failed to load sample dataset")
            return

        if "molecule_id" not in df.columns:
            df["molecule_id"] = [f"MOL_{i:04d}" for i in range(len(df))]

        st.session_state.scaffold_data = df
        st.success(f"✅ Loaded {len(df)} molecules")

    # File uploader for user's dataset
    st.markdown("Or Upload your CSV file")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type="csv",
        key="scaffold_csv_upload",
        help="CSV must contain 'smiles' column. 'standard_value' (IC50 in nM) is optional—will be fetched if missing.",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "smiles" not in df.columns:
                st.error("CSV must contain 'smiles' column")
                return

            # If standard_value is missing, attempt to fetch from sample dataset
            if "standard_value" not in df.columns:
                st.warning(
                    "⚠️ 'standard_value' column not found. Attempting to fetch IC50 values..."
                )

                df = fetch_missing_ic50_values(df)

                if "standard_value" not in df.columns or df["standard_value"].isna().all():
                    st.error(
                        "Could not retrieve IC50 values. Please provide 'standard_value' column in your CSV."
                    )
                    return
                elif df["standard_value"].isna().any():
                    st.warning(
                        f"⚠️ Could only match IC50 values for {df['standard_value'].notna().sum()}/{len(df)} molecules. "
                        f"Consider providing 'standard_value' for all molecules."
                    )

            if "molecule_id" not in df.columns:
                df["molecule_id"] = [f"MOL_{i:04d}" for i in range(len(df))]

            st.session_state.scaffold_data = df
            st.success(f"✅ Loaded {len(df)} molecules")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return

    # Get data from session state if available
    df = st.session_state.get("scaffold_data", None)

    # Display analysis only if data is loaded
    if df is None or df.empty:
        st.info("Upload a CSV file or load the sample dataset to begin analysis.")
        return

    try:
        # Display IC50 summary statistics
        st.subheader("Data Summary")
        stats = get_ic50_summary_stats(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Molecules", stats["total_molecules"])
        with col2:
            st.metric("With IC50 Data", stats["with_ic50"])
        with col3:
            st.metric("Coverage (%)", f"{stats['coverage_percent']:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Activity Range", stats["activity_range"])
        with col2:
            st.metric("Median IC50", f"{stats['median_ic50']:.1f} nM")

        # Add scaffolds to dataframe
        with st.spinner("Extracting scaffolds..."):
            df_with_scaffolds = add_scaffolds_to_dataframe(df)

        if df_with_scaffolds.empty:
            st.error("No valid scaffolds found")
            return

        st.success(f"Extracted scaffolds for {len(df_with_scaffolds)} molecules")

        # Display scaffold groups
        st.subheader("Scaffold Groups")
        scaffold_summary = summarize_scaffolds(df_with_scaffolds)

        if scaffold_summary.empty:
            st.warning("No scaffold groups found")
        else:
            st.metric("Unique Scaffolds", len(scaffold_summary))

            # Display table with formatting
            display_df = scaffold_summary.reset_index(drop=True).copy()
            display_df["avg_activity"] = display_df["avg_activity"].round(2)
            display_df["min_activity"] = display_df["min_activity"].round(2)
            display_df["max_activity"] = display_df["max_activity"].round(2)

            st.dataframe(
                display_df,
                column_config={
                    "scaffold": st.column_config.TextColumn("Scaffold SMILES", width="large"),
                    "molecule_count": st.column_config.NumberColumn("Molecules"),
                    "avg_activity": st.column_config.NumberColumn("Avg IC50 (nM)"),
                    "min_activity": st.column_config.NumberColumn("Min IC50 (nM)"),
                    "max_activity": st.column_config.NumberColumn("Max IC50 (nM)"),
                },
                width="stretch",
            )

            # Download scaffold summary
            csv_scaffold = scaffold_summary.to_csv(index=False)
            st.download_button(
                label="Download Scaffold Summary",
                data=csv_scaffold,
                file_name="scaffold_summary.csv",
                mime="text/csv",
                width="content",
            )

        st.markdown("")
        st.markdown("")

        # Activity cliff detection
        st.subheader("Activity Cliff Detection")
        st.markdown(
            "Activity cliffs are pairs of similar molecules with large differences in potency. "
            "They provide insights into which structural features affect binding affinity."
        )

        col1, col2 = st.columns(2)
        with col1:
            sim_threshold = st.slider(
                "Similarity Threshold",
                0.50,
                1.0,
                0.85,
                step=0.01,
                help="Minimum Tanimoto similarity",
            )
        with col2:
            ratio_threshold = st.slider(
                "Activity Ratio Threshold",
                10.0,
                500.0,
                100.0,
                step=10.0,
                help="Minimum IC50 ratio (fold change)",
            )

        if st.button("Detect Activity Cliffs", type="primary", width="content"):
            with st.spinner("Detecting activity cliffs..."):
                cliffs_df = detect_activity_cliffs(
                    df_with_scaffolds,
                    similarity_threshold=sim_threshold,
                    activity_ratio_threshold=ratio_threshold,
                )

            if len(cliffs_df) > 0:
                st.success(f"Found {len(cliffs_df)} activity cliffs")

                # Display cliffs table
                display_cliffs = cliffs_df[
                    [
                        "mol1",
                        "mol2",
                        "similarity",
                        "activity_ratio",
                        "ic50_molecule_1",
                        "ic50_molecule_2",
                    ]
                ].copy()

                st.dataframe(
                    display_cliffs,
                    column_config={
                        "mol1": st.column_config.TextColumn("Molecule 1"),
                        "mol2": st.column_config.TextColumn("Molecule 2"),
                        "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
                        "activity_ratio": st.column_config.NumberColumn(
                            "Activity Ratio", format="%.1f x"
                        ),
                        "ic50_molecule_1": st.column_config.NumberColumn(
                            "IC50 Molecule 1 (nM)", format="%.2f"
                        ),
                        "ic50_molecule_2": st.column_config.NumberColumn(
                            "IC50 Molecule 2 (nM)", format="%.2f"
                        ),
                    },
                    width="stretch",
                    hide_index=True,
                )

                # Download cliffs
                csv_cliffs = cliffs_df.to_csv(index=False)
                st.download_button(
                    label="Download Activity Cliffs",
                    data=csv_cliffs,
                    file_name="activity_cliffs.csv",
                    mime="text/csv",
                    width="content",
                )
            else:
                st.info(
                    "💡 No activity cliffs found with current thresholds. **Tip:** Try lowering the similarity threshold (e.g., 0.70-0.75) or activity ratio threshold (e.g., 50-75) to find more cliff pairs. "
                    "Activity cliffs are relatively rare—they represent pairs of structurally similar molecules with dramatically different potencies, which is valuable but not common in single scaffold groups."
                )

    except Exception as e:
        logger.exception(f"Error in Scaffold & SAR Explorer: {e}")
        st.error(f"An error occurred: {str(e)}")


def render_overview():

    st.subheader("MoleculeInsight")

    st.markdown(
        "MoleculeInsight is a **cheminformatics analysis platform** for molecular exploration, prediction, and screening."
    )

    st.divider()

    st.markdown(
        """
        <style>
        .tool-card {
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .tool-card:hover {
            transform: translateY(-3px);
        }

        .tool-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="tool-card">
            <div class="tool-title">Single Molecule</div>
            Analyze molecular properties, visualize structures, and explore known bioactivity for a single compound.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="tool-card">
            <div class="tool-title">Virtual Screening</div>
            Batch screen molecules through the QSAR pipeline with drug-likeness filtering and activity ranking.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="tool-card">
            <div class="tool-title">Similarity Search</div>
            Find structurally similar molecules using molecular fingerprints and similarity scoring.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="tool-card">
            <div class="tool-title">Scaffold & SAR Explorer</div>
            Extract Murcko scaffolds, identify activity cliffs, and analyze structure–activity relationships.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="tool-card">
            <div class="tool-title">QSAR Model</div>
            Predict EGFR binding affinity (pIC50) using an XGBoost QSAR model trained on ChEMBL bioactivity data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========== THE STORY ==========
    st.divider()
    st.markdown("### MoleculeInsight: The Drug Discovery Journey")

    st.info(
        "🎯 **Current Focus:** MoleculeInsight is optimized for **EGFR (Epidermal Growth Factor Receptor)** target discovery. The QSAR model and all predictions are currently trained for EGFR binding affinity. Future versions will support additional targets."
    )

    st.markdown("""
    **Imagine you're a medicinal chemist trying to discover a new EGFR inhibitor.** You have ONE interesting candidate molecule 
    and want to understand it, find similar compounds, predict effectiveness, and screen thousands of candidates.
    
    That's the entire journey MoleculeInsight takes you through:
    """)

    st.markdown("#### The Drug Discovery Cycle")

    # Circular flow diagram using Plotly
    labels = [
        "💡 Idea",
        "🔬 Single\nMolecule",
        "🔎 Similarity\nSearch",
        "📊 QSAR\nPredict",
        "⚡ Screening",
        "🔗 SAR",
    ]

    n = len(labels)
    angles = np.linspace(np.pi, np.pi - 2 * np.pi, n, endpoint=False)

    # Increase radius for better spacing
    radius = 3.2
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    fig = go.Figure()

    # Smooth circular connections with arrows
    for i in range(n):
        x0, y0 = x[i], y[i]
        x1, y1 = x[(i + 1) % n], y[(i + 1) % n]

        # Calculate offset to avoid overlap with boxes
        dx = x1 - x0
        dy = y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        offset = 0.5

        if dist > 0:
            dx_norm = dx / dist
            dy_norm = dy / dist
            # Shorten line to avoid boxes
            x0_line = x0 + dx_norm * offset
            y0_line = y0 + dy_norm * offset
            x1_line = x1 - dx_norm * offset
            y1_line = y1 - dy_norm * offset
        else:
            x0_line, y0_line, x1_line, y1_line = x0, y0, x1, y1

        fig.add_shape(
            type="line",
            x0=x0_line,
            y0=y0_line,
            x1=x1_line,
            y1=y1_line,
            line=dict(color="rgba(100,100,100,0.4)", width=2),
            layer="below",
        )

        # Add arrow annotation at midpoint
        mid_x = (x0_line + x1_line) / 2
        mid_y = (y0_line + y1_line) / 2
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            ax=x0_line,
            ay=y0_line,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="rgba(100,100,100,0.6)",
        )

    # Draw rounded rectangles using shapes + text
    for i, (xi, yi, label) in enumerate(zip(x, y, labels, strict=True)):
        node_color = "#22c55e" if i == 0 else "rgba(102,126,234,0.9)"
        node_text = ("🟢 START:" + label) if i == 0 else label

        w, h = 2.2, 1  # width & height of box
        r = 0.2  # corner radius

        # Rounded rectangle path
        path = f"""
        M {xi - w / 2 + r},{yi - h / 2}
        L {xi + w / 2 - r},{yi - h / 2}
        Q {xi + w / 2},{yi - h / 2} {xi + w / 2},{yi - h / 2 + r}
        L {xi + w / 2},{yi + h / 2 - r}
        Q {xi + w / 2},{yi + h / 2} {xi + w / 2 - r},{yi + h / 2}
        L {xi - w / 2 + r},{yi + h / 2}
        Q {xi - w / 2},{yi + h / 2} {xi - w / 2},{yi + h / 2 - r}
        L {xi - w / 2},{yi - h / 2 + r}
        Q {xi - w / 2},{yi - h / 2} {xi - w / 2 + r},{yi - h / 2}
        Z
        """

        fig.add_shape(
            type="path",
            path=path,
            fillcolor=node_color,
            line=dict(color="rgba(255,255,255,0.6)", width=1),
            layer="above",
        )

        # Add text as annotation
        fig.add_annotation(
            x=xi,
            y=yi,
            text=node_text,
            showarrow=False,
            font=dict(size=13, color="white"),
            xanchor="center",
            yanchor="middle",
            align="center",
        )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[-4.8, 4.8]),
        yaxis=dict(visible=False, range=[-3.5, 3.5]),
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(t=5, b=5, l=5, r=5),
    )

    fig.update_xaxes(scaleanchor="y", scaleratio=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Each Tool Explained")
    st.markdown("""
    **1️⃣ Single Molecule Analysis — Your Magnifying Glass**
    - You have ONE candidate and want to understand everything about it
    - Input: SMILES code (text shorthand for molecular structures)
    - Get: Structure visualization, molecular properties (MW, LogP, TPSA), Lipinski drug-likeness check, known bioactivity from ChEMBL
    - When to use: Early stage - you found an interesting compound in the lab or literature
    
    **2️⃣ Similarity Search — Your Library Scout**
    - Found a good molecule? Ask: "Are there similar ones in my library?"
    - Input: Query molecules + reference library (both as CSV with SMILES)
    - Get: Ranking table showing which library compounds match your query (0.0 = different, 1.0 = identical)
    - Uses: Morgan fingerprints (2048-bit codes representing molecular shape) + Tanimoto similarity scoring
    - When to use: You have a lead compound and want to find analogs
    
    **3️⃣ QSAR Model — Your Prediction Engine**
    - The brain of MoleculeInsight - a machine learning model trained on thousands of real bioactivity experiments
    - Input: SMILES string
    - Get: Predicted binding affinity to EGFR (pIC50), SHAP feature importance (which parts of your molecule matter most)
    - Model: XGBoost trained on ChEMBL EGFR data with 2,056 molecular features (Morgan fingerprints + RDKit descriptors)
    - When to use: Estimate EGFR binding strength before lab testing
    
    **4️⃣ Virtual Screening — Your Batch Processor**
    - You have a CSV with THOUSANDS of molecules and want to rank them all at once
    - Input: CSV with molecule IDs and SMILES strings
    - Get: Scored ranking of top hits (by predicted pIC50), filtered for drug-likeness (Lipinski compliance)
    - Features: Validates SMILES, computes features, predicts binding, filters out unlikely drugs, ranks results
    - When to use: High-throughput screening of large compound libraries
    
    **5️⃣ Scaffold & SAR Explorer — Your Pattern Detective**
    - You have a dataset of active compounds with known IC50 values (experimental data)
    - Ask: "What core scaffolds are common? Which structural changes increase/decrease activity?"
    - Input: CSV with SMILES and IC50 values
    - Get: Unique scaffolds (core structures), activity statistics per scaffold, activity cliffs (similar molecules with big potency differences)
    - When to use: Understand structure-activity relationships (SAR) - what chemical features drive EGFR binding strength
    """)

    st.markdown("#### Quick Reference: When to Use Each Tool")

    quick_ref = pd.DataFrame(
        [
            {
                "Your Question": "What is this molecule?",
                "Use This Tool": "1️⃣ Single Molecule",
                "Input": "SMILES string",
                "You Get": "Properties, bioactivity, drug-likeness check",
            },
            {
                "Your Question": "Are there similar EGFR inhibitors?",
                "Use This Tool": "2️⃣ Similarity Search",
                "Input": "Query SMILES + Library CSV",
                "You Get": "Ranked similar molecules with structural images",
            },
            {
                "Your Question": "Will it bind to EGFR?",
                "Use This Tool": "3️⃣ QSAR Model",
                "Input": "SMILES string",
                "You Get": "Predicted binding affinity (pIC50) + importance scores",
            },
            {
                "Your Question": "Screen 10,000 compounds?",
                "Use This Tool": "4️⃣ Virtual Screening",
                "Input": "CSV with many SMILES",
                "You Get": "Ranked top hits (drug-like only)",
            },
            {
                "Your Question": "What makes strong EGFR binders?",
                "Use This Tool": "5️⃣ Scaffold & SAR",
                "Input": "IC50 dataset",
                "You Get": "Core scaffolds + activity patterns + cliffs",
            },
        ]
    )

    st.dataframe(quick_ref, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown(
        "**Ready to get started?** Pick a tab above to begin your EGFR drug discovery journey!"
    )


def with_footer(page_func):
    def wrapper():
        # Wrap page + footer in a single flex container
        st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)

        # Content wrapper
        st.markdown('<div class="content">', unsafe_allow_html=True)
        # Page content
        page_func()
        st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown(
            """
            <div class='footer'>
                © 2026 <a href='https://sinembudak.com/' target='_blank'>Sinem Demirkaya-Budak</a> |
                <a href='https://github.com/sinemdemirkayabudak/moleculeinsight' target='_blank'>Source Code</a> 
            </div>
            </div> <!-- end page-wrapper -->
            """,
            unsafe_allow_html=True,
        )

    return wrapper


def render_app() -> None:
    """Main application renderer with navigation.

    Displays the application header, navigation buttons, and routes to appropriate
    page based on user selection (Single Molecule or Similarity Search).

    Returns:
        None. Renders Streamlit components directly.
    """

    # Inject SEO meta tags
    st.components.v1.html(
        """
        <script>
        const addMetaTag = (name, content, isProperty = false) => {
            const attr = isProperty ? 'property' : 'name';
            if (!document.querySelector(`meta[${attr}="${name}"]`)) {
                const meta = document.createElement('meta');
                meta.setAttribute(attr, name);
                meta.content = content;
                document.head.appendChild(meta);
            }
        };
        
        // SEO
        addMetaTag('description', 'MoleculeInsight: Predict EGFR binding affinity, screen compound libraries, and analyze structure-activity relationships using validated XGBoost QSAR models. Features similarity search, scaffold analysis, and SHAP explainability.');
        addMetaTag('keywords', 'QSAR, drug discovery, molecular similarity, cheminformatics, EGFR, XGBoost, virtual screening');
        addMetaTag('author', 'Sinem Demirkaya-Budak');
        
        // Open Graph
        addMetaTag('og:title', 'MoleculeInsight - AI-Powered Drug Discovery', true);
        addMetaTag('og:description', 'Predict EGFR binding affinity using QSAR models.', true);
        addMetaTag('og:type', 'website', true);
        addMetaTag('og:url', 'https://sinembudak.com', true);
        </script>
        """,
        height=0,
    )

    st.markdown(
        """
        <style>
        /* Keep block container spacing for content */ 
                .block-container { 
                max-width: 80vw !important;  /* 80% of viewport width */
                padding-top: 1.5rem; 
                padding-bottom: 1rem; 
                margin-top: 1rem; 
                margin-bottom: 4rem; }

        /* Full viewport height wrapper */
        .page-wrapper {
            display: flex;
            flex-direction: column;
            min-height: 100%
        }
        .footer {
            bottom: 0;
            left: 0;
            width: 100%;
            z-index: 999;

            text-align: center;
            padding: 20px 0;
            font-size: 13px;

            backdrop-filter: blur(10px);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Define pages ----
    pages = [
        st.Page(
            with_footer(render_overview),
            title=" MOLECULEINSIGHT ",
            icon="💊",
            default=True,
            url_path="home",
        ),
        st.Page(
            with_footer(render_single_molecule), title="Single Molecule", url_path="single-molecule"
        ),
        st.Page(
            with_footer(render_similarity_search),
            title="Similarity Search",
            url_path="similarity-search",
        ),
        st.Page(with_footer(render_qsar_dashboard), title="QSAR Model", url_path="qsar-model"),
        st.Page(
            with_footer(render_virtual_screening),
            title="Virtual Screening",
            url_path="virtual-screening",
        ),
        st.Page(
            with_footer(render_scaffold_sar),
            title="Scaffold & SAR Explorer",
            url_path="scaffold-sar-explorer",
        ),
    ]

    # ---- Navigation ----
    pg = st.navigation(pages, position="top")

    # Run selected page
    pg.run()

# User Interface for Streamlit app - handles rendering of components,
# user interactions, and display logic.
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rdkit.Chem import Draw

from app.chembl import get_compound_bioactivity_from_mol
from app.config import logger
from app.molecule import get_molecule, get_rdkit_properties, lipinski_rules
from app.pubchem import get_pubchem_metadata
from app.qsar.features import compute_morgan_fingerprints, compute_rdkit_descriptors
from app.qsar.predict import QSARPredictor
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

    # Caption and button on same line
    col1, col2 = st.columns([6, 1], gap="small")

    with col1:
        st.caption(
            "💡 Tip: Click on an image to see a zoomed-in view of the query and reference molecules"
        )

    with col2:
        csv = prepare_csv_export(results_df)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="similarity_results.csv",
            mime="text/csv",
            width="content",
        )

    display_df = results_df.drop(columns=["Reference SMILES", "Query SMILES"], errors="ignore")
    st.dataframe(
        display_df,
        column_config={"Structures": st.column_config.ImageColumn("Structures", width="medium")},
        width="stretch",
    )


def display_ranking_plots(plots_dict: dict[str, Figure]) -> None:
    """Display ranking plots with query dropdown selector.

    Parameters:
        plots_dict (dict): Dictionary mapping query_name -> matplotlib figure object

    Returns:
        None. Renders Streamlit components directly. Returns early if plots_dict is empty.
    """
    if not plots_dict:
        return

    st.subheader("Similarity Ranking Plots")
    unique_queries = list(plots_dict.keys())

    if len(unique_queries) > 0:
        selected_query = st.selectbox(
            "Select query molecule to view ranking plot:", unique_queries, key="query_plot_select"
        )

        if selected_query in plots_dict:
            fig = plots_dict[selected_query]
            st.pyplot(fig)
            plt.close(fig)


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

    Allows users to input SMILES or select example molecules via sidebar.
    Displays molecular properties, Lipinski rule-of-5 compliance, and bioactivity data from ChEMBL.

    Returns:
        None. Renders Streamlit components directly.
    """
    # Clear cached similarity search results when switching pages
    if st.session_state.get("last_page") == "Similarity Search":
        st.session_state.pop("results_df", None)
        st.session_state.pop("query_plots", None)
        st.session_state.pop("sample_query_path", None)
        st.session_state.pop("sample_ref_path", None)
    st.session_state.last_page = "Single Molecule"

    # Example molecules dropdown
    examples = {
        "None": "",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Ethanol": "CCO",
        "Benzene": "c1ccccc1",
        "Naproxen": "COc1ccc2cc(ccc2c1)[C@@H](C)C(=O)O",
        "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    }

    selected_example = st.sidebar.selectbox(
        "Load Example Molecule:",
        options=["None"] + sorted([k for k in examples.keys() if k != "None"]),
        index=0,
        key="example_select",
    )

    # Get SMILES from selected example or from user input
    if selected_example != "None":
        initial_smiles = examples[selected_example]
    else:
        initial_smiles = st.session_state.get("example_smiles", "")

    smiles = st.sidebar.text_input(
        "Enter SMILES:",
        value=initial_smiles,
        label_visibility="visible",
        help="SMILES: Simplified Molecular Input Line Entry System",
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
                    st.subheader("Properties")
                    st.write(f"- IUPAC Name: {meta.get('iupac', 'N/A')}")
                    st.write(f"- Common Name: {meta.get('common', 'N/A')}")
                    st.write(f"- CID: {meta.get('cid', 'N/A')}")
                    st.write(f"- InChIKey: {meta.get('inchikey', 'N/A')}")
                    st.write(f"- Molecular Weight (MW): {mw:.2f}")
                    st.write(f"- LogP (octanol-water): {logp:.2f}")
                    st.write(f"- Topological Polar Surface Area (TPSA): {tpsa:.2f}")
                    st.write(f"- H-bond Donors (HBD): {hbd}")
                    st.write(f"- H-bond Acceptors (HBA): {hba}")
                    st.write(f"- Rotatable Bonds: {rotb}")

        with tab2:
            st.subheader("Lipinski Rule-of-5 Compliance")
            for rule, passed in rules.items():
                st.text(f"- {rule}   \t{'✔ Passed' if passed else '✘ Violated'}")
            st.write("\n")
            st.write(f"Total violations: {violations}")

        with tab3:
            st.subheader("Bioactivity Evidence (ChEMBL)")

            # Bioactivity records limit control
            bioactivity_limit = st.slider(
                "Bioactivity Records Limit",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="Number of bioactivity records to retrieve from ChEMBL",
            )

            bioactivity_data = get_compound_bioactivity_from_mol(mol, limit=bioactivity_limit)
            if bioactivity_data.get("success"):
                activities = bioactivity_data.get("bioactivity", {}).get("activities", [])
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

    # ========== SIDEBAR: INPUTS AND CONTROLS ==========
    st.sidebar.markdown("### Settings")

    # Load sample data button
    if st.sidebar.button("Load Sample Data", help="Load example molecules for quick testing"):
        sample_query = "app/data/sample/query_molecules.csv"
        sample_ref = "app/data/sample/reference_library.csv"
        st.session_state.sample_query_path = sample_query
        st.session_state.sample_ref_path = sample_ref
        st.sidebar.success("Sample data loaded!")

    st.sidebar.markdown("**Or upload your own CSV files:**")

    # File uploads in sidebar
    query_file = st.sidebar.file_uploader("Query molecules (CSV)", key="query_file")
    ref_file = st.sidebar.file_uploader("Reference library (CSV)", key="ref_file")

    st.sidebar.markdown("---")

    # Parameter controls in sidebar
    radius = st.sidebar.number_input(
        "Fingerprint Radius",
        min_value=0,
        max_value=5,
        value=2,
        help="0=atoms, 1=neighbors, 2=ECFP4 (standard), 3+=extended",
    )

    top_n = st.sidebar.number_input("Top N Results", min_value=1, max_value=100, value=20)

    show_plots = st.sidebar.checkbox("Show Similarity Ranking Plots", value=True)

    st.sidebar.markdown("---")

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

    run_search = st.sidebar.button("Run Similarity Search", type="primary", width="stretch")

    # ========== MAIN AREA: DESCRIPTION AND RESULTS ==========
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
            query_example = pd.DataFrame(
                {
                    "smiles": ["CCO", "CCN", "CCC", "CC(=O)O", "c1ccccc1"],
                    "name": ["Ethanol", "Ethylamine", "Propane", "AceticAcid", "Benzene"],
                }
            )
            st.dataframe(query_example, hide_index=True, width="stretch")

        with col2:
            st.subheader("Reference Library Example")
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
            st.divider()
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
    st.session_state.last_page = "QSAR Model"

    st.subheader("EGFR pIC50 Prediction Model")

    st.markdown("""
    XGBoost QSAR model trained on ChEMBL bioactivity data to predict EGFR binding affinity (pIC50) from molecular structures.
    """)

    # Create 3 tabs
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Make Predictions", "Learn More"])

    # ========== TAB 1: MODEL PERFORMANCE OVERVIEW ==========
    with tab1:
        st.markdown("""
        This XGBoost QSAR model predicts **EGFR binding affinity (pIC50)** from molecular structures.
        Trained on ChEMBL bioactivity data with cross-validated performance metrics.
        """)

        # Load and display performance metrics
        metrics_path = Path(__file__).parent / "qsar" / "saved_models" / "egfr_performance.json"
        if metrics_path.exists():
            import json

            with open(metrics_path) as f:
                metrics = json.load(f)

            # Extract XGBoost metrics (best model)
            test_r2 = metrics.get("test_metrics", {}).get("xgb_r2", 0)
            cv_r2_mean = metrics.get("cv_metrics", {}).get("xgb_cv_r2_mean", 0)
            cv_r2_std = metrics.get("cv_metrics", {}).get("xgb_cv_r2_std", 0)

            # Display key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test R²", f"{test_r2:.4f}")
            with col2:
                st.metric("Test MAE", "0.573 pIC50", help="Mean Absolute Error on test set")
            with col3:
                st.metric("CV R² (mean)", f"{cv_r2_mean:.4f}")
            with col4:
                st.metric("CV R² (std)", f"±{cv_r2_std:.4f}", help="5-fold cross-validation")

        st.markdown("---")
        st.markdown("### Performance Plots")

        # Define all plots
        plots_dir = Path(__file__).parent / "qsar" / "visualizations"
        plot_files = {
            "01_residuals.png": "Residuals (Predictions - Actual)",
            "02_predictions_vs_actual.png": "Calibration Plot",
            "03_feature_importance.png": "Top 20 Features (SHAP-based)",
            "04_error_distribution.png": "Error Distribution",
            "05_model_summary.png": "Model Performance Summary",
            "06_shap_heatmap.png": "SHAP Feature Contribution Heatmap",
        }

        # Dropdown to select plot
        selected_plot = st.selectbox(
            "Select a plot to view:",
            options=sorted(plot_files.keys()),
            format_func=lambda x: plot_files[x],
            index=4,
            label_visibility="collapsed",
        )

        # Display selected plot
        plot_path = plots_dir / selected_plot
        if plot_path.exists():
            st.image(str(plot_path), width="stretch")

    # ========== TAB 2: PREDICTION INTERFACE ==========
    with tab2:
        st.markdown("""
        Enter a **SMILES string** to predict its binding affinity (pIC50) to EGFR.
        The model will show the predicted value and binding strength interpretation.
        """)

        # Initialize session state for SMILES input
        if "smiles_input_value" not in st.session_state:
            st.session_state.smiles_input_value = ""

        # SMILES input with example molecules
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**Enter Custom SMILES**")
            smiles_input = st.text_input(
                "Enter SMILES string:",
                value=st.session_state.smiles_input_value,
                placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                help="Simplified Molecular Input Line Entry System",
                label_visibility="collapsed",
            )

            # Update session state with text input changes
            if smiles_input:
                st.session_state.smiles_input_value = smiles_input

        with col2:
            st.markdown("**Quick Examples**")
            example_molecules = {
                "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            }

            selected_example = st.selectbox(
                "Quick examples:",
                options=["None"] + list(example_molecules.keys()),
                label_visibility="collapsed",
            )

            if selected_example != "None" and selected_example:
                st.session_state.smiles_input_value = example_molecules[selected_example]

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
                                                import json as json_module

                                                metadata = json_module.load(f)
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
                                    col1, col2 = st.columns([1, 1])

                                    with col1:
                                        st.markdown("**Molecular Structure**")
                                        img = Draw.MolToImage(mol, size=(300, 300))
                                        st.image(img, width="stretch")

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
                                    st.markdown("**Feature Importance for This Prediction**")

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
                                            import json

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
                                        colors_list = []

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
                                                colors_list.append(plt.cm.Blues(0.6))
                                            elif idx in rdkit_descriptors:
                                                top_features.append(rdkit_descriptors[idx])
                                                colors_list.append(plt.cm.Greens(0.5))
                                            else:
                                                top_features.append(f"RDKit_{idx - 2048}")
                                                colors_list.append(plt.cm.Greens(0.5))

                                        # Create advanced bar chart with extra wide size for labels
                                        fig, ax = plt.subplots(figsize=(15, 6))
                                        bars = ax.barh(
                                            range(len(top_features)),
                                            top_values,
                                            color=colors_list,
                                            edgecolor="black",
                                            linewidth=1,
                                            alpha=0.85,
                                        )

                                        # Styling
                                        ax.set_yticks(range(len(top_features)))
                                        ax.set_yticklabels(top_features, fontsize=9)
                                        ax.set_xlabel(
                                            "Feature Importance (Weighted by Presence in Molecule)",
                                            fontsize=11,
                                            fontweight="bold",
                                        )
                                        ax.set_title(
                                            "Top 10 Most Important Features for This EGFR Binding Prediction",
                                            fontsize=12,
                                            fontweight="bold",
                                            pad=15,
                                        )
                                        ax.invert_yaxis()  # Highest at top
                                        ax.grid(True, alpha=0.3, axis="x", linestyle="--")

                                        # Set x-axis limit to accommodate value labels
                                        max_x = (
                                            max(top_values) * 1.08 if max(top_values) > 0 else 0.1
                                        )
                                        ax.set_xlim(0, max_x)

                                        # Add value labels on bars - consistent positioning for all
                                        for bar, val in zip(bars, top_values, strict=False):
                                            width = bar.get_width()
                                            ax.text(
                                                width + max(top_values) * 0.02,
                                                bar.get_y() + bar.get_height() / 2.0,
                                                f"{val:.4f}",
                                                va="center",
                                                fontsize=8,
                                            )

                                        # Add legend
                                        legend_elements = [
                                            Patch(
                                                facecolor=plt.cm.Blues(0.6),
                                                edgecolor="black",
                                                label="Morgan Bits (circular substructures)",
                                            ),
                                            Patch(
                                                facecolor=plt.cm.Greens(0.5),
                                                edgecolor="black",
                                                label="RDKit Descriptors (molecular properties)",
                                            ),
                                        ]
                                        ax.legend(
                                            handles=legend_elements,
                                            loc="lower right",
                                            fontsize=9,
                                            framealpha=0.95,
                                            edgecolor="black",
                                        )

                                        # Use tight_layout to ensure everything fits
                                        plt.tight_layout()
                                        # Force figure to be rendered fresh
                                        st.pyplot(fig, width="stretch")
                                        plt.close("all")  # Close all figures to prevent caching

                                        st.caption(
                                            "Scores show per-molecule feature importance (model importance weighted by feature presence). Morgan bits include SMILES substructure annotations where available."
                                        )
                                    except Exception as e:
                                        st.warning(
                                            f"Could not display feature importance: {str(e)[:150]}"
                                        )

                                    # Store for potential later use
                                    st.session_state.prediction_result = {
                                        "smiles": smiles_input,
                                        "features": X_combined,
                                        "prediction": y_pred,
                                        "mol": mol,
                                    }

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

    # ========== TAB 3: LEARN MORE ==========
    with tab3:
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
    st.subheader("Screened Molecules (Ranked by Predicted Activity)")

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
        hide_index=True,
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

    # ========== SIDEBAR: INPUTS AND CONTROLS ==========
    st.sidebar.markdown("### Upload SMILES Data")

    # Load sample data button
    if st.sidebar.button("Load Sample Data", help="Load example molecules for quick testing"):
        st.session_state.sample_smiles_path = "app/data/sample/screening_sample.csv"
        st.sidebar.success("Sample data loaded!")

    st.sidebar.markdown("**Or upload your own CSV file:**")

    # File upload in sidebar
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (columns: molecule_id, smiles)",
        type=["csv"],
        key="screening_file",
    )

    st.sidebar.markdown("---")

    # Determine data source
    csv_df = None
    data_source = None

    if "sample_smiles_path" in st.session_state:
        try:
            csv_df = pd.read_csv(st.session_state.sample_smiles_path)
            data_source = "sample"
        except Exception as e:
            st.sidebar.error(f"Failed to load sample data: {e}")

    elif uploaded_file:
        try:
            csv_df = pd.read_csv(uploaded_file)
            data_source = "uploaded"
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV file: {e}")

    run_screening = st.sidebar.button("Run Virtual Screening", type="primary", width="stretch")

    # ========== MAIN AREA: DESCRIPTION AND RESULTS ==========
    st.subheader("Virtual Screening Pipeline (Batch QSAR Predictions)")

    st.markdown("""
    Upload a CSV file with SMILES strings to screen molecules against the EGFR QSAR model.
    
    **Pipeline Steps:**
    1. Validate SMILES strings and compute molecular features
    2. Generate QSAR predictions (pIC50 binding affinity)
    3. Calculate drug-likeness metrics (QED score, Lipinski violations)
    4. Filter molecules by drug-likeness criteria (≤1 Lipinski violation)
    5. Rank results by predicted activity (descending)
    """)

    # Show file format example
    with st.expander("View CSV Format Example"):
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


def render_app() -> None:
    """Main application renderer with sidebar navigation.

    Displays the application header, sidebar navigation buttons, and routes to appropriate
    page based on user selection (Single Molecule or Similarity Search).

    Returns:
        None. Renders Streamlit components directly.
    """
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                margin-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main header (constant across all pages)
    st.header("MoleculeInsight", divider="gray", text_alignment="center")

    # Sidebar navigation with improved styling
    st.sidebar.markdown("### Analysis Mode")

    col1, col2 = st.sidebar.columns(2)

    single_mol = col1.button("Single\nMolecule", width="stretch", key="nav_single")
    similarity = col2.button("Similarity\nSearch", width="stretch", key="nav_similarity")

    col3, col4 = st.sidebar.columns(2)

    qsar = col3.button("QSAR\nModel", width="stretch", key="nav_qsar")
    screening = col4.button("Virtual\nScreening", width="stretch", key="nav_screening")

    st.sidebar.divider()

    # Initialize page state if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = None

    # Update page state based on button clicks
    if single_mol:
        st.session_state.current_page = "Single Molecule"
    if similarity:
        st.session_state.current_page = "Similarity Search"
    if qsar:
        st.session_state.current_page = "QSAR Model"
    if screening:
        st.session_state.current_page = "Virtual Screening"

    # Render selected page or welcome message
    if st.session_state.current_page == "Single Molecule":
        render_single_molecule()
    elif st.session_state.current_page == "Similarity Search":
        render_similarity_search()
    elif st.session_state.current_page == "QSAR Model":
        render_qsar_dashboard()
    elif st.session_state.current_page == "Virtual Screening":
        render_virtual_screening()
    else:
        st.markdown(
            """
        <div style="text-align: center; padding: 2rem; color: #666;">
            <h3>Welcome to MoleculeInsight</h3>
            <p>Select an analysis mode from the sidebar to get started:</p>
            <ul style="list-style-type: none;">
                <li><strong>Single Molecule:</strong> Analyze properties and bioactivity of a single compound</li>
                <li><strong>Similarity Search:</strong> Find structurally similar molecules from a reference library</li>
                <li><strong>QSAR Model:</strong> Predict EGFR binding affinity (pIC50) from XGBoost QSAR model trained on ChEMBL bioactivity data</li>
                <li><strong>Virtual Screening:</strong> Batch screen molecules through QSAR pipeline with drug-likeness filtering</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

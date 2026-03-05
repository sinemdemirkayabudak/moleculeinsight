import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from rdkit.Chem import Draw

from app.chembl import get_compound_bioactivity_from_mol
from app.molecule import get_molecule, get_rdkit_properties, lipinski_rules
from app.pubchem import get_pubchem_metadata
from app.similarity_search import create_structure_image, prepare_csv_export, run_similarity_search
from app.utils import safe_execute
from app.validators import validate_smiles


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

    st.sidebar.divider()

    # Initialize page state if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = None

    # Update page state based on button clicks
    if single_mol:
        st.session_state.current_page = "Single Molecule"
    if similarity:
        st.session_state.current_page = "Similarity Search"

    # Render selected page or welcome message
    if st.session_state.current_page == "Single Molecule":
        render_single_molecule()
    elif st.session_state.current_page == "Similarity Search":
        render_similarity_search()
    else:
        st.markdown(
            """
        <div style="text-align: center; padding: 2rem; color: #666;">
            <h3>Welcome to MoleculeInsight</h3>
            <p>Select an analysis mode from the sidebar to get started:</p>
            <ul style="list-style-type: none;">
                <li><strong>Single Molecule:</strong> Analyze properties and bioactivity of a single compound</li>
                <li><strong>Similarity Search:</strong> Find structurally similar molecules from a reference library</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

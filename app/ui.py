import pandas as pd
import streamlit as st
from rdkit.Chem import Draw

from app.chembl import get_chembl_molecule, get_compound_bioactivity_from_mol
from app.components.cards import render_metric_card
from app.molecule import get_molecule, get_rdkit_properties, lipinski_rules
from app.pubchem import get_pubchem_metadata
from app.utils import safe_execute
from app.validators import validate_smiles


def render_app():
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
    st.header("MoleculeInsight", divider="gray", text_alignment="center")

    smiles = st.sidebar.text_input(
        "Enter SMILES:",
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
            st.error("Property calculation failed")
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
                with st.container(height=450):
                    st.subheader("Molecular Structure")
                    img = Draw.MolToImage(mol, size=(400, 300))
                    st.image(img)

            with col2:
                with st.container(height=450):
                    meta = get_pubchem_metadata(mol)
                    st.subheader("Properties")
                    st.write(f"- IUPAC Name: {meta['iupac']}")
                    st.write(f"- Common Name: {meta['common']}")
                    st.write(f"- CID: {meta['cid']}")
                    st.write(f"- InChIKey: {meta['inchikey']}")
                    st.write(f"- Molecular Weight (MW): {mw:.2f}")
                    st.write(f"- LogP (octanol-water): {logp:.2f}")
                    st.write(f"- Topological Polar Surface Area (TPSA): {tpsa:.2f}")
                    st.write(f"- H-bond Donors (HBD): {hbd}")
                    st.write(f"- H-bond Acceptors (HBA): {hba}")
                    st.write(f"- Rotatable Bonds: {rotb}")

        with tab2:
            with st.container(height=300):
                st.subheader("Lipinski Rule-of-5 Compliance")
                for rule, passed in rules.items():
                    st.text(f"- {rule}   \t{'✔ Passed' if passed else '✘ Violated'}")
                st.write("\n")
                st.write(f"Total violations: {violations}")

        with tab3:
            with st.container(height=700):
                st.subheader("Bioactivity Evidence (ChEMBL)")
                bioactivity_data = get_compound_bioactivity_from_mol(mol)
                if bioactivity_data.get("success"):
                    df = pd.DataFrame(bioactivity_data["bioactivity"]["activities"])

                    st.dataframe(df)

                    st.subheader("📊 Compound Summary")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        render_metric_card("PubChem CID", meta["cid"], "🧬")

                    with col2:
                        chembl_id = get_chembl_molecule(meta["inchikey"]).get("chembl_id", "N/A")

                        render_metric_card("ChEMBL ID", chembl_id, "🧪")

                    with col3:
                        count = bioactivity_data["bioactivity"]["count"]

                        render_metric_card("Bioactivity Records", count, "📑")

                else:
                    st.error("Bioactivity data retrieval failed")

import streamlit as st

from app.ui import render_app

if __name__ == "__main__":
    st.set_page_config(
        page_title="MoleculeInsight",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": """
            # MoleculeInsight
            Predict EGFR binding affinity, screen compound libraries, and analyze structure-activity relationships using validated XGBoost QSAR model. 
            ## Features 
            - QSAR modeling with XGBoost (R²=0.70)
            - Molecular similarity search
            - Virtual screening pipeline
            - Scaffold & SAR analysis
            - SHAP explainability
            
            ## Links
            - Source Code: [GitHub](https://github.com/sinemdemirkayabudak/moleculeinsight)
            - Documentation: [Readme](https://github.com/sinemdemirkayabudak/moleculeinsight#readme)
            
            ---
            
            © 2026 Sinem Demirkaya-Budak
            """,
            "Report a bug": "https://sinembudak.com/contact",
        },
    )
    render_app()

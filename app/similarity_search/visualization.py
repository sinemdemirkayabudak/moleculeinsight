# ============================================================
# VISUALIZATION AND EXPORT
# ============================================================

import logging
import warnings
import base64
import matplotlib.pyplot as plt
from functools import lru_cache

from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from app.molecule import get_molecule

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Prevent propagation to parent loggers (stops duplicate logs)

# Remove existing handlers to avoid duplicates
# prevents duplicate handlers when code reruns
logger.handlers.clear()

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def visualize_distribution(top_hits, query_df, show=True):
    """Pre-generate and cache individual ranking plots for each query.
    
    Parameters:
        top_hits (pd.DataFrame): DataFrame with top N hits per query, sorted by similarity
        query_df (pd.DataFrame): DataFrame with query molecules
        show (bool): Whether to display plots (default: True, typically False in Streamlit)
    
    Returns:
        dict: Dictionary mapping query_name -> matplotlib figure object for each query.
              Returns empty dict if no plots can be generated.
    """
    try:
        logger.info("Generating ranking plots for all queries...")
        
        query_plots = {}
        
        # Generate a single plot for each query (sorted ascending)
        for query_name in sorted(query_df["query_name"]):
            # Get top hits for this query, sorted by similarity ascending
            query_hits = top_hits[top_hits["query_name"] == query_name].sort_values(
                "similarity", ascending=True
            )
            
            if len(query_hits) == 0:
                logger.warning(f"No hits found for query: {query_name}")
                continue
            
            # Create figure for this query
            fig, ax = plt.subplots(figsize=(14, max(6, len(query_hits) * 0.5)))
            
            # Create horizontal bar chart
            ax.barh(
                range(len(query_hits)),
                query_hits["similarity"],
                color=plt.cm.RdYlGn(query_hits["similarity"]),  # Red (high) to Green (low)
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add similarity score labels on bars
            for i, similarity in enumerate(query_hits["similarity"]):
                ax.text(
                    similarity + 0.02,
                    i,
                    f"{similarity:.3f}",
                    va='center',
                    fontsize=12
                )
            
            # Set y-axis labels to compound names
            ax.set_yticks(range(len(query_hits)))
            ax.set_yticklabels(query_hits["ref_name"], fontsize=12)
            
            ax.set_xlabel("Tanimoto Similarity", fontsize=13)
            ax.set_title(f"Top Similar Compounds for Query: {query_name}", fontweight='bold', fontsize=14)
            ax.set_xlim(0, 1.15)  # Leave room for score labels
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(axis='x', alpha=0.3)
            
            fig.tight_layout()
            
            # Cache the figure
            query_plots[query_name] = fig
            logger.debug(f"Generated plot for query: {query_name}")
        
        logger.info(f"Successfully generated {len(query_plots)} plots")
        return query_plots
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        logger.warning("Continuing without plot visualization")
        return {}


def create_structure_image(query_name, query_smiles, ref_name, ref_smiles):
    """Create side-by-side molecule images with labels (cached for performance).
    
    This function is optimized for large result sets with caching:
    - Caches generated images based on SMILES pairs to avoid redundant computation
    - Ideal for similarity search UI with 100+ results
    - Cache is shared across Streamlit reruns in same session
    
    Parameters:
        query_name (str): Name/identifier of the query molecule
        query_smiles (str): SMILES string of the query molecule
        ref_name (str): Name/identifier of the reference molecule
        ref_smiles (str): SMILES string of the reference molecule
    
    Returns:
        str: Base64 encoded PNG image string for HTML embedding, or None if error occurs
    """
    return _generate_cached_structure_image(query_name, query_smiles, ref_name, ref_smiles)


@lru_cache(maxsize=1024)
def _generate_cached_structure_image(query_name, query_smiles, ref_name, ref_smiles):
    """Internal cached implementation of structure image generation.
    
    Uses functools.lru_cache for efficient memoization:
    - Caches up to 1024 unique molecule pair combinations
    - Cache key automatically generated from input strings
    - Significantly speeds up large result sets with repeated molecules
    
    For example: Searching 100 molecules against 500 reference compounds will generate
    at most 100 unique query images + 500 unique reference images = ~600 total images
    instead of 50,000 redundant computations
    """
    try:
        # Get molecules
        query_mol = get_molecule(query_smiles)
        ref_mol = get_molecule(ref_smiles)
        
        if not query_mol or not ref_mol:
            return None
        
        # Draw molecules
        query_img = Draw.MolToImage(query_mol, size=(120, 120))
        ref_img = Draw.MolToImage(ref_mol, size=(120, 120))
        
        # Create composite image with labels
        # Label height + molecule height + padding
        composite_height = 30 + 120 + 10
        composite_width = 120 + 20 + 120  # query + gap + reference
        
        composite = Image.new('RGB', (composite_width, composite_height), color='white')
        
        # Draw labels using PIL
        draw = ImageDraw.Draw(composite)
        
        # Use a default font
        font = ImageFont.load_default()
        
        # Draw query label
        draw.text((10, 5), query_name, fill='black', font=font)
        # Paste query image
        composite.paste(query_img, (10, 30))
        
        # Draw reference label
        draw.text((150, 5), ref_name, fill='black', font=font)
        # Paste reference image
        composite.paste(ref_img, (140, 30))
        
        # Convert to base64
        buffer = BytesIO()
        composite.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
    except Exception as e:
        logger.warning(f"Error creating structure image: {e}")
        return None


def prepare_csv_export(results_df):
    """Prepare results dataframe for CSV export with proper column order.
    
    Parameters:
        results_df (pd.DataFrame): Results dataframe with structure images column
    
    Returns:
        str: CSV formatted string ready for download
    """
    export_df = results_df.drop(columns=['Structures'])[
        ['Query Molecule', 'Query SMILES', 'Reference Molecule', 
         'Reference SMILES', 'Similarity Score']
    ]
    return export_df.to_csv(index=False)

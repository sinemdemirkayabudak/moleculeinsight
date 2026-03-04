# ============================================================
# VALIDATION UTILITIES
# ============================================================

import logging
import warnings

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


def validate_parameters(radius, top_n):
    """
    Validate fingerprint and ranking parameters.
    
    Raises:
        ValueError: If any parameter is invalid.
    """
    if radius < 0:
        raise ValueError(f"Radius must be non-negative, got {radius}")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    logger.debug(f"Parameters validated: radius={radius}, top_n={top_n}")


def validate_dataframe(df, name, required_columns):
    """
    Validate that dataframe is non-empty and contains required columns.
    
    Parameters:
        df: pandas DataFrame
        name: Name of the dataframe (for logging)
        required_columns: List of column names that must exist
        
    Raises:
        ValueError: If dataframe is empty or missing columns.
    """
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name} missing columns: {missing_cols}")
    
    logger.debug(f"{name} validation passed")

import logging
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
CHEMBL_BASE_URL = os.getenv("CHEMBL_BASE_URL", "https://www.ebi.ac.uk/chembl/api/data")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("moleculeinsight.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

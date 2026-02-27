import logging
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
PUBCHEM_CID_URL = os.getenv(
    "PUBCHEM_CID_URL", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/cids/JSON"
)
PUBCHEM_PROP_URL = os.getenv(
    "PUBCHEM_PROP_URL",
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/IUPACName/JSON",
)
PUBCHEM_SYN_URL = os.getenv(
    "PUBCHEM_SYN_URL", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/synonyms/JSON"
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("moleculeinsight.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

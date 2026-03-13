# EGFR Feature Annotations

## Overview
`egfr_feature_annotations.json` is a comprehensive file containing pre-computed SMILES substructure annotations for the XGBoost EGFR binding affinity prediction model.

## File Structure

The JSON file contains three main sections:

### 1. Metadata
- **version**: File format version (currently 1.0)
- **created**: Timestamp when annotations were generated
- **source**: Data source used for annotations (ChEMBL API with 5000 EGFR molecules)
- **total_morgan_bits**: Number of Morgan bits (2048)
- **annotated_bits**: Number of bits with substructure annotations (2009)
- **coverage**: Percentage of bits that have annotations (98.1%)

### 2. Morgan Bits (morgan_bits)
Dictionary mapping Morgan bit indices to SMILES substructure strings:
```json
{
  "1": "C",
  "2": "C(CC)CC",
  "3": "c(NC)(cc)cc",
  ...
  "2047": "N/A"  // Not found in training data
}
```
- Keys: Morgan bit index (0-2047)
- Values: SMILES substructure or "N/A" if not found

### 3. RDKit Descriptors (rdkit_descriptors)
Fixed mapping of RDKit descriptor indices to human-readable names:
```json
{
  "2048": "MW (Molecular Weight)",
  "2049": "LogP (Lipophilicity)",
  "2050": "TPSA (Polar Surface Area)",
  "2051": "HBD (Hydrogen Bond Donors)",
  "2052": "HBA (Hydrogen Bond Acceptors)",
  "2053": "RotBonds (Rotatable Bonds)",
  "2054": "AromaticRings",
  "2055": "RingCount"
}
```

## Usage in Code

### Loading the file
```python
import json

with open('app/qsar/saved_models/egfr_feature_annotations.json') as f:
    anno_data = json.load(f)

# Access components
metadata = anno_data['metadata']
morgan_bits = {int(k): v for k, v in anno_data['morgan_bits'].items()}
rdkit_descriptors = anno_data['rdkit_descriptors']
```

### In the UI (app/ui.py)
The Make Predictions tab automatically loads this file and:
1. For each Morgan bit in top features: displays `Morgan_Bit[index] → [substructure]` if annotated
2. For RDKit descriptors: displays the human-readable name
3. Falls back to generic names (e.g., "Morgan_Bit0079") for unannotated bits

## Data Source
- **Training molecules**: ~5000 EGFR inhibitors from ChEMBL API
- **Generation method**: For each Morgan bit, searches training data for a molecule containing that bit, then extracts the SMILES substructure
- **Coverage**: 2009/2048 bits (98.1%) - only 39 bits rare enough to not appear in any training molecule

## Reuse
This file is meant to be:
- **Reusable**: Contains all necessary feature annotations for model predictions
- **Immutable**: Generated once during training, never needs regeneration
- **Portable**: Can be moved to other directories/projects using the same model
- **Backward compatible**: Can fall back to showing generic bit names if not available

## Example Output
For an EGFR inhibitor prediction, users see:
```
1. Morgan_Bit1452 → n(c)c (importance: 22.2%)
2. Morgan_Bit0491 → N(c)c (importance: 11.9%)
3. Morgan_Bit1465 → c(c(N)c)c(c)c (importance: 4.7%)
```

Instead of just "Morgan_Bit1452", "Morgan_Bit0491", etc.

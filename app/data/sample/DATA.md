# Sample Data

Sample CSV files for testing and demo purposes.

## Files

| File | Columns | Use |
|------|---------|-----|
| `query_molecules.csv` | smiles, name | Query molecules for similarity search (5 molecules) |
| `reference_library.csv` | smiles, name | Reference library for comparison (9 molecules) |
| `moleculeinsight_test_dataset.csv` | molecule_id, smiles, activity_label, source, notes | Test data with edge cases (26 entries) |
| `lipinski_test_dataset.csv` | molecule_id, smiles, activity_label, source, notes | Lipinski rule testing (16 entries) |

## CSV Format

**Similarity Search Files:**
```csv
smiles,name
CCO,Ethanol
CC(=O)O,AceticAcid
```

**Test Datasets:**
```csv
molecule_id,smiles,activity_label,source,notes
MOL_001,CC(=O)OC1=CC=CC=C1C(=O)O,1.0,known_drug,Aspirin
```

## Usage

In app UI: Click "Load Sample Data" in Similarity Search tab

Command line:
```bash
uv run python -m app.similarity_search \
  --query_file app/data/sample/query_molecules.csv \
  --reference_file app/data/sample/reference_library.csv
```

## Notes

- All files are intentionally small for quick testing
- Test datasets include edge cases and invalid entries
- SMILES format: [SMILES Guide](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)

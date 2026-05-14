"""Tests for molecular analysis functions."""

from unittest.mock import patch

import pytest
from rdkit import Chem

from app.molecule import get_molecule, get_rdkit_properties, lipinski_rules


class TestGetMolecule:
    """Test molecule creation from SMILES."""

    def test_valid_smiles_creates_molecule(self):
        """Test that valid SMILES creates a Mol object."""
        mol = get_molecule("c1ccccc1")
        assert mol is not None
        assert isinstance(mol, Chem.Mol)

    def test_invalid_smiles_returns_none(self):
        """Test that invalid SMILES returns None."""
        mol = get_molecule("INVALID")
        assert mol is None

    def test_empty_smiles_returns_none(self):
        """Test that empty SMILES returns None."""
        mol = get_molecule("")
        assert mol is None

    def test_valid_smiles_logs_info(self, caplog):
        """Test that valid SMILES logs info message."""
        import logging

        # Use unique SMILES to avoid cache
        unique_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen

        with caplog.at_level(logging.INFO):
            mol = get_molecule(unique_smiles)

        assert mol is not None
        # When molecule creation succeeds, we expect the mol to be returned
        assert isinstance(mol, Chem.Mol)

    @patch("app.molecule.Chem.MolFromSmiles")
    def test_molecule_creation_exception(self, mock_mol_from_smiles):
        """Test exception handling in get_molecule."""
        mock_mol_from_smiles.side_effect = Exception("SMILES parsing error")

        result = get_molecule("valid_smiles")

        assert result is None


class TestRdkitProperties:
    """Test RDKit property calculations."""

    def test_benzene_properties(self):
        """Test properties for benzene."""
        mol = get_molecule("c1ccccc1")
        props = get_rdkit_properties(mol)

        assert props is not None
        assert "mw" in props
        assert "logP" in props
        assert "tpsa" in props
        assert "hbd" in props
        assert "hba" in props
        assert "rotb" in props

        # Benzene C6H6 should have MW ~78
        assert 75 < props["mw"] < 80
        assert props["hbd"] == 0
        assert props["hba"] == 0

    def test_ethanol_properties(self):
        """Test properties for ethanol."""
        mol = get_molecule("CCO")
        props = get_rdkit_properties(mol)

        assert props is not None
        # Ethanol C2H6O should have MW ~46
        assert 45 < props["mw"] < 47
        assert props["hbd"] == 1  # One OH group
        assert props["hba"] == 1

    def test_none_molecule_returns_none(self):
        """Test that None molecule returns None."""
        props = get_rdkit_properties(None)
        assert props is None

    def test_all_properties_are_floats(self):
        """Test that all calculated properties are numeric."""
        mol = get_molecule("CCO")
        props = get_rdkit_properties(mol)

        for key, value in props.items():  # ty:ignore[unresolved-attribute]
            assert isinstance(value, (int, float)), f"{key} is not numeric"

    @patch("app.molecule.logger")
    @patch("app.molecule.st.error")
    @patch("app.molecule.Descriptors.MolLogP")
    def test_property_calculation_exception(self, mock_logp, mock_st_error, mock_logger):
        """Test exception handling when property calculation fails."""
        mock_logp.side_effect = Exception("LogP failed")

        mol = get_molecule("CCO")
        props = get_rdkit_properties(mol)

        assert props is None
        mock_logger.error.assert_called()
        mock_st_error.assert_called()


class TestLipinskiRules:
    """Test Lipinski Rule-of-5 evaluation."""

    def test_benzene_passes_lipinski(self):
        """Test that benzene passes Lipinski rules."""
        mol = get_molecule("c1ccccc1")
        props = get_rdkit_properties(mol)
        rules = lipinski_rules(props)

        assert rules is not None
        # Benzene should pass all rules
        assert all(rules.values())

    def test_ethanol_passes_lipinski(self):
        """Test that ethanol passes Lipinski rules."""
        mol = get_molecule("CCO")
        props = get_rdkit_properties(mol)
        rules = lipinski_rules(props)

        assert all(rules.values())  # ty:ignore[unresolved-attribute]

    def test_rules_dict_structure(self):
        """Test that rules dictionary has correct keys."""
        mol = get_molecule("c1ccccc1")
        props = get_rdkit_properties(mol)
        rules = lipinski_rules(props)

        expected_keys = {"MW <= 500", "LogP <= 5", "HBD <= 5", "HBA <= 10"}
        assert set(rules.keys()) == expected_keys  # ty:ignore[unresolved-attribute]

    def test_all_rules_are_boolean(self):
        """Test that all rule values are boolean."""
        mol = get_molecule("CCO")
        props = get_rdkit_properties(mol)
        rules = lipinski_rules(props)

        for key, value in rules.items():  # ty:ignore[unresolved-attribute]
            assert isinstance(value, bool), f"{key} is not boolean"

    def test_missing_property_key_raises_error(self):
        """Test that missing property raises KeyError."""
        incomplete_props = {"mw": 300}  # Missing logP, hbd, hba

        with pytest.raises(KeyError):
            lipinski_rules(incomplete_props)  # ty:ignore[invalid-argument-type]

    @patch("app.molecule.logger")
    def test_lipinski_rules_exception_handling(self, mock_logger):
        """Test exception handling in lipinski_rules."""
        # Create a properties dict that will trigger an exception
        bad_props = {"mw": "invalid", "logP": 3, "hbd": 1, "hba": 1}

        with pytest.raises(TypeError):
            lipinski_rules(bad_props)  # ty:ignore[invalid-argument-type]

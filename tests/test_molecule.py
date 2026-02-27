"""Tests for molecular analysis functions."""

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

        for key, value in props.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric"


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

        assert all(rules.values())

    def test_rules_dict_structure(self):
        """Test that rules dictionary has correct keys."""
        mol = get_molecule("c1ccccc1")
        props = get_rdkit_properties(mol)
        rules = lipinski_rules(props)

        expected_keys = {"MW <= 500", "LogP <= 5", "HBD <= 5", "HBA <= 10"}
        assert set(rules.keys()) == expected_keys

    def test_all_rules_are_boolean(self):
        """Test that all rule values are boolean."""
        mol = get_molecule("CCO")
        props = get_rdkit_properties(mol)
        rules = lipinski_rules(props)

        for key, value in rules.items():
            assert isinstance(value, bool), f"{key} is not boolean"

    def test_missing_property_key_raises_error(self):
        """Test that missing property raises KeyError."""
        incomplete_props = {"mw": 300}  # Missing logP, hbd, hba

        with pytest.raises(KeyError):
            lipinski_rules(incomplete_props)

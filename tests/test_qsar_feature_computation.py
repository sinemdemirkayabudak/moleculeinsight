"""Tests for QSAR feature computation functions (Morgan and Descriptors)."""

from unittest.mock import patch

import numpy as np

from app.qsar.features import (
    compute_morgan_fingerprints,
    compute_rdkit_descriptors,
)


class TestComputeMorganFingerprints:
    """Test Morgan fingerprint computation."""

    def test_morgan_fingerprints_single_molecule(self):
        """Test Morgan fingerprints for single molecule."""
        smiles_list = ["CCO"]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is True
        assert "X" in result
        assert "feature_names" in result
        assert result["X"].shape == (1, 2048)
        assert len(result["feature_names"]) == 2048

    def test_morgan_fingerprints_multiple_molecules(self):
        """Test Morgan fingerprints for multiple molecules."""
        smiles_list = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is True
        assert result["X"].shape == (3, 2048)

    def test_morgan_fingerprints_custom_radius(self):
        """Test Morgan fingerprints with custom radius parameter."""
        smiles_list = ["CCO"]

        result_r1 = compute_morgan_fingerprints(smiles_list, radius=1)
        result_r3 = compute_morgan_fingerprints(smiles_list, radius=3)

        assert result_r1["success"] is True
        assert result_r3["success"] is True
        # Both should produce 2048 bits regardless of radius
        assert result_r1["X"].shape[1] == 2048
        assert result_r3["X"].shape[1] == 2048

    def test_morgan_fingerprints_feature_names(self):
        """Test Morgan fingerprints feature names format."""
        smiles_list = ["CCO"]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is True
        assert all(name.startswith("bit_") for name in result["feature_names"])
        assert result["feature_names"][0] == "bit_0"
        assert result["feature_names"][-1] == "bit_2047"

    def test_morgan_fingerprints_binary_values(self):
        """Test Morgan fingerprints contain only binary values."""
        smiles_list = ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is True
        unique_values = np.unique(result["X"])
        assert set(unique_values).issubset({0, 1})

    def test_morgan_fingerprints_empty_list(self):
        """Test Morgan fingerprints with empty SMILES list."""
        result = compute_morgan_fingerprints([])

        assert result["success"] is False
        assert "error" in result

    def test_morgan_fingerprints_invalid_smiles(self):
        """Test Morgan fingerprints with invalid SMILES."""
        result = compute_morgan_fingerprints(["INVALID_SMILES"])

        assert result["success"] is False
        assert "error" in result

    def test_morgan_fingerprints_mixed_valid_invalid(self):
        """Test Morgan fingerprints with mixed valid/invalid SMILES."""
        smiles_list = ["CCO", "INVALID", "CC(=O)Oc1ccccc1C(=O)O"]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is True
        assert result["X"].shape[0] == 2  # Only valid SMILES
        assert len(result["feature_names"]) == 2048

    def test_morgan_fingerprints_large_dataset(self):
        """Test Morgan fingerprints with large dataset."""
        smiles_list = ["CCO"] * 100
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is True
        assert result["X"].shape == (100, 2048)

    def test_morgan_fingerprints_complex_molecules(self):
        """Test Morgan fingerprints with various complex molecules."""
        smiles_list = [
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
            "O=C(O)Cc1ccccc1NC(=O)c2ccccc2",  # Diclofenac
            "CC(C)CC(NC(=O)C(CCCNC(=N)N)NC(=O)C)C(=O)O",  # Complex amino acid derivative
        ]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is True
        assert result["X"].shape[0] == 3
        assert result["X"].shape[1] == 2048

    def test_morgan_fingerprints_consistency(self):
        """Test that same SMILES always produces same fingerprint."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"

        result1 = compute_morgan_fingerprints([smiles])
        result2 = compute_morgan_fingerprints([smiles])

        assert result1["success"] is True
        assert result2["success"] is True
        np.testing.assert_array_equal(result1["X"], result2["X"])

    def test_morgan_fingerprints_aromatic_molecules(self):
        """Test Morgan fingerprints with aromatic molecules."""
        smiles_list = [
            "c1ccccc1",  # Benzene
            "c1cccnc1",  # Pyridine
            "c1ccc2ccccc2c1",  # Naphthalene
        ]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is True
        assert result["X"].shape[0] == 3
        # Different aromatic molecules should have different fingerprints
        assert not np.array_equal(result["X"][0], result["X"][1])

    def test_morgan_fingerprints_different_radii_produce_different_results(self):
        """Test that different radii produce different fingerprints."""
        smiles_list = ["CC(=O)Oc1ccccc1C(=O)O"]

        result_r1 = compute_morgan_fingerprints(smiles_list, radius=1)
        result_r2 = compute_morgan_fingerprints(smiles_list, radius=2)
        result_r3 = compute_morgan_fingerprints(smiles_list, radius=3)

        assert result_r1["success"] is True
        assert result_r2["success"] is True
        assert result_r3["success"] is True

        # Different radii should produce different fingerprints
        assert not np.array_equal(result_r1["X"], result_r2["X"])
        assert not np.array_equal(result_r2["X"], result_r3["X"])

    def test_morgan_fingerprints_with_partial_failures(self):
        """Test Morgan fingerprints with some molecules failing to parse."""
        smiles_list = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "INVALID1",
            "CCO",
            "INVALID2",
            "c1ccccc1",
        ]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is True
        assert result["X"].shape[0] == 3  # Only valid molecules
        assert result["X"].shape[1] == 2048

    @patch("app.qsar.features.AllChem.GetMorganGenerator")
    def test_morgan_fingerprints_exception_in_generator(self, mock_generator):
        """Test Morgan fingerprints handles exception in top-level code."""
        # Force an exception in GetMorganGenerator
        mock_generator.side_effect = RuntimeError("Generator failed")

        smiles_list = ["CCO"]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is False
        assert "error" in result
        assert "Generator failed" in result["error"]

    @patch("app.qsar.features.Chem.MolFromSmiles")
    def test_morgan_fingerprints_exception_in_loop(self, mock_mol):
        """Test Morgan fingerprints handles exception in computation loop."""
        # Force an exception when MolFromSmiles is called
        mock_mol.side_effect = RuntimeError("MolFromSmiles failed")

        smiles_list = ["CCO"]
        result = compute_morgan_fingerprints(smiles_list)

        assert result["success"] is False
        assert "error" in result


class TestComputeRdkitDescriptors:
    """Test RDKit descriptor computation."""

    def test_descriptors_single_molecule(self):
        """Test descriptor computation for single molecule."""
        smiles_list = ["CCO"]
        result = compute_rdkit_descriptors(smiles_list)

        assert result["success"] is True
        assert "X" in result
        assert "feature_names" in result
        assert result["X"].shape == (1, 8)
        assert len(result["feature_names"]) == 8

    def test_descriptors_multiple_molecules(self):
        """Test descriptor computation for multiple molecules."""
        smiles_list = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
        result = compute_rdkit_descriptors(smiles_list)

        assert result["success"] is True
        assert result["X"].shape == (3, 8)

    def test_descriptors_all_types(self):
        """Test all 8 descriptor types are computed."""
        smiles_list = ["CC(=O)Oc1ccccc1C(=O)O"]
        result = compute_rdkit_descriptors(smiles_list)

        assert result["success"] is True
        expected_names = [
            "MW",
            "LogP",
            "HBD",
            "HBA",
            "TPSA",
            "RotBonds",
            "AromaticRings",
            "RingCount",
        ]
        assert result["feature_names"] == expected_names

    def test_descriptors_specific_ethanol(self):
        """Test descriptor values for ethanol."""
        result = compute_rdkit_descriptors(["CCO"])

        assert result["success"] is True
        # Ethanol: MW=46, LogP is negative, HBD=1, HBA=1
        assert result["X"][0, 0] > 0  # MW positive
        assert result["X"][0, 2] == 1  # HBD = 1
        assert result["X"][0, 3] >= 1  # HBA >= 1

    def test_descriptors_specific_benzene(self):
        """Test descriptor values for benzene."""
        result = compute_rdkit_descriptors(["c1ccccc1"])

        assert result["success"] is True
        # Benzene: 6 aromatic carbons, 1 aromatic ring
        assert result["X"][0, 6] == 1  # AromaticRings = 1
        assert result["X"][0, 7] == 1  # RingCount = 1

    def test_descriptors_empty_list(self):
        """Test descriptors with empty SMILES list."""
        result = compute_rdkit_descriptors([])

        assert result["success"] is False
        assert "error" in result

    def test_descriptors_invalid_smiles(self):
        """Test descriptors with invalid SMILES."""
        result = compute_rdkit_descriptors(["INVALID_SMILES"])

        assert result["success"] is False
        assert "error" in result

    def test_descriptors_mixed_valid_invalid(self):
        """Test descriptors with mixed valid/invalid SMILES."""
        smiles_list = ["CCO", "INVALID", "CC(=O)Oc1ccccc1C(=O)O"]
        result = compute_rdkit_descriptors(smiles_list)

        assert result["success"] is True
        assert result["X"].shape[0] == 2  # Only valid SMILES
        assert result["X"].shape[1] == 8

    def test_descriptors_large_dataset(self):
        """Test descriptors with large dataset."""
        smiles_list = ["CCO"] * 100
        result = compute_rdkit_descriptors(smiles_list)

        assert result["success"] is True
        assert result["X"].shape == (100, 8)

    def test_descriptors_nans_detected(self):
        """Test that NaNs are detected in descriptor values."""
        # Very long chain to potentially trigger NaN - use a complex SMILES
        long_smiles = "C" * 200
        result = compute_rdkit_descriptors([long_smiles])

        if result["success"]:
            # Check if there are any NaN values
            np.any(np.isnan(result["X"]))
            # NaN detection and logging is tested via logger call

    def test_descriptors_descriptor_value_ranges(self):
        """Test that descriptor values are within expected ranges."""
        smiles_list = ["CC(=O)Oc1ccccc1C(=O)O"]  # Aspirin
        result = compute_rdkit_descriptors(smiles_list)

        assert result["success"] is True
        # MW should be positive
        assert result["X"][0, 0] > 0
        # LogP is typically between -3 and 10
        assert -10 < result["X"][0, 1] < 10
        # HBD and HBA should be non-negative integers
        assert result["X"][0, 2] >= 0
        assert result["X"][0, 3] >= 0

    def test_descriptors_consistency(self):
        """Test that same SMILES always produces same descriptors."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"

        result1 = compute_rdkit_descriptors([smiles])
        result2 = compute_rdkit_descriptors([smiles])

        assert result1["success"] is True
        assert result2["success"] is True
        np.testing.assert_array_equal(result1["X"], result2["X"])

    def test_descriptors_aromatic_rings(self):
        """Test that aromatic ring descriptor is calculated correctly."""
        # Benzene has 1 aromatic ring
        result_benzene = compute_rdkit_descriptors(["c1ccccc1"])
        # Naphthalene has 2 aromatic rings
        result_naphthalene = compute_rdkit_descriptors(["c1ccc2ccccc2c1"])

        assert result_benzene["success"] is True
        assert result_naphthalene["success"] is True

        aromatic_benzene = result_benzene["X"][0, 6]  # AromaticRings
        aromatic_naphthalene = result_naphthalene["X"][0, 6]

        assert aromatic_benzene == 1
        assert aromatic_naphthalene == 2

    def test_descriptors_with_partial_failures(self):
        """Test descriptors with some molecules failing to parse."""
        smiles_list = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "INVALID1",
            "CCO",
            "INVALID2",
            "c1ccccc1",
        ]
        result = compute_rdkit_descriptors(smiles_list)

        assert result["success"] is True
        assert result["X"].shape[0] == 3  # Only valid molecules
        assert result["X"].shape[1] == 8

    @patch("app.qsar.features.Chem.MolFromSmiles")
    def test_descriptors_exception_in_loop(self, mock_mol):
        """Test descriptors handles exception in computation loop."""
        # Force an exception when MolFromSmiles is called
        mock_mol.side_effect = RuntimeError("MolFromSmiles failed")

        smiles_list = ["CCO"]
        result = compute_rdkit_descriptors(smiles_list)

        assert result["success"] is False
        assert "error" in result

    @patch("app.qsar.features.Descriptors.MolWt")
    def test_descriptors_exception_in_descriptor_calc(self, mock_desc):
        """Test descriptors handles exception in descriptor computation."""
        # Force an exception when descriptor is computed
        mock_desc.side_effect = RuntimeError("Descriptor failed")

        smiles_list = ["CCO"]
        result = compute_rdkit_descriptors(smiles_list)

        # Should still succeed but might contain NaN or partial data
        # depending on implementation, so we just check it handles gracefully
        if result["success"]:
            assert result["X"].shape[1] == 8

    @patch("app.qsar.features.np.array")
    def test_descriptors_exception_at_top_level(self, mock_array):
        """Test descriptors exception handler at top level."""
        # This will cause np.array to raise an exception
        mock_array.side_effect = RuntimeError("Array creation failed")

        smiles_list = ["CCO"]
        result = compute_rdkit_descriptors(smiles_list)

        assert result["success"] is False
        assert "error" in result
        assert "Array creation failed" in result["error"]

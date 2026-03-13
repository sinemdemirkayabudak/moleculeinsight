"""Tests for virtual screening pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from rdkit import Chem

from app.virtual_screening import (
    compute_qed_score,
    count_lipinski_violations,
    extract_descriptor_values,
    run_virtual_screening_pipeline,
)


class TestCountLipinskiViolations:
    """Test Lipinski violation counter."""

    def test_benzene_no_violations(self, benzene_molecule):
        """Test benzene has no Lipinski violations."""
        violations = count_lipinski_violations(benzene_molecule)
        assert violations == 0

    def test_aspirin_one_violation(self, aspirin_molecule):
        """Test aspirin has one Lipinski violation."""
        violations = count_lipinski_violations(aspirin_molecule)
        assert isinstance(violations, int)
        assert violations >= 0

    def test_ethanol_no_violations(self, ethanol_molecule):
        """Test ethanol has no violations."""
        violations = count_lipinski_violations(ethanol_molecule)
        assert violations == 0

    def test_invalid_molecule_returns_four(self):
        """Test that invalid molecule returns 4 violations."""
        violations = count_lipinski_violations(None)
        assert violations == 4

    @patch("app.virtual_screening.get_rdkit_properties")
    def test_get_properties_failure_returns_four(self, mock_get_props):
        """Test that get_rdkit_properties failure returns 4."""
        mock_get_props.return_value = None
        mol = Chem.MolFromSmiles("c1ccccc1")
        violations = count_lipinski_violations(mol)
        assert violations == 4

    @patch("app.virtual_screening.safe_execute")
    @patch("app.virtual_screening.get_rdkit_properties")
    def test_lipinski_rules_failure_returns_four(self, mock_get_props, mock_safe_exec):
        """Test that lipinski_rules failure returns 4."""
        mock_get_props.return_value = {
            "mw": 78,
            "logP": 2.0,
            "tpsa": 0,
            "hbd": 0,
            "hba": 0,
            "rotb": 0,
        }
        mock_safe_exec.return_value = None
        mol = Chem.MolFromSmiles("c1ccccc1")
        violations = count_lipinski_violations(mol)
        assert violations == 4

    @patch("app.virtual_screening.get_rdkit_properties")
    def test_exception_handling(self, mock_get_props):
        """Test exception handling returns 4."""
        mock_get_props.side_effect = Exception("Test error")
        mol = Chem.MolFromSmiles("c1ccccc1")
        violations = count_lipinski_violations(mol)
        assert violations == 4

    @patch("app.virtual_screening.safe_execute")
    @patch("app.virtual_screening.get_rdkit_properties")
    def test_all_rules_passed(self, mock_get_props, mock_safe_exec):
        """Test molecule with all rules passed."""
        mock_get_props.return_value = {
            "mw": 300,
            "logP": 3.0,
            "tpsa": 75,
            "hbd": 2,
            "hba": 5,
            "rotb": 5,
        }
        mock_safe_exec.return_value = {
            "mw": True,
            "logp": True,
            "hbd": True,
            "hba": True,
            "rotb": True,
        }
        mol = Chem.MolFromSmiles("c1ccccc1")
        violations = count_lipinski_violations(mol)
        assert violations == 0

    @patch("app.virtual_screening.safe_execute")
    @patch("app.virtual_screening.get_rdkit_properties")
    def test_multiple_violations(self, mock_get_props, mock_safe_exec):
        """Test molecule with multiple violations."""
        mock_get_props.return_value = {
            "mw": 600,
            "logP": 6.0,
            "tpsa": 150,
            "hbd": 10,
            "hba": 12,
            "rotb": 15,
        }
        mock_safe_exec.return_value = {
            "mw": False,
            "logp": False,
            "hbd": False,
            "hba": False,
            "rotb": False,
        }
        mol = Chem.MolFromSmiles("c1ccccc1")
        violations = count_lipinski_violations(mol)
        assert violations == 5


class TestComputeQedScore:
    """Test QED score computation."""

    def test_benzene_qed_score(self, benzene_molecule):
        """Test QED computation for benzene."""
        qed = compute_qed_score(benzene_molecule)
        assert qed is not None
        assert isinstance(qed, float)
        assert 0 <= qed <= 1

    def test_ethanol_qed_score(self, ethanol_molecule):
        """Test QED computation for ethanol."""
        qed = compute_qed_score(ethanol_molecule)
        assert qed is not None
        assert isinstance(qed, float)
        assert 0 <= qed <= 1

    def test_complex_molecule_qed(self):
        """Test QED for complex molecule."""
        mol = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")  # Ibuprofen
        qed = compute_qed_score(mol)
        assert qed is not None
        assert 0 <= qed <= 1

    def test_invalid_molecule_returns_none(self):
        """Test that invalid molecule returns None."""
        qed = compute_qed_score(None)
        assert qed is None

    @patch("app.virtual_screening.QED.qed")
    def test_qed_exception_returns_none(self, mock_qed):
        """Test that QED exception returns None."""
        mock_qed.side_effect = Exception("QED error")
        mol = Chem.MolFromSmiles("c1ccccc1")
        qed = compute_qed_score(mol)
        assert qed is None


class TestExtractDescriptorValues:
    """Test descriptor extraction."""

    def test_extract_mw_and_logp(self):
        """Test extracting MW and LogP."""
        descriptors = np.array(
            [[100.0, 2.5, 50.0, 1, 2, 3, 0, 1], [200.0, 3.5, 60.0, 2, 3, 4, 1, 2]]
        )
        feature_names = [
            "MW",
            "LogP",
            "TPSA",
            "HBD",
            "HBA",
            "RotBonds",
            "AromaticRings",
            "RingCount",
        ]

        result = extract_descriptor_values(descriptors, feature_names)

        assert "MW" in result
        assert "LogP" in result
        assert len(result["MW"]) == 2
        assert len(result["LogP"]) == 2
        np.testing.assert_array_equal(result["MW"], [100.0, 200.0])
        np.testing.assert_array_equal(result["LogP"], [2.5, 3.5])

    def test_extract_single_row(self):
        """Test extracting from single row."""
        descriptors = np.array([[150.0, 3.0, 55.0, 1, 2, 3, 0, 1]])
        feature_names = [
            "MW",
            "LogP",
            "TPSA",
            "HBD",
            "HBA",
            "RotBonds",
            "AromaticRings",
            "RingCount",
        ]

        result = extract_descriptor_values(descriptors, feature_names)

        assert result["MW"][0] == 150.0
        assert result["LogP"][0] == 3.0

    def test_missing_mw_logp(self):
        """Test handling when MW or LogP not in features."""
        descriptors = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        feature_names = [
            "NotMW",
            "NotLogP",
            "TPSA",
            "HBD",
            "HBA",
            "RotBonds",
            "AromaticRings",
            "RingCount",
        ]

        result = extract_descriptor_values(descriptors, feature_names)

        assert "MW" in result
        assert "LogP" in result
        assert len(result["MW"]) == 0
        assert len(result["LogP"]) == 0

    def test_empty_descriptors(self):
        """Test with empty descriptors array."""
        descriptors = np.array([]).reshape(0, 8)
        feature_names = [
            "MW",
            "LogP",
            "TPSA",
            "HBD",
            "HBA",
            "RotBonds",
            "AromaticRings",
            "RingCount",
        ]

        result = extract_descriptor_values(descriptors, feature_names)

        assert len(result["MW"]) == 0
        assert len(result["LogP"]) == 0


class TestRunVirtualScreeningPipeline:
    """Test main virtual screening pipeline."""

    def test_empty_dataframe(self):
        """Test pipeline with empty input."""
        df = pd.DataFrame({"molecule_id": [], "smiles": []})

        result = run_virtual_screening_pipeline(df)

        assert result["success"] is False
        assert result["invalid_smiles"] == 0
        assert result["total_uploaded"] == 0
        assert len(result["results"]) == 0

    def test_all_invalid_smiles(self):
        """Test pipeline with all invalid SMILES."""
        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1", "MOL_2"],
                "smiles": ["INVALID", "BADSMILES"],
            }
        )

        result = run_virtual_screening_pipeline(df)

        assert result["success"] is False
        assert result["invalid_smiles"] == 2
        assert result["total_uploaded"] == 2
        assert len(result["results"]) == 0

    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_feature_computation_failure_morgan(self, mock_morgan, mock_rdkit):
        """Test pipeline when Morgan fingerprints fail."""
        mock_morgan.return_value = {"success": False, "error": "Morgan error"}
        mock_rdkit.return_value = {"success": True, "X": np.array([[1, 2, 3, 4, 5, 6, 7, 8]])}

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1"],
                "smiles": ["c1ccccc1"],
            }
        )

        result = run_virtual_screening_pipeline(df)

        assert result["success"] is False
        assert "Feature computation failed" in result["error"]

    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_feature_computation_failure_rdkit(self, mock_morgan, mock_rdkit):
        """Test pipeline when RDKit descriptors fail."""
        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048])}
        mock_rdkit.return_value = {"success": False, "error": "RDKit error"}

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1"],
                "smiles": ["c1ccccc1"],
            }
        )

        result = run_virtual_screening_pipeline(df)

        assert result["success"] is False
        assert "Feature computation failed" in result["error"]

    @patch("app.virtual_screening.joblib.load")
    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_model_not_found(self, mock_morgan, mock_rdkit, mock_load):
        """Test pipeline when model file not found."""
        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048])}
        mock_rdkit.return_value = {
            "success": True,
            "X": np.array([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "feature_names": [
                "MW",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "RotBonds",
                "AromaticRings",
                "RingCount",
            ],
        }

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1"],
                "smiles": ["c1ccccc1"],
            }
        )

        with patch("app.virtual_screening.Path.exists", return_value=False):
            result = run_virtual_screening_pipeline(df)

        assert result["success"] is False
        assert "QSAR model not found" in result["error"]

    @patch("app.virtual_screening.QSARPredictor.predict")
    @patch("app.virtual_screening.joblib.load")
    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_successful_screening_all_pass(self, mock_morgan, mock_rdkit, mock_load, mock_predict):
        """Test successful pipeline with all molecules passing Lipinski."""
        # Mock Morgan fingerprints
        mock_morgan.return_value = {
            "success": True,
            "X": np.array([[1] * 2048, [1] * 2048]),
        }

        # Mock RDKit descriptors
        mock_rdkit.return_value = {
            "success": True,
            "X": np.array([[100.0, 2.5, 50.0, 1, 2, 3, 0, 1], [150.0, 3.0, 60.0, 2, 3, 3, 1, 2]]),
            "feature_names": [
                "MW",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "RotBonds",
                "AromaticRings",
                "RingCount",
            ],
        }

        # Mock QSAR predictions
        mock_predict.return_value = np.array([6.5, 7.2])

        # Mock model loading
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1", "MOL_2"],
                "smiles": ["CCO", "c1ccccc1"],
            }
        )

        with patch("app.virtual_screening.Path.exists", return_value=True):
            with patch("app.virtual_screening.count_lipinski_violations", side_effect=[0, 1]):
                with patch("app.virtual_screening.compute_qed_score", side_effect=[0.85, 0.92]):
                    result = run_virtual_screening_pipeline(df)

        assert result["success"] is True
        assert result["total_uploaded"] == 2
        assert result["invalid_smiles"] == 0
        assert result["final_screened"] == 2
        assert len(result["results"]) == 2
        # Results should be sorted by predicted activity (descending)
        assert result["results"].iloc[0]["Predicted Activity (pIC50)"] == 7.2

    @patch("app.virtual_screening.QSARPredictor.predict")
    @patch("app.virtual_screening.joblib.load")
    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_successful_screening_some_lipinski_fail(
        self, mock_morgan, mock_rdkit, mock_load, mock_predict
    ):
        """Test pipeline with some molecules failing Lipinski filter."""
        mock_morgan.return_value = {
            "success": True,
            "X": np.array([[1] * 2048, [1] * 2048, [1] * 2048]),
        }

        mock_rdkit.return_value = {
            "success": True,
            "X": np.array(
                [
                    [100.0, 2.5, 50.0, 1, 2, 3, 0, 1],
                    [150.0, 3.0, 60.0, 2, 3, 3, 1, 2],
                    [600.0, 5.0, 150.0, 5, 8, 10, 2, 3],
                ]
            ),
            "feature_names": [
                "MW",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "RotBonds",
                "AromaticRings",
                "RingCount",
            ],
        }

        mock_predict.return_value = np.array([6.5, 7.2, 8.0])
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1", "MOL_2", "MOL_3"],
                "smiles": ["CCO", "c1ccccc1", "C" * 100],
            }
        )

        with patch("app.virtual_screening.Path.exists", return_value=True):
            with patch("app.virtual_screening.count_lipinski_violations", side_effect=[0, 1, 3]):
                with patch(
                    "app.virtual_screening.compute_qed_score", side_effect=[0.85, 0.92, 0.50]
                ):
                    result = run_virtual_screening_pipeline(df)

        assert result["success"] is True
        assert result["total_uploaded"] == 3
        assert result["invalid_smiles"] == 0
        assert result["final_screened"] == 2
        assert result["lipinski_filtered"] == 1
        assert len(result["results"]) == 2

    @patch("app.virtual_screening.QSARPredictor.predict")
    @patch("app.virtual_screening.joblib.load")
    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_screening_with_nan_qed(self, mock_morgan, mock_rdkit, mock_load, mock_predict):
        """Test pipeline when QED computation returns None."""
        mock_morgan.return_value = {
            "success": True,
            "X": np.array([[1] * 2048]),
        }

        mock_rdkit.return_value = {
            "success": True,
            "X": np.array([[100.0, 2.5, 50.0, 1, 2, 3, 0, 1]]),
            "feature_names": [
                "MW",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "RotBonds",
                "AromaticRings",
                "RingCount",
            ],
        }

        mock_predict.return_value = np.array([6.5])
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1"],
                "smiles": ["c1ccccc1"],
            }
        )

        with patch("app.virtual_screening.Path.exists", return_value=True):
            with patch("app.virtual_screening.count_lipinski_violations", return_value=0):
                with patch("app.virtual_screening.compute_qed_score", return_value=None):
                    result = run_virtual_screening_pipeline(df)

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert pd.isna(result["results"].iloc[0]["QED Score"])

    def test_pipeline_with_missing_columns(self):
        """Test pipeline with missing required columns."""
        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1"],
                # Missing "smiles" column
            }
        )

        result = run_virtual_screening_pipeline(df)

        assert result["success"] is False
        assert result["invalid_smiles"] == 1

    def test_pipeline_with_empty_smiles(self):
        """Test pipeline with empty SMILES strings."""
        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1", "MOL_2"],
                "smiles": ["", "  "],
            }
        )

        result = run_virtual_screening_pipeline(df)

        assert result["success"] is False
        assert result["invalid_smiles"] == 2

    def test_pipeline_with_mixed_valid_invalid(self):
        """Test pipeline with mix of valid and invalid SMILES."""
        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1", "MOL_2", "MOL_3", "MOL_4"],
                "smiles": ["CCO", "INVALID", "c1ccccc1", "BADSMILES"],
            }
        )

        # This should fail at feature computation since 2 valid SMILES
        with patch("app.virtual_screening.compute_morgan_fingerprints") as mock_morgan:
            with patch("app.virtual_screening.compute_rdkit_descriptors") as mock_rdkit:
                mock_morgan.return_value = {
                    "success": True,
                    "X": np.array([[1] * 2048, [1] * 2048]),
                }
                mock_rdkit.return_value = {
                    "success": True,
                    "X": np.array(
                        [[100.0, 2.5, 50.0, 1, 2, 3, 0, 1], [150.0, 3.0, 60.0, 2, 3, 3, 1, 2]]
                    ),
                    "feature_names": [
                        "MW",
                        "LogP",
                        "TPSA",
                        "HBD",
                        "HBA",
                        "RotBonds",
                        "AromaticRings",
                        "RingCount",
                    ],
                }
                with patch(
                    "app.virtual_screening.QSARPredictor.predict", return_value=np.array([6.5, 7.2])
                ):
                    with patch("app.virtual_screening.joblib.load", return_value=MagicMock()):
                        with patch("app.virtual_screening.Path.exists", return_value=True):
                            with patch(
                                "app.virtual_screening.count_lipinski_violations",
                                side_effect=[0, 0],
                            ):
                                with patch(
                                    "app.virtual_screening.compute_qed_score",
                                    side_effect=[0.85, 0.92],
                                ):
                                    result = run_virtual_screening_pipeline(df)

        assert result["invalid_smiles"] == 2
        assert result["total_uploaded"] == 4

    @patch("app.virtual_screening.QSARPredictor.predict")
    @patch("app.virtual_screening.joblib.load")
    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_results_sorting(self, mock_morgan, mock_rdkit, mock_load, mock_predict):
        """Test that results are sorted by predicted activity descending."""
        mock_morgan.return_value = {
            "success": True,
            "X": np.array([[1] * 2048, [1] * 2048, [1] * 2048]),
        }

        mock_rdkit.return_value = {
            "success": True,
            "X": np.array(
                [
                    [100.0, 2.5, 50.0, 1, 2, 3, 0, 1],
                    [150.0, 3.0, 60.0, 2, 3, 3, 1, 2],
                    [120.0, 2.8, 55.0, 1, 2, 3, 0, 1],
                ]
            ),
            "feature_names": [
                "MW",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "RotBonds",
                "AromaticRings",
                "RingCount",
            ],
        }

        # Return predictions in order that will test sorting
        mock_predict.return_value = np.array([5.5, 8.2, 6.7])
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1", "MOL_2", "MOL_3"],
                "smiles": ["CCO", "c1ccccc1", "CCN"],
            }
        )

        with patch("app.virtual_screening.Path.exists", return_value=True):
            with patch("app.virtual_screening.count_lipinski_violations", return_value=0):
                with patch("app.virtual_screening.compute_qed_score", return_value=0.85):
                    result = run_virtual_screening_pipeline(df)

        assert result["success"] is True
        # Check sorting: highest pIC50 first
        assert result["results"].iloc[0]["Predicted Activity (pIC50)"] == 8.2
        assert result["results"].iloc[1]["Predicted Activity (pIC50)"] == 6.7
        assert result["results"].iloc[2]["Predicted Activity (pIC50)"] == 5.5

    @patch("app.virtual_screening.QSARPredictor.predict")
    @patch("app.virtual_screening.joblib.load")
    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_exception_in_pipeline(self, mock_morgan, mock_rdkit, mock_load, mock_predict):
        """Test exception handling in pipeline."""
        mock_morgan.side_effect = Exception("Unexpected error")

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1"],
                "smiles": ["CCO"],
            }
        )

        result = run_virtual_screening_pipeline(df)

        assert result["success"] is False
        assert "error" in result
        assert result["final_screened"] == 0

    @patch("app.virtual_screening.QSARPredictor.predict")
    @patch("app.virtual_screening.joblib.load")
    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_descriptor_extraction_with_nan_values(
        self, mock_morgan, mock_rdkit, mock_load, mock_predict
    ):
        """Test pipeline when descriptor values are NaN."""
        mock_morgan.return_value = {
            "success": True,
            "X": np.array([[1] * 2048]),
        }

        mock_rdkit.return_value = {
            "success": True,
            "X": np.array([[np.nan, np.nan, 50.0, 1, 2, 3, 0, 1]]),
            "feature_names": [
                "MW",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "RotBonds",
                "AromaticRings",
                "RingCount",
            ],
        }

        mock_predict.return_value = np.array([6.5])
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1"],
                "smiles": ["c1ccccc1"],
            }
        )

        with patch("app.virtual_screening.Path.exists", return_value=True):
            with patch("app.virtual_screening.count_lipinski_violations", return_value=0):
                with patch("app.virtual_screening.compute_qed_score", return_value=0.85):
                    result = run_virtual_screening_pipeline(df)

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert pd.isna(result["results"].iloc[0]["MW"])
        assert pd.isna(result["results"].iloc[0]["LogP"])

    @patch("app.virtual_screening.QSARPredictor.predict")
    @patch("app.virtual_screening.joblib.load")
    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_exactly_one_lipinski_violation(self, mock_morgan, mock_rdkit, mock_load, mock_predict):
        """Test that molecules with exactly 1 violation pass (not filtered)."""
        mock_morgan.return_value = {
            "success": True,
            "X": np.array([[1] * 2048]),
        }

        mock_rdkit.return_value = {
            "success": True,
            "X": np.array([[100.0, 2.5, 50.0, 1, 2, 3, 0, 1]]),
            "feature_names": [
                "MW",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "RotBonds",
                "AromaticRings",
                "RingCount",
            ],
        }

        mock_predict.return_value = np.array([6.5])
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1"],
                "smiles": ["c1ccccc1"],
            }
        )

        with patch("app.virtual_screening.Path.exists", return_value=True):
            with patch("app.virtual_screening.count_lipinski_violations", return_value=1):
                with patch("app.virtual_screening.compute_qed_score", return_value=0.85):
                    result = run_virtual_screening_pipeline(df)

        assert result["success"] is True
        assert result["final_screened"] == 1
        assert result["lipinski_filtered"] == 0

    @patch("app.virtual_screening.QSARPredictor.predict")
    @patch("app.virtual_screening.joblib.load")
    @patch("app.virtual_screening.compute_rdkit_descriptors")
    @patch("app.virtual_screening.compute_morgan_fingerprints")
    def test_two_lipinski_violations_filtered(
        self, mock_morgan, mock_rdkit, mock_load, mock_predict
    ):
        """Test that molecules with >1 violation are filtered."""
        mock_morgan.return_value = {
            "success": True,
            "X": np.array([[1] * 2048]),
        }

        mock_rdkit.return_value = {
            "success": True,
            "X": np.array([[100.0, 2.5, 50.0, 1, 2, 3, 0, 1]]),
            "feature_names": [
                "MW",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "RotBonds",
                "AromaticRings",
                "RingCount",
            ],
        }

        mock_predict.return_value = np.array([6.5])
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        df = pd.DataFrame(
            {
                "molecule_id": ["MOL_1"],
                "smiles": ["c1ccccc1"],
            }
        )

        with patch("app.virtual_screening.Path.exists", return_value=True):
            with patch("app.virtual_screening.count_lipinski_violations", return_value=2):
                with patch("app.virtual_screening.compute_qed_score", return_value=0.85):
                    result = run_virtual_screening_pipeline(df)

        assert result["success"] is True
        assert result["final_screened"] == 0
        assert result["lipinski_filtered"] == 1

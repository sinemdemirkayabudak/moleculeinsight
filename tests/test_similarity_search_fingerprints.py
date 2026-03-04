"""Tests for similarity_search fingerprints module."""

import pytest
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity

from app.similarity_search.fingerprints import (
    get_morgan_fp,
    similarity_search,
)


class TestGetMorganFp:
    """Test Morgan fingerprint generation."""

    def test_valid_molecule_fingerprint(self):
        """Test fingerprint generation for valid molecule."""
        mol = Chem.MolFromSmiles("CCO")
        fp = get_morgan_fp(mol, radius=2)
        assert fp is not None
        assert len(fp) == 2048

    def test_different_radii(self):
        """Test fingerprints with different radii."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        fp0 = get_morgan_fp(mol, radius=0)
        fp2 = get_morgan_fp(mol, radius=2)
        fp5 = get_morgan_fp(mol, radius=5)
        
        assert fp0 is not None
        assert fp2 is not None
        assert fp5 is not None
        # Different radii should produce different fingerprints
        assert TanimotoSimilarity(fp0, fp2) < 1.0

    def test_identical_molecules_same_fp(self):
        """Test identical molecules produce identical fingerprints."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")
        fp1 = get_morgan_fp(mol1, radius=2)
        fp2 = get_morgan_fp(mol2, radius=2)
        
        assert TanimotoSimilarity(fp1, fp2) == 1.0

    def test_none_molecule(self):
        """Test fingerprint generation with None molecule."""
        with pytest.raises((TypeError, AttributeError)):
            get_morgan_fp(None, radius=2)

    def test_simple_vs_complex_molecule(self):
        """Test fingerprints for simple vs complex molecules."""
        simple = Chem.MolFromSmiles("C")
        complex_mol = Chem.MolFromSmiles("c1ccc2c(c1)ccc3c2cccc3")
        
        fp_simple = get_morgan_fp(simple, radius=2)
        fp_complex = get_morgan_fp(complex_mol, radius=2)
        
        # Fingerprints should be different
        assert TanimotoSimilarity(fp_simple, fp_complex) < 1.0


class TestSimilaritySearch:
    """Test Tanimoto similarity scoring."""

    def test_identical_fingerprints(self):
        """Test similarity of identical fingerprints."""
        mol = Chem.MolFromSmiles("CCO")
        fp = get_morgan_fp(mol, radius=2)
        
        similarity = similarity_search(fp, [fp])
        assert similarity[0] == 1.0

    def test_different_molecules_similarity(self):
        """Test similarity between different molecules."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CC")
        
        fp1 = get_morgan_fp(mol1, radius=2)
        fp2 = get_morgan_fp(mol2, radius=2)
        
        similarities = similarity_search(fp1, [fp2, fp1])
        
        assert len(similarities) == 2
        assert 0 <= similarities[0] < 1.0
        assert similarities[1] == 1.0

    def test_multiple_reference_fps(self):
        """Test similarity against multiple reference fingerprints."""
        query = Chem.MolFromSmiles("CCO")
        ref1 = Chem.MolFromSmiles("CCO")
        ref2 = Chem.MolFromSmiles("CCCO")
        ref3 = Chem.MolFromSmiles("c1ccccc1")
        
        query_fp = get_morgan_fp(query, radius=2)
        ref_fps = [
            get_morgan_fp(ref1, radius=2),
            get_morgan_fp(ref2, radius=2),
            get_morgan_fp(ref3, radius=2),
        ]
        
        similarities = similarity_search(query_fp, ref_fps)
        
        assert len(similarities) == 3
        assert all(0 <= s <= 1.0 for s in similarities)
        # Query should be most similar to itself
        assert similarities[0] > similarities[1]
        assert similarities[0] > similarities[2]

    def test_empty_reference_list(self):
        """Test with empty reference fingerprint list."""
        query = Chem.MolFromSmiles("CCO")
        query_fp = get_morgan_fp(query, radius=2)
        
        similarities = similarity_search(query_fp, [])
        assert similarities == []

    def test_similarity_range(self):
        """Test that all similarities are in valid range."""
        query = Chem.MolFromSmiles("c1ccccc1")
        refs = [
            Chem.MolFromSmiles("C"),
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("c1ccccc1C"),
        ]
        
        query_fp = get_morgan_fp(query, radius=2)
        ref_fps = [get_morgan_fp(r, radius=2) for r in refs]
        
        similarities = similarity_search(query_fp, ref_fps)
        
        assert all(0 <= s <= 1.0 for s in similarities)

    def test_invalid_fingerprint_type(self):
        """Test similarity_search with invalid fingerprint type."""
        # Pass a string instead of fingerprint object
        with pytest.raises(Exception):
            similarity_search("not_a_fingerprint", ["also_not_fp"])

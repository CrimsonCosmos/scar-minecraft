"""Tests for real-world signal encoders."""

import numpy as np
import pytest

from fpi.live.encoder import NumericEncoder, TextEncoder, AutoEncoder


class TestNumericEncoder:
    def test_nearby_values_similar(self):
        """50.0 and 52.0 should have high cosine similarity."""
        enc = NumericEncoder()
        s1 = enc.encode(50.0)
        s2 = enc.encode(52.0)
        sim = s1.cosine_similarity(s2)
        assert sim > 0.9, f"Nearby values should be similar, got {sim}"

    def test_distant_values_different(self):
        """50.0 and 95.0 should have LOW cosine similarity.

        This is THE key test. Raw scalars [50.0] and [95.0] have cosine
        similarity ~1.0 because they are colinear. Gaussian basis encoding
        must fix this.
        """
        enc = NumericEncoder()
        s1 = enc.encode(50.0)
        s2 = enc.encode(95.0)
        sim = s1.cosine_similarity(s2)
        assert sim < 0.5, f"Distant values should be different, got {sim}"

    def test_signal_dimensionality(self):
        enc = NumericEncoder(num_bases=64)
        sig = enc.encode(42.0)
        assert sig.dim == 64

    def test_modality(self):
        enc = NumericEncoder()
        sig = enc.encode(42.0)
        assert sig.modality == "numeric"

    def test_auto_expand_range(self):
        enc = NumericEncoder(range_min=0, range_max=100)
        enc.encode(200.0)
        assert enc.range_max > 100

    def test_auto_expand_range_negative(self):
        enc = NumericEncoder(range_min=0, range_max=100)
        enc.encode(-50.0)
        assert enc.range_min < 0

    def test_identical_values_identical_signals(self):
        enc = NumericEncoder()
        s1 = enc.encode(42.0)
        s2 = enc.encode(42.0)
        sim = s1.cosine_similarity(s2)
        assert sim == pytest.approx(1.0)

    def test_zero_value(self):
        enc = NumericEncoder()
        sig = enc.encode(0.0)
        assert sig.dim == 64
        assert np.any(sig.data > 0)


class TestTextEncoder:
    def test_similar_lines_similar(self):
        """Similar log lines should have high cosine similarity."""
        enc = TextEncoder()
        s1 = enc.encode("ERROR: connection timeout at 10.0.0.1:8080")
        s2 = enc.encode("ERROR: connection timeout at 10.0.0.2:8080")
        sim = s1.cosine_similarity(s2)
        assert sim > 0.8, f"Similar log lines should be similar, got {sim}"

    def test_different_lines_different(self):
        """Very different log lines should have low cosine similarity."""
        enc = TextEncoder()
        s1 = enc.encode("ERROR: connection timeout at 10.0.0.1:8080")
        s2 = enc.encode("INFO: user logged in successfully from dashboard")
        sim = s1.cosine_similarity(s2)
        assert sim < 0.7, f"Different log lines should be different, got {sim}"

    def test_normalized(self):
        """Output should be L2-normalized."""
        enc = TextEncoder()
        sig = enc.encode("test line with some content")
        norm = np.linalg.norm(sig.data)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_empty_string(self):
        enc = TextEncoder()
        sig = enc.encode("")
        assert sig.dim == 64
        assert np.allclose(sig.data, 0.0)  # No n-grams extracted

    def test_dimensionality(self):
        enc = TextEncoder(dim=64)
        sig = enc.encode("hello world")
        assert sig.dim == 64

    def test_modality(self):
        enc = TextEncoder()
        sig = enc.encode("hello")
        assert sig.modality == "text"


class TestAutoEncoder:
    def test_numeric_detection(self):
        enc = AutoEncoder()
        sig = enc.encode("42.5")
        assert sig.modality == "numeric"

    def test_text_detection(self):
        enc = AutoEncoder()
        sig = enc.encode("ERROR: something happened")
        assert sig.modality == "text"

    def test_percentage_detection(self):
        enc = AutoEncoder()
        sig = enc.encode("45%")
        assert sig.modality == "numeric"

    def test_integer_detection(self):
        enc = AutoEncoder()
        sig = enc.encode("100")
        assert sig.modality == "numeric"

    def test_negative_number(self):
        enc = AutoEncoder()
        sig = enc.encode("-3.14")
        assert sig.modality == "numeric"

    def test_consistent_dimensionality(self):
        """Both numeric and text should produce same dimensionality."""
        enc = AutoEncoder()
        sig_num = enc.encode("42.5")
        sig_text = enc.encode("hello world")
        assert sig_num.dim == sig_text.dim == 64

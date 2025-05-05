import pytest

import concord.modes as modes


def test_annotate_local_exact(monkeypatch):
    # Force embed_sentence to return a constant vector and cosine_sim to 1.0
    monkeypatch.setattr(modes, "embed_sentence", lambda x, cfg: [1, 1, 1])
    monkeypatch.setattr(modes, "cosine_sim", lambda v1, v2: 1.0)
    cfg = {"engine": {"sim_threshold": 0.5}}
    result = modes.annotate_local("a", "b", cfg)
    assert result["label"] == "Exact"
    assert result["similarity"] == 1.0


def test_annotate_local_different(monkeypatch):
    # Force cosine_sim to a low value
    monkeypatch.setattr(modes, "embed_sentence", lambda x, cfg: [1, 2, 3])
    monkeypatch.setattr(modes, "cosine_sim", lambda v1, v2: 0.123)
    cfg = {"engine": {"sim_threshold": 0.5}}
    result = modes.annotate_local("x", "y", cfg)
    assert result["label"] == "Different"
    assert result["similarity"] == pytest.approx(0.123)

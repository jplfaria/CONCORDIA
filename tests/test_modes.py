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


def test_zero_shot_with_sim_hint(monkeypatch):
    import concord.pipeline as pipeline

    # stub LLM call
    monkeypatch.setattr(pipeline, "_call_llm", lambda a, b, prompt, cfg: ("Exact", "E"))
    import concord.modes as modes

    # stub embed + sim
    monkeypatch.setattr(modes, "embed_sentence", lambda x, cfg: [1, 2, 3])
    monkeypatch.setattr(modes, "cosine_sim", lambda v1, v2: 0.555)
    # without sim_hint flag
    cfg = {"engine": {}, "llm": {}, "embedding": {}}
    res = modes.annotate_zero_shot("A", "B", cfg)
    assert res["similarity"] is None
    # with sim_hint flag
    cfg["engine"]["sim_hint"] = True
    res2 = modes.annotate_zero_shot("A", "B", cfg)
    assert res2["similarity"] == 0.555


def test_vote_with_sim_hint(monkeypatch):
    import concord.modes as modes

    # stub embed + sim
    monkeypatch.setattr(modes, "embed_sentence", lambda x, cfg: [0, 0])
    monkeypatch.setattr(modes, "cosine_sim", lambda v1, v2: 0.789)

    # stub LLM client chat
    class DummyClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, prompt):
            return "**Exact — evidence**"

    monkeypatch.setattr(modes, "ArgoGatewayClient", DummyClient)
    # without sim_hint flag
    cfg = {"engine": {}, "llm": {}}
    res = modes.annotate_vote("A", "B", cfg)
    assert res["similarity"] is None
    # with sim_hint flag
    cfg["engine"]["sim_hint"] = True
    res2 = modes.annotate_vote("A", "B", cfg)
    assert res2["similarity"] == 0.789

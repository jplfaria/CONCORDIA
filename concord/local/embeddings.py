from sentence_transformers import SentenceTransformer, util

_MODEL_ID = "pritamdeka/biobert-base-cased-v1-simcse"  # ✅ public
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_ID)
    return _model

def similarity(a: str, b: str) -> float:
    emb = get_model().encode([a, b], convert_to_tensor=True, device="cpu")
    return float(util.cos_sim(emb[0], emb[1]))
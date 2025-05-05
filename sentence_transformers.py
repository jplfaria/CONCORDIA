"""
Dummy sentence_transformers module for testing purposes.
Provides a stub SentenceTransformer class and util.pytorch_cos_sim function.
"""
import torch


class SentenceTransformer:
    def __init__(self, model_id):
        # Stub init: do nothing
        pass

    def to(self, device):
        # Stub method: do nothing
        return self

    def encode(self, texts, convert_to_tensor=True, device=None):
        # Return zeros tensor or list of tensors
        if isinstance(texts, (list, tuple)):
            return [torch.zeros(768) for _ in texts]
        return torch.zeros(768)


class Util:
    @staticmethod
    def pytorch_cos_sim(vec1, vec2):
        # Compute cosine similarity
        return torch.nn.functional.cosine_similarity(
            vec1.unsqueeze(0), vec2.unsqueeze(0)
        )


util = Util()

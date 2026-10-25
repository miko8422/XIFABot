"""Whole-screen image embeddings (SigLIP2) for "how much did the meaning of the screen change"."""

import numpy as np
from PIL import Image

MODEL = "google/siglip2-base-patch16-224"


class ImageEmbedder:
    def __init__(self, model: str = MODEL):
        import torch
        from transformers import AutoModel, AutoProcessor
        self._torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._proc = AutoProcessor.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model).eval().to(self.device)

    def embed(self, image: Image.Image) -> np.ndarray:
        with self._torch.inference_mode():
            inputs = self._proc(images=[image], return_tensors="pt").to(self.device)
            out = self._model.get_image_features(**inputs)
            vec = out if isinstance(out, self._torch.Tensor) else out.pooler_output
            vec = self._torch.nn.functional.normalize(vec.float(), dim=-1)
        return vec[0].cpu().numpy()

"""
End-to-end inference pipeline.
"""

import torch

from quant_pipeline.models.nlp.distilbert import load_model, load_tokenizer
from quant_pipeline.quantization.utils import apply_quantization
from quant_pipeline.data.loaders import load_sst2


class Pipeline:
    """
    Quantized inference pipeline for DistilBERT on SST-2.

    Parameters
    ----------
    model_name : str
        Model identifier (currently only 'distilbert').
    precision : str
        Precision mode: 'fp32', 'fp16', 'int8_ptq', 'int8_qat'.
    """

    def __init__(self, model_name="distilbert", precision="fp32"):
        self.device = torch.device("cpu")
        self.mode = precision

        self.model = load_model()
        self.tokenizer = load_tokenizer()

        train_data = None
        if precision == "int8_qat":
            train_data = load_sst2(split="train", sample_size=300)

        self.model = apply_quantization(
            self.model,
            precision,
            tokenizer=self.tokenizer,
            train_data=train_data,
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        """
        Predict sentiment for a single text input.

        Returns
        -------
        dict
            Keys: label (int), confidence (float), mode (str)
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

        return {
            "label": pred,
            "confidence": torch.softmax(logits, dim=1).max().item(),
            "mode": self.mode,
        }
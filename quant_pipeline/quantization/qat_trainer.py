"""
QAT fine-tuning trainer.

Runs a short training loop so the model adapts to upcoming
quantization precision reduction.
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm


def train_qat(model, tokenizer, texts, labels, epochs=1, lr=5e-5):
    """
    Fine-tune a model for quantization-aware training.

    Parameters
    ----------
    model : nn.Module
        The FP32 model to fine-tune.
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding text.
    texts : list[str]
        Training sentences.
    labels : list[int]
        Corresponding labels.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.

    Returns
    -------
    nn.Module
        The fine-tuned model (still FP32, ready for quantization).
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm(
            zip(texts, labels),
            total=len(texts),
            desc=f"QAT Epoch {epoch + 1}/{epochs}",
        )

        for text, label in loop:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, padding=True
            )
            outputs = model(**inputs, labels=torch.tensor([label]))
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(texts), 1)
        print(f"  Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

    model.eval()
    return model
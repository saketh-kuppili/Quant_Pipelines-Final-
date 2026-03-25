"""
DistilBERT model and tokenizer loading.

Loads the pretrained distilbert-base-uncased-finetuned-sst-2-english
model for binary sentiment classification on SST-2.
"""

import os
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def load_model():
    """Load the pretrained DistilBERT model for SST-2 sentiment classification."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, token=os.environ.get("HF_TOKEN")
    )


def load_tokenizer():
    """Load the tokenizer for the pretrained DistilBERT model."""
    return AutoTokenizer.from_pretrained(
        MODEL_NAME, token=os.environ.get("HF_TOKEN")
    )
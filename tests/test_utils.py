import pytest
from unittest.mock import patch, mock_open
from experiments.models.toxicity_classifier.utils import regular_encode, load_train_set
from transformers import AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_regular_encode():
    texts = ["Hello world", "Testing"]
    tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-roberta-base")
    maxlen = 10
    encoded_tokens, encoded_masks = regular_encode(texts, tokenizer, maxlen)
    assert encoded_tokens.shape == (2, maxlen)
    assert encoded_masks.shape == (2, maxlen)

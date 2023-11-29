import pytest
from utils import regular_encode, load_train_set
from transformers import AutoTokenizer


def test_regular_encode():
    texts = ["Hello world", "Testing"]
    tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-roberta-base")
    maxlen = 10
    encoded_tokens, encoded_masks = regular_encode(texts, tokenizer, maxlen)
    assert encoded_tokens.shape == (2, maxlen)
    assert encoded_masks.shape == (2, maxlen)


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="id,text,lang,toxic\n1,Hello,en,1",
)
def test_load_train_set(mock_file):
    file_name = "fake_train.csv"
    text_field = "text"
    sentiment_fields = ["toxic"]
    lang_field = "lang"
    data_by_lang = load_train_set(file_name, text_field, sentiment_fields, lang_field)
    assert "en" in data_by_lang
    assert len(data_by_lang["en"]) == 1

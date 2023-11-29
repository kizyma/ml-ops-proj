import pytest
from unittest.mock import patch, mock_open
from experiments.models.toxicity_classifier.train import (
    setup_environment,
    configure_gpu,
)
from utils import regular_encode, load_train_set, build_classifier
from transformers import AutoTokenizer


def test_setup_environment():
    with patch("wandb.login"), patch("wandb.init"):
        config = setup_environment(42, "test_project", "test_entity", "test_model")
        assert config.model_name == "test_model"


def test_configure_gpu():
    with patch("tf.config.list_physical_devices", return_value=[]), patch(
        "print"
    ) as mock_print:
        configure_gpu()
        mock_print.assert_called()  # Modify this based on what you expect


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


def test_build_classifier():
    transformer_name = "jplu/tf-xlm-roberta-base"
    max_len = 10
    lr = 0.001
    classifier_model = build_classifier(transformer_name, max_len, lr)
    assert classifier_model is not None
    assert isinstance(classifier_model, tf.keras.Model)

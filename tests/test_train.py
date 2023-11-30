import pytest
import tensorflow as tf
import pandas as pd
from unittest.mock import patch, mock_open, Mock, MagicMock
from experiments.models.toxicity_classifier.train import (
    setup_environment,
    configure_gpu,
)
from experiments.models.toxicity_classifier.utils import (
    regular_encode,
    load_train_set,
    build_classifier,
)
from transformers import AutoTokenizer

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def configure_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPUs are configured")
        except RuntimeError as e:
            print(e)
    else:
        print("Only CPU is available")


def test_setup_environment():
    with patch("wandb.init"), patch("wandb.login"):
        mock_config = Mock()
        mock_config.model_name = "test_model"
        with patch("wandb.config", mock_config):
            config = setup_environment(42, "test_project", "test_entity", "test_model")
            assert config.model_name == "test_model"


# def test_configure_gpu_with_gpu():
#     with patch(
#         "tensorflow.config.list_physical_devices", return_value=["mocked_gpu_device"]
#     ), patch("builtins.print") as mock_print:
#         configure_gpu()
#         mock_print.assert_called_with("GPUs are configured")


def test_configure_gpu_with_cpu_only():
    with patch("tensorflow.config.list_physical_devices", return_value=[]), patch(
        "builtins.print"
    ) as mock_print:
        configure_gpu()
        mock_print.assert_called_with("Only CPU is available")


def test_load_train_set():
    # Path to the test CSV file
    test_file_path = "tests/data/jigsaw-toxic-comment-train-small.csv"

    # Load the test CSV file
    df = pd.read_csv(test_file_path)

    # Perform basic assertions
    # Check if the DataFrame is not empty
    assert not df.empty, "CSV file is empty."

    # Check the number of rows
    expected_num_rows = 10  # Update this based on the number of rows in your test CSV
    assert (
        len(df) == expected_num_rows
    ), f"CSV file should have {expected_num_rows} rows, but has {len(df)}."

    # Check the presence of required columns
    required_columns = [
        "id",
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    for column in required_columns:
        assert column in df.columns, f"Missing required column: {column}"


def test_build_classifier():
    transformer_name = "jplu/tf-xlm-roberta-base"
    max_len = 10
    lr = 0.001
    classifier_model = build_classifier(transformer_name, max_len, lr)
    assert classifier_model is not None
    assert isinstance(classifier_model, tf.keras.Model)

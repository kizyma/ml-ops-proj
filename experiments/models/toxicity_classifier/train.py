import os
import time
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from transformers import AutoTokenizer, XLMRobertaConfig
from utils import (
    load_train_set,
    load_test_set,
    build_dataset,
    build_classifier_v2,
    train_classifier,
    predict_with_classifier,
    show_roc_auc,
)


# Set up the environment
def setup_environment(seed: int, project_name: str, entity_name: str, model_name: str):
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Register TensorFlow Addons optimizers and layers
    tfa.register_all()

    # Initialize W&B
    wandb.login()
    wandb.init(project=project_name, entity=entity_name)

    # Configuration object for hyperparameters
    config = wandb.config
    config.model_name = model_name
    return config


# Configure GPU usage
def configure_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


# Load and prepare training and validation datasets
def load_and_prepare_data(dataset_dir, config):
    # Load training dataset
    corpus_for_training = load_train_set(
        os.path.join(dataset_dir, "jigsaw-toxic-comment-train.csv"),
        text_field="comment_text",
        lang_field="lang",
        sentiment_fields=[
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ],
    )
    assert "en" in corpus_for_training

    # Load validation dataset
    multilingual_corpus = load_train_set(
        os.path.join(dataset_dir, "validation.csv"),
        text_field="comment_text",
        lang_field="lang",
        sentiment_fields=["toxic"],
    )
    assert "en" not in multilingual_corpus

    # Split data for validation and extending training set
    nonenglish_languages = sorted(list(multilingual_corpus.keys()))
    corpus_for_validation = dict()
    for lang in nonenglish_languages:
        random.shuffle(multilingual_corpus[lang])
        n = len(multilingual_corpus[lang]) // 2
        corpus_for_validation[lang] = multilingual_corpus[lang][:n]
        corpus_for_training[lang] = multilingual_corpus[lang][n:]

    return corpus_for_training, corpus_for_validation


# Train the model and perform validation
def train_and_validate(
    corpus_for_training, corpus_for_validation, config, model_save_path
):
    # Load tokenizer and config
    xlmroberta_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    xlmroberta_config = XLMRobertaConfig.from_pretrained(config.model_name)

    # Prepare TensorFlow datasets
    dataset_for_training, n_batches_per_data = build_dataset(
        texts=corpus_for_training,
        dataset_size=150000,
        tokenizer=xlmroberta_tokenizer,
        maxlen=config.max_seq_len,
        batch_size=config.batch_size,
        shuffle=True,
    )

    dataset_for_validation, n_batches_per_epoch = build_dataset(
        texts=corpus_for_validation,
        dataset_size=6000,
        tokenizer=xlmroberta_tokenizer,
        maxlen=config.max_seq_len,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Build the classification model
    xlmr_based_classifier = build_classifier_v2(
        transformer_name=config.model_name,
        max_len=config.max_seq_len,
        lr=config.learning_rate,
    )

    # Training
    experiment_start_time = time.time()
    train_classifier(
        nn=xlmr_based_classifier,
        trainset=dataset_for_training,
        steps_per_trainset=n_batches_per_data,
        steps_per_epoch=min(5 * n_batches_per_epoch, n_batches_per_data),
        validset=dataset_for_validation,
        max_duration=int(round(2.0 * 3600.0 - (time.time() - experiment_start_time))),
        classifier_file_name=model_save_path,
    )

    # Validation and logging results
    val_predictions = predict_with_classifier(
        texts=corpus_for_validation,
        tokenizer=xlmroberta_tokenizer,
        maxlen=config.max_seq_len,
        classifier=xlmr_based_classifier,
        batch_size=config.batch_size,
    )

    calculated_probas = []
    true_labels = []
    for lang in val_predictions:
        probabilities_, true_labels_ = val_predictions[lang]
        calculated_probas.append(probabilities_)
        true_labels.append(true_labels_)
    calculated_probas = np.concatenate(calculated_probas)
    true_labels = np.concatenate(true_labels)

    roc_auc = show_roc_auc(
        y_true=true_labels, probabilities=calculated_probas, label="multi"
    )
    wandb.log({"final_roc_auc": roc_auc})

    return xlmr_based_classifier, val_predictions


# Load test dataset and make predictions
def load_test_and_predict(
    dataset_dir, xlmroberta_tokenizer, xlmr_based_classifier, config
):
    # Load test dataset
    texts_for_submission = load_test_set(
        os.path.join(dataset_dir, "test.csv"),
        text_field="content",
        lang_field="lang",
        id_field="id",
    )

    # Print the number of samples per language
    for language in sorted(list(texts_for_submission.keys())):
        print(f"{language}\t\t{len(texts_for_submission[language])} samples")

    # Predictions on the submission set
    final_predictions = predict_with_classifier(
        texts=texts_for_submission,
        tokenizer=xlmroberta_tokenizer,
        maxlen=config.max_seq_len,
        classifier=xlmr_based_classifier,
        batch_size=config.batch_size,
    )


# Main Function
def main():
    # Set up the environment
    config = setup_environment(
        seed=42,
        project_name="toxic_text_classification",
        entity_name="kizimayarik01",
        model_name="jplu/tf-xlm-roberta-base",
    )

    # Configure GPU usage
    configure_gpu()

    # Specify the dataset directory and model save path
    dataset_dir = "/path/to/your/dataset/directory"
    model_save_path = "/path/to/your/model/save/path.h5"

    # Load and prepare the training and validation datasets
    corpus_for_training, corpus_for_validation = load_and_prepare_data(
        dataset_dir, config
    )

    # Train the model and perform validation
    trained_classifier, val_predictions = train_and_validate(
        corpus_for_training, corpus_for_validation, config, model_save_path
    )

    # Load test dataset and make predictions
    load_test_and_predict(dataset_dir, xlmroberta_tokenizer, trained_classifier, config)


if __name__ == "__main__":
    main()

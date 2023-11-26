# Standard library imports
import codecs
import copy
import csv
import gc
import os
import random
import time
from typing import Dict, List, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (
    Input,
    Dropout,
    Dense,
    GlobalAvgPool1D,
    GlobalMaxPool1D,
)
from tensorflow.keras.regularizers import l2
from tensorflow_addons.optimizers import AdamW
from transformers import (
    TFXLMRobertaModel,
    XLMRobertaConfig,
    AutoTokenizer,
    XLMRobertaTokenizer,
)

# External tooling (like Weights & Biases)
import wandb


class WandBCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            wandb.log(logs)


def generate_random_seed() -> int:
    """
    generate_random_seed

    Returns
    -------
    int
        Random integer
    """
    return random.randint(0, 2147483648)


def regular_encode(
    texts: List[str], tokenizer: XLMRobertaTokenizer, maxlen: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    regular_encode Tokenize inputs for XLMRoberta Model

    Parameters
    ----------
    texts : List[str]
        Texts to be tokenized.
    tokenizer : XLMRobertaTokenizer
        Model Tokenizer Instance.
    maxlen : int
        Max token lenght.

    Returns
    -------
    encoded_tokens [np.ndarray]
        The encoded tokens.

    encoded_masks [np.ndarray]
        The masks for the encoded tokens.
    """
    err_msg = '"{0}" is wrong type for the text list!'.format(type(texts))
    assert isinstance(texts, list) or isinstance(texts, tuple), err_msg
    enc_di = tokenizer.batch_encode_plus(
        texts, return_token_type_ids=False, padding="max_length", max_length=maxlen
    )
    err_msg = "{0} != {1}".format(len(texts), len(enc_di["input_ids"]))
    assert len(texts) == len(enc_di["input_ids"]), err_msg
    err_msg = "{0} != {1}".format(len(texts), len(enc_di["attention_mask"]))
    assert len(texts) == len(enc_di["attention_mask"]), err_msg
    encoded_tokens = np.zeros((len(texts), maxlen), dtype=np.int32)
    encoded_masks = np.zeros((len(texts), maxlen), dtype=np.int32)
    for sample_idx, (encoded_cur_text, encoded_cur_mask) in enumerate(
        zip(enc_di["input_ids"], enc_di["attention_mask"])
    ):
        n_text = len(encoded_cur_text)
        n_mask = len(encoded_cur_mask)
        err_msg = 'Tokens and masks of texts "{0}" are different! ' "{1} != {2}".format(
            texts[sample_idx], n_text, n_mask
        )
        assert n_text == n_mask, err_msg
        if n_text >= maxlen:
            encoded_tokens[sample_idx] = np.array(
                encoded_cur_text[0:maxlen], dtype=np.int32
            )
            encoded_masks[sample_idx] = np.array(
                encoded_cur_mask[0:maxlen], dtype=np.int32
            )
        else:
            padding = [0 for _ in range(maxlen - n_text)]
            encoded_tokens[sample_idx] = np.array(
                encoded_cur_text + padding, dtype=np.int32
            )
            encoded_masks[sample_idx] = np.array(
                encoded_cur_mask + padding, dtype=np.int32
            )
    return encoded_tokens, encoded_masks


def sample_texts(lang_texts, target_size, shuffle):
    """
    Samples texts from the provided list to match the target dataset size.

    Args:
        lang_texts (List[Tuple[str, int]]): List of texts with labels.
        target_size (int): Target size of samples to extract.
        shuffle (bool): Flag to indicate if the sampling should be random.

    Returns:
        List[Tuple[str, int]]: Sampled list of texts with labels.
    """
    sampled_texts = []
    if len(lang_texts) > target_size:
        sampled_texts = random.sample(lang_texts, k=target_size)
    elif len(lang_texts) < target_size:
        sampled_texts = lang_texts.copy()
        additional_samples = target_size - len(lang_texts)
        while additional_samples >= len(lang_texts):
            sampled_texts += lang_texts
            additional_samples -= len(lang_texts)
        if additional_samples > 0:
            sampled_texts += random.sample(lang_texts, k=additional_samples)
    else:
        sampled_texts = lang_texts.copy()
    return sampled_texts


def validate_header(row, required_fields):
    """
    Validate if the necessary fields are present in the header row.

    Args:
        row: The header row of the CSV file.
        required_fields (List[str]): The list of required fields to validate.

    Returns:
        Dict[str, int]: A dictionary with field names as keys and their indices as values.
    """
    field_indices = {}
    for field in required_fields:
        assert field in row, f'File line 1 is wrong! Field "{field}" is not found!'
        field_indices[field] = row.index(field)
    return field_indices


def process_row(row, field_indices, line_idx, process_type):
    """
    Process a single row of the CSV file for text, language, and additional data (either sentiment or ID).

    Args:
        row: A single row from the CSV file.
        field_indices (Dict[str, int]): Dictionary of field names and their indices in the row.
        line_idx (int): Current line index for error reporting.
        process_type (str): Type of processing ('train' for sentiment data, 'test' for ID data).

    Returns:
        Tuple of language, text, and additional data (sentiment label or ID).
    """
    err_msg_base = f"File line {line_idx} is wrong: "
    text = row[field_indices["text_field"]].strip()
    assert len(text) > 0, err_msg_base + "Text is empty!"

    cur_lang = (
        row[field_indices["lang_field"]].strip()
        if "lang_field" in field_indices
        else "en"
    )
    assert len(cur_lang) > 0, err_msg_base + "Language is empty!"

    if process_type == "train":
        max_proba = max(
            float(row[field_indices[field]])
            for field in field_indices
            if field.startswith("sentiment_field")
        )
        new_label = 1 if max_proba >= 0.5 else 0
        return cur_lang, text, new_label
    elif process_type == "test":
        try:
            id_value = int(row[field_indices["id_field"]])
            assert id_value >= 0, (
                err_msg_base + f"{row[field_indices['id_field']]} is wrong ID!"
            )
        except ValueError:
            raise ValueError(
                err_msg_base
                + f"{row[field_indices['id_field']]} is not a valid integer ID!"
            )
        return cur_lang, text, id_value


def load_train_set(
    file_name: str, text_field: str, sentiment_fields: List[str], lang_field: str
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Load and process training data from a CSV file. Validates header fields, processes each row for text,
    sentiment, and language data, and aggregates the results by language.

    Args:
        file_name (str): Path to the CSV file containing the data.
        text_field (str): The name of the field containing the text data.
        sentiment_fields (List[str]): List of fields containing sentiment data.
        lang_field (str): The name of the field containing language information.

    Returns:
        Dict[str, List[Tuple[str, int]]]: A dictionary with language as keys and a list of tuples containing text and sentiment label.
    """
    assert len(sentiment_fields) > 0, "List of sentiment fields is empty!"
    data_by_lang = {}
    line_idx = 1

    with codecs.open(file_name, mode="r", encoding="utf-8", errors="ignore") as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=",")
        header = next(data_reader)
        required_fields = (
            [text_field]
            + ["sentiment_field" + str(i) for i in range(len(sentiment_fields))]
            + [lang_field]
        )
        field_indices = validate_header(header, required_fields)

        for row in data_reader:
            if len(row) == len(header):
                cur_lang, text, new_label = process_row(
                    row, field_indices, line_idx, "train"
                )
                if cur_lang not in data_by_lang:
                    data_by_lang[cur_lang] = []
                data_by_lang[cur_lang].append((text, new_label))

            if line_idx % 10000 == 0:
                print(f'{line_idx} lines of "{file_name}" have been processed...')
            line_idx += 1

    if (line_idx - 1) % 10000 != 0:
        print(f'{line_idx - 1} lines of "{file_name}" have been processed...')

    return data_by_lang


def load_test_set(
    file_name: str, id_field: str, text_field: str, lang_field: str
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Load and process test data from a CSV file. Validates header fields, processes each row for ID, text,
    and language data, and aggregates the results by language.

    Args:
        file_name (str): Path to the CSV file containing the data.
        id_field (str): The name of the field containing the ID.
        text_field (str): The name of the field containing the text data.
        lang_field (str): The name of the field containing language information.

    Returns:
        Dict[str, List[Tuple[str, int]]]: A dictionary with language as keys and a list of tuples containing text and ID.
    """
    data_by_lang = {}
    line_idx = 1

    with codecs.open(file_name, mode="r", encoding="utf-8", errors="ignore") as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=",")
        header = next(data_reader)
        required_fields = [id_field, text_field, lang_field]
        field_indices = validate_header(header, required_fields)

        for row in data_reader:
            if len(row) == len(header):
                cur_lang, text, id_value = process_row(
                    row, field_indices, line_idx, "test"
                )
                if cur_lang not in data_by_lang:
                    data_by_lang[cur_lang] = []
                data_by_lang[cur_lang].append((text, id_value))

            if line_idx % 10000 == 0:
                print(f'{line_idx} lines of "{file_name}" have been processed...')
            line_idx += 1

    if (line_idx - 1) % 10000 != 0:
        print(f'{line_idx - 1} lines of "{file_name}" have been processed...')

    return data_by_lang


def build_dataset(
    texts: Dict[str, List[Tuple[str, int]]],
    dataset_size: int,
    tokenizer: XLMRobertaTokenizer,
    maxlen: int,
    batch_size: int,
    shuffle: bool,
) -> Tuple[tf.data.Dataset, int]:
    """
    Build a dataset from the given texts, tokenizer, and configuration parameters.

    Args:
        texts (Dict[str, List[Tuple[str, int]]]): Texts grouped by language, each with a label.
        dataset_size (int): Total size of the dataset to be built.
        tokenizer (XLMRobertaTokenizer): Tokenizer to be used for text encoding.
        maxlen (int): Maximum length of the tokenized sequence.
        batch_size (int): Batch size for the dataset.
        shuffle (bool): Flag to indicate if the dataset should be shuffled.

    Returns:
        Tuple[tf.data.Dataset, int]: A TensorFlow dataset and the number of steps per epoch.
    """
    texts_and_labels = []
    dataset_size_by_lang = dataset_size // len(texts)

    for lang, lang_texts in texts.items():
        print(f"{lang}:")
        sampled_texts = sample_texts(lang_texts, dataset_size_by_lang, shuffle)
        texts_and_labels += sampled_texts
        print(f"  number of samples is {len(sampled_texts)};")

    if shuffle:
        random.shuffle(texts_and_labels)

    n_steps = len(texts_and_labels) // batch_size
    print(f"Samples number of the data set is {len(texts_and_labels)}.")

    tokens_of_texts, mask_of_texts = regular_encode(
        texts=[text for text, _ in texts_and_labels], tokenizer=tokenizer, maxlen=maxlen
    )
    toxicity_labels = np.array([label for _, label in texts_and_labels], dtype=np.int32)
    print(
        f"Number of positive siamese samples is {int(sum(toxicity_labels))} from {toxicity_labels.shape[0]}."
    )

    err_msg = (
        f"{len(texts_and_labels)} is too small number of samples for the data set!"
    )
    assert n_steps >= 50 or not shuffle, err_msg

    dataset = tf.data.Dataset.from_tensor_slices(
        ((tokens_of_texts, mask_of_texts), toxicity_labels)
    )
    if shuffle:
        dataset = dataset.repeat().batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)

    del texts_and_labels
    return dataset, n_steps


def build_classifier_multilabel(
    transformer_name: str, max_len: int, lr: float
) -> tf.keras.Model:
    """
    build_classifier Create Keras model.

    Parameters
    ----------
    transformer_name : str
        Transformer model to be used.
    max_len : int
        Max token length for transformer.
    lr : float
        Rearning rate.

    Returns
    -------
    tf.keras.Model
        Keras model instance.
    """
    word_ids = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="base_word_ids"
    )
    attention_mask = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="base_attention_mask"
    )
    transformer_layer = TFXLMRobertaModel.from_pretrained(
        pretrained_model_name_or_path=transformer_name, name="Transformer"
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    pooled_output = tf.keras.layers.GlobalAvgPool1D(name="AvePool")(
        sequence_output, mask=attention_mask
    )
    kernel_init = tf.keras.initializers.GlorotNormal(seed=generate_random_seed())
    bias_init = tf.keras.initializers.Constant(value=0.0)
    cls_layer = tf.keras.layers.Dense(
        units=6,
        activation="sigmoid",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name="OutputLayer",
    )(pooled_output)
    cls_model = tf.keras.Model(
        inputs=[word_ids, attention_mask], outputs=cls_layer, name="ToxicityClassifier"
    )
    cls_model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss="binary_crossentropy",
    )
    return cls_model


def build_transformer_inputs(max_len: int):
    """
    Build inputs for the transformer model.

    Args:
        max_len (int): Maximum sequence length.

    Returns:
        Tuple of Input tensors for word ids and attention mask.
    """
    word_ids = Input(shape=(max_len,), dtype=tf.int32, name="base_word_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="base_attention_mask")
    return word_ids, attention_mask


def apply_pooling(sequence_output, attention_mask, pooling_type: str):
    """
    Apply pooling to the sequence output of a transformer model.

    Args:
        sequence_output: Output from the transformer model.
        attention_mask: Attention mask for the input sequences.
        pooling_type (str): Type of pooling ('average' or 'max').

    Returns:
        Pooled output tensor.
    """
    if pooling_type == "average":
        return GlobalAvgPool1D(name="AvePool")(sequence_output, mask=attention_mask)
    elif pooling_type == "max":
        mask_expanded = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)
        masked_sequence = sequence_output + (1.0 - mask_expanded) * -1e9
        return GlobalMaxPool1D(name="MaxPool")(masked_sequence)


def build_classifier(transformer_name: str, max_len: int, lr: float) -> tf.keras.Model:
    """
    Build a classifier model using a transformer as the base layer.

    Args:
        transformer_name (str): Name of the pretrained transformer model.
        max_len (int): Maximum sequence length.
        lr (float): Learning rate for the model optimizer.

    Returns:
        tf.keras.Model: Compiled classification model.
    """
    word_ids, attention_mask = build_transformer_inputs(max_len)
    transformer_layer = TFXLMRobertaModel.from_pretrained(
        pretrained_model_name_or_path=transformer_name, name="Transformer"
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    dropout_layer = Dropout(0.3)(sequence_output)
    pooled_output = apply_pooling(dropout_layer, attention_mask, pooling_type="average")

    l2_regularizer = l2(1e-5)
    cls_layer = Dense(
        units=1,
        activation="sigmoid",
        kernel_initializer=tf.keras.initializers.GlorotNormal(
            seed=generate_random_seed()
        ),
        bias_initializer=tf.keras.initializers.Constant(value=0.0),
        kernel_regularizer=l2_regularizer,
        name="OutputLayer",
    )(pooled_output)

    cls_model = tf.keras.Model(
        inputs=[word_ids, attention_mask], outputs=cls_layer, name="ToxicityClassifier"
    )
    cls_model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=1e-5), loss="binary_crossentropy"
    )
    return cls_model


def build_classifier_v2(
    transformer_name: str, max_len: int, lr: float
) -> tf.keras.Model:
    """
    Build a classifier model using a transformer as the base layer and applying global max pooling.

    Args:
        transformer_name (str): Name of the pretrained transformer model.
        max_len (int): Maximum sequence length.
        lr (float): Learning rate for the model optimizer.

    Returns:
        tf.keras.Model: Compiled classification model.
    """
    word_ids, attention_mask = build_transformer_inputs(max_len)
    transformer_layer = TFXLMRobertaModel.from_pretrained(
        pretrained_model_name_or_path=transformer_name, name="Transformer"
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    dropout_layer = Dropout(0.5)(sequence_output)
    pooled_output = apply_pooling(dropout_layer, attention_mask, pooling_type="max")

    l2_regularizer = l2(1e-5)
    cls_layer = Dense(
        units=1,
        activation="sigmoid",
        kernel_initializer=tf.keras.initializers.GlorotNormal(
            seed=generate_random_seed()
        ),
        bias_initializer=tf.keras.initializers.Constant(value=0.0),
        kernel_regularizer=l2_regularizer,
        name="OutputLayer",
    )(pooled_output)

    cls_model = tf.keras.Model(
        inputs=[word_ids, attention_mask], outputs=cls_layer, name="ToxicityClassifier"
    )
    cls_model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=1e-5), loss="binary_crossentropy"
    )
    return cls_model


def show_training_process(
    history: tf.keras.callbacks.History, metric_name: str, figure_id: int = 1
):
    """
    show_training_process
        Compute metrics and plot validation curves

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Keras model history object.
    metric_name : str
        Name of the metric to be computed.
    figure_id : int, optional
        Unique identifier for plot, by default 1.
    """
    val_metric_name = "val_" + metric_name
    err_msg = 'The metric "{0}" is not found! Available metrics are: {1}'.format(
        metric_name, list(history.history.keys())
    )
    assert metric_name in history.history, err_msg
    plt.figure(figure_id, figsize=(5, 5))
    plt.plot(
        list(range(len(history.history[metric_name]))),
        history.history[metric_name],
        label="Training {0}".format(metric_name),
    )
    if val_metric_name in history.history:
        assert len(history.history[metric_name]) == len(
            history.history["val_" + metric_name]
        )
        plt.plot(
            list(range(len(history.history["val_" + metric_name]))),
            history.history["val_" + metric_name],
            label="Validation {0}".format(metric_name),
        )
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title("Training process")
    plt.legend(loc="best")
    plt.show()


def train_classifier(
    nn: tf.keras.Model,
    trainset: tf.data.Dataset,
    steps_per_trainset: int,
    steps_per_epoch: int,
    validset: tf.data.Dataset,
    max_duration: int,
    classifier_file_name: str,
):
    """
    train_classifier
        Train Keras model

    Parameters
    ----------
    nn : tf.keras.Model
        Keras model instance
    trainset : tf.data.Dataset
        Train data
    steps_per_trainset : int
    steps_per_epoch : int
        Steps per epoch.
    validset : tf.data.Dataset
        validation Data
    max_duration : int
        Estimated time of training.
    classifier_file_name : str
        Model name
    """
    assert steps_per_trainset >= steps_per_epoch
    n_epochs = int(round(10.0 * steps_per_trainset / float(steps_per_epoch)))
    print(
        f"Maximal duration of the XLMR-based classifier training is {max_duration} seconds."
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
            verbose=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
            save_best_only=True,
            filepath=classifier_file_name,
        ),
        tfa.callbacks.TimeStopping(seconds=max_duration, verbose=True),
    ]
    # Include the WandB callback
    callbacks.append(WandBCallback())
    history = nn.fit(
        trainset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validset,
        epochs=n_epochs,
        callbacks=callbacks,
    )
    show_training_process(history, "loss")
    nn.load_weights(classifier_file_name)


def sample_indices(text_length, max_size_per_lang):
    """
    Sample indices based on the maximum size per language.

    Args:
        text_length (int): Length of the text list for a particular language.
        max_size_per_lang (int): Maximum number of samples per language.

    Returns:
        List[int]: List of sampled indices.
    """
    if max_size_per_lang > 0 and text_length > max_size_per_lang:
        return random.sample(population=range(text_length), k=max_size_per_lang)
    return list(range(text_length))


def predict_batches(classifier, tokens_of_texts, mask_of_texts, batch_size):
    """
    Predict the output in batches.

    Args:
        classifier (tf.keras.Model): The classifier model.
        tokens_of_texts (np.ndarray): Tokenized texts.
        mask_of_texts (np.ndarray): Masks for the texts.
        batch_size (int): Size of each batch.

    Returns:
        np.ndarray: Concatenated predictions.
    """
    predictions = []
    n_batches = int(np.ceil(len(tokens_of_texts) / float(batch_size)))

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(len(tokens_of_texts), batch_start + batch_size)
        res = classifier.predict_on_batch(
            [
                tokens_of_texts[batch_start:batch_end],
                mask_of_texts[batch_start:batch_end],
            ]
        )
        predictions.append(res.reshape((res.shape[0],)))

    return np.concatenate(predictions)


def predict_with_classifier(
    texts: Dict[str, List[Tuple[str, int]]],
    tokenizer: XLMRobertaTokenizer,
    maxlen: int,
    classifier: tf.keras.Model,
    batch_size: int,
    max_dataset_size: int = 0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict using the classifier for texts in different languages.

    Args:
        texts (Dict[str, List[Tuple[str, int]]]): Texts to predict, grouped by language.
        tokenizer (XLMRobertaTokenizer): Tokenizer for text encoding.
        maxlen (int): Maximum length of sequences.
        classifier (tf.keras.Model): Classifier model.
        batch_size (int): Batch size for prediction.
        max_dataset_size (int): Maximum size of the dataset to be used for prediction.

    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: Predictions and identifiers, grouped by language.
    """
    languages = sorted(texts.keys())
    predictions_by_languages = {}
    max_size_per_lang = (
        max_dataset_size // len(languages) if max_dataset_size > 0 else 0
    )
    assert (
        max_size_per_lang > 0 or max_dataset_size == 0
    ), f"{max_dataset_size} is too small number of dataset samples!"

    for cur_lang in languages:
        selected_indices = sample_indices(len(texts[cur_lang]), max_size_per_lang)
        tokens_of_texts, mask_of_texts = regular_encode(
            [texts[cur_lang][idx][0] for idx in selected_indices], tokenizer, maxlen
        )
        predictions = predict_batches(
            classifier, tokens_of_texts, mask_of_texts, batch_size
        )
        identifiers = np.array(
            [texts[cur_lang][idx][1] for idx in selected_indices], dtype=np.int32
        )
        predictions_by_languages[cur_lang] = (predictions, identifiers)

    return predictions_by_languages


def show_roc_auc(
    y_true: np.ndarray, probabilities: np.ndarray, label: str, figure_id: int = 1
):
    plt.figure(figure_id, figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--")
    print(
        "ROC-AUC score for {0} is {1:.9f}".format(
            label, roc_auc_score(y_true=y_true, y_score=probabilities)
        )
    )
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=probabilities)
    plt.plot(fpr, tpr, label=label.title())
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.show()


def plot_learning_evolution(r):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(r.history["loss"], label="Loss")
    plt.plot(r.history["val_loss"], label="val_Loss")
    plt.title("Loss evolution during trainig")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(r.history["AUC"], label="AUC")
    plt.plot(r.history["val_AUC"], label="val_AUC")
    plt.title("AUC score evolution during trainig")
    plt.legend()

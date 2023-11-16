import os
import time
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import AutoTokenizer, XLMRobertaConfig
import wandb

from utils import load_train_set, load_test_set, build_dataset, build_classifier, build_classifier_v2, train_classifier, predict_with_classifier, show_roc_auc

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Register TensorFlow Addons optimizers and layers
tfa.register_all()

# W&B login and initialization
wandb.login()
wandb.init(project='toxic_text_classification', entity='kizimayarik01')

# W&B configuration
config = wandb.config
config.learning_rate = 1e-5
config.max_seq_len = 256
config.model_name = "jplu/tf-xlm-roberta-base"
config.batch_size = 32  # Adjust this based on your GPU

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Dataset directory
dataset_dir = "/home/krabli/Desktop/ml-ops/ml-ops-proj/data/dataset"
model_save_path = "/home/krabli/Desktop/ml-ops/ml-ops-proj/data/models/xlmr_for_toxicity.h5"

# Load tokenizer and config
xlmroberta_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
xlmroberta_config = XLMRobertaConfig.from_pretrained(config.model_name)

# Ensure sequence length is within model limits
config.sentence_embedding_size = xlmroberta_config.hidden_size
assert config.max_seq_len <= xlmroberta_config.max_position_embeddings

# Loading and preparing training and validation datasets
corpus_for_training = load_train_set(
    os.path.join(dataset_dir, "jigsaw-toxic-comment-train.csv"),
    text_field="comment_text", lang_field="lang", sentiment_fields=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
)
assert "en" in corpus_for_training

multilingual_corpus = load_train_set(
    os.path.join(dataset_dir, "validation.csv"),
    text_field="comment_text", lang_field="lang", sentiment_fields=["toxic"],
)
assert "en" not in multilingual_corpus

# Splitting data for validation and extending training set
nonenglish_languages = sorted(list(multilingual_corpus.keys()))
corpus_for_validation = dict()
for lang in nonenglish_languages:
    random.shuffle(multilingual_corpus[lang])
    n = len(multilingual_corpus[lang]) // 2
    corpus_for_validation[lang] = multilingual_corpus[lang][:n]
    corpus_for_training[lang] = multilingual_corpus[lang][n:]

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

# Model building
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
    classifier_file_name=model_save_path,  # Use the model save path here
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

roc_auc = show_roc_auc(y_true=true_labels, probabilities=calculated_probas, label="multi")
wandb.log({"final_roc_auc": roc_auc})

# Load test dataset and print the number of samples per language
texts_for_submission = load_test_set(
    os.path.join(dataset_dir, "test.csv"),
    text_field="content",
    lang_field="lang",
    id_field="id",
)

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

# (Optional) Log final_predictions to W&B, or save them as needed
wandb.log({"final_predictions": final_predictions})

# Close W&B run
wandb.finish()

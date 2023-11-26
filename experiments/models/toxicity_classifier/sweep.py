import os
import wandb
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from transformers import AutoTokenizer, TFXLMRobertaModel
from utils import load_train_set, build_dataset
from wandb.keras import WandbCallback


# Set up the W&B sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-6,
            'max': 1e-4
        },
        'batch_size': {
            'values': [16, 32, 64] # might be a good idea to reduce a batch size, caught
        },
        'dropout_rate': {
            'min': 0.1,
            'max': 0.5
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'max_len': {
            'value': 256  # Define a fixed max_len or a range if variable length is needed
        }
    }
}


# Tokenizer for text preprocessing
tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-roberta-base")


def build_classifier(transformer_name, max_len, lr, dropout_rate, optimizer_name):
    # Define input layers
    word_ids = Input(shape=(max_len,), dtype=tf.int32, name="base_word_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="base_attention_mask")
    
    # Load pre-trained transformer layer
    transformer_layer = TFXLMRobertaModel.from_pretrained(transformer_name, name="Transformer")
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    
    # Apply dropout for regularization
    dropout = Dropout(dropout_rate)(sequence_output)

    # Global Average Pooling
    pooled_output = GlobalAveragePooling1D()(dropout)

    # Output layer with sigmoid activation
    output = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-5))(pooled_output)
    
    # Select optimizer based on the parameter
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Compile the model
    model = tf.keras.Model(inputs=[word_ids, attention_mask], outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model


def preprocess_and_load_data(file_path, max_len, batch_size, tokenizer, is_validation=False):
    # Load the data
    sentiment_fields = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] if not is_validation else ["toxic"]
    corpus = load_train_set(
        file_path,
        text_field="comment_text",
        lang_field="lang",
        sentiment_fields=sentiment_fields
    )

    # Handling for validation dataset
    if is_validation:
        nonenglish_languages = sorted(list(corpus.keys()))
        corpus_for_validation = dict()
        for lang in nonenglish_languages:
            random.shuffle(corpus[lang])
            n = len(corpus[lang]) // 2
            corpus_for_validation[lang] = corpus[lang][:n]
            corpus[lang] = corpus[lang][n:]
        corpus = corpus_for_validation

    # Prepare TensorFlow datasets
    dataset, steps = build_dataset(
        texts=corpus,
        dataset_size=150000 if not is_validation else 6000,
        tokenizer=tokenizer,
        maxlen=max_len,
        batch_size=batch_size,
        shuffle=True
    )
    return dataset, steps


def train():
    wandb.init()
    config = wandb.config
    dataset_dir = "/home/krabli/Desktop/ml-ops/ml-ops-proj/data/dataset"  

    # Training data
    train_data, train_steps = preprocess_and_load_data(
        os.path.join(dataset_dir, "jigsaw-toxic-comment-train.csv"), 
        config.max_len, 
        config.batch_size,
        tokenizer
    )

    # Validation data
    val_data, val_steps = preprocess_and_load_data(
        os.path.join(dataset_dir, "validation.csv"), 
        config.max_len, 
        config.batch_size,
        tokenizer,
        is_validation=True
    )

    model = build_classifier("jplu/tf-xlm-roberta-base", config.max_len, 
                                config.learning_rate, config.dropout_rate,
                                config.optimizer)

    history = model.fit(
        train_data, 
        steps_per_epoch=train_steps, 
        validation_data=val_data, 
        validation_steps=val_steps, 
        epochs=10, 
        callbacks=[WandbCallback()]
    )

    for metric, values in history.history.items():
        for epoch, value in enumerate(values):
            wandb.log({f'{metric}_epoch': value}, step=epoch)

    val_loss, val_accuracy = model.evaluate(val_data, steps=val_steps)
    wandb.log({'val_accuracy': val_accuracy, 'val_loss': val_loss})

    wandb.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="toxicity_classification_sweep")
    wandb.agent(sweep_id, train, count=25)

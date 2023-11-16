import numpy as np
import tensorflow as tf
from transformers import TFXLMRobertaModel, AutoTokenizer
import wandb
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from utils import preprocess_data, load_train_set, load_test_set

# Set up the W&B sweep configuration
sweep_config = {
    'method': 'bayesian',  
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
            'values': [16, 32, 64]
        },
        'dropout_rate': {
            'min': 0.1,
            'max': 0.5
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        }
    }
}

# Tokenizer for text preprocessing
tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-roberta-base")


def build_classifier(transformer_name, max_len, lr, dropout_rate):
    word_ids = Input(shape=(max_len,), dtype=tf.int32, name="base_word_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="base_attention_mask")
    
    transformer_layer = TFXLMRobertaModel.from_pretrained(transformer_name, name="Transformer")
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    
    dropout = Dropout(dropout_rate)(sequence_output)
    pooled_output = GlobalAveragePooling1D()(dropout)
    output = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-5))(pooled_output)
    
    model = tf.keras.Model(inputs=[word_ids, attention_mask], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')
    return model



def train():
    # Initialize a new W&B run
    wandb.init()

    # Extract hyperparameters from W&B config
    config = wandb.config

    # Load your dataset
    dataset_dir = "/home/krabli/Desktop/ml-ops/ml-ops-proj/data/dataset"  # Update this path
    train_texts, train_labels = load_train_set(os.path.join(dataset_dir, "train.csv"))
    val_texts, val_labels = load_test_set(os.path.join(dataset_dir, "val.csv"))

    # Preprocess data
    train_data = preprocess_data(train_texts, train_labels, config.max_len, config.batch_size)
    val_data = preprocess_data(val_texts, val_labels, config.max_len, config.batch_size)

    # Build the model
    model = build_classifier("jplu/tf-xlm-roberta-base", config.max_len, config.learning_rate, config.dropout_rate)

    # Train the model
    history = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[WandbCallback()])

    # Log additional metrics or perform evaluation
    val_loss, val_accuracy = model.evaluate(val_data)
    wandb.log({'val_accuracy': val_accuracy, 'val_loss': val_loss})

    wandb.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="toxicity_classification_sweep")
    wandb.agent(sweep_id, train, count=25)

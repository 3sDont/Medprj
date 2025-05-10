
# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
import re
# standard
import numpy as np
import pandas as pd

# visualize
from tqdm.notebook import tqdm
from itertools import product

# system
import pickle     ## saving library
import os         ## file manager
import sys
import multiprocessing
from multiprocessing import Pool
import time

import functools

import datetime


from tensorflow.keras.layers import Input, Layer
from transformers import AutoConfig, TFAutoModel
from tensorflow.keras.utils import to_categorical
from transformers import AutoTokenizer, AutoModel, AutoConfig

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer
from transformers import AutoConfig, TFAutoModel

# Hàm bổ trợ
def reduce_sum(te):
    return tf.reduce_sum(te, 1) / 512

def l2(x, z=1):
    return tf.sqrt(tf.reduce_sum(tf.square(x), z))  # shape (n,)

def cosine(x, y):
    u = tf.tensordot(x, tf.transpose(y), axes=1)  # shape (n,t) and (m,t) => axes=1
    v = tf.tensordot(l2(x, z=2), tf.transpose(l2(y)), axes=0)  # shape (n,) and (m,) => axes=0
    return u / v

# Lớp Keras bọc TFDistilBertModel
class BertLayer(Layer):
    def __init__(self, model_name, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        config = AutoConfig.from_pretrained(model_name, dropout=0.2, attention_dropout=0.2)
        config.output_hidden_states = False
        # Use TFAutoModelForMaskedLM for TensorFlow compatibility
        self.transformer_model = TFAutoModel.from_pretrained(model_name, config=config,from_pt=True)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        # Access the output correctly for TFAutoModelForMaskedLM
        return self.transformer_model(input_ids=input_ids, attention_mask=attention_mask,)[0]

def create_model_WithAim(aims_input_ids_in,aims_input_masks_in,num_classes,model_name):
    # Đầu vào
    input_ids_in = Input(shape=(512,), name='input_token', dtype='int32')
    input_masks_in = Input(shape=(512,), name='masked_token', dtype='int8')

    # Chuyển đổi kiểu dữ liệu nếu cần
    aims_input_ids_in = tf.cast(aims_input_ids_in, dtype=tf.int32)
    aims_input_masks_in = tf.cast(aims_input_masks_in, dtype=tf.int8)

    # Embedding từ transformer
    transformer_layer = BertLayer(model_name=model_name)
    embedding_layer = transformer_layer([input_ids_in, input_masks_in])
    embedding_layer1 = transformer_layer([input_ids_in, input_masks_in])[:, 0, :]  # CLS token

    # Embedding cho aims
    # aims_embedding_layer = transformer_layer([aims_input_ids_in, aims_input_masks_in])[:, 0, :]

# Embedding cho aims (using batch processing)
    dataset = tf.data.Dataset.from_tensor_slices((aims_input_ids_in, aims_input_masks_in)).batch(64) # Batch size of 32
    aims_embedding_layer = []
    for batch_ids, batch_masks in dataset:
        batch_embedding = transformer_layer([batch_ids, batch_masks])[:, 0, :]
        aims_embedding_layer.append(batch_embedding)
    aims_embedding_layer = tf.concat(aims_embedding_layer, axis=0)

#     Tính cosine similarity
    b = tf.keras.layers.Reshape((1, 768))(embedding_layer1)
    Y = tf.keras.layers.Lambda(
        lambda t: cosine(t, aims_embedding_layer),
        output_shape=(1406,),  # Thay num_classes bằng giá trị phù hợp
        name="lambda"
        )(b)
    Y = tf.keras.layers.Reshape((1406,))(Y)

    # Feature extraction
    X1 = tf.keras.layers.Conv1D(200, 4, activation='relu', padding="same")(embedding_layer)
    X1 = tf.keras.layers.GlobalMaxPooling1D()(X1)

    X2 = tf.keras.layers.Conv1D(200, 3, activation='relu', padding="same")(embedding_layer)
    X2 = tf.keras.layers.GlobalMaxPooling1D()(X2)

    X3 = tf.keras.layers.Conv1D(200, 2, activation='relu', padding="same")(embedding_layer)
    X3 = tf.keras.layers.GlobalMaxPooling1D()(X3)

    X = tf.keras.layers.Concatenate(axis=1)([X1, X2, X3,Y])
    X = tf.keras.layers.Dense(500, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(400, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)

    # Output layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(X)

    # Tạo model
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=output)
    return model

def create_model_NoAim(num_classes,model_name):
    # Đầu vào
    input_ids_in = Input(shape=(512,), name='input_token', dtype='int32')
    input_masks_in = Input(shape=(512,), name='masked_token', dtype='int8')

    # Embedding từ transformer
    transformer_layer = BertLayer(model_name=model_name)
    embedding_layer = transformer_layer([input_ids_in, input_masks_in])


    # Feature extraction
    X1 = tf.keras.layers.Conv1D(200, 4, activation='relu', padding="same")(embedding_layer)
    X1 = tf.keras.layers.GlobalMaxPooling1D()(X1)

    X2 = tf.keras.layers.Conv1D(200, 3, activation='relu', padding="same")(embedding_layer)
    X2 = tf.keras.layers.GlobalMaxPooling1D()(X2)

    X3 = tf.keras.layers.Conv1D(200, 2, activation='relu', padding="same")(embedding_layer)
    X3 = tf.keras.layers.GlobalMaxPooling1D()(X3)

    X = tf.keras.layers.Concatenate(axis=1)([X1, X2, X3])
    X = tf.keras.layers.Dense(500, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(400, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)

    # Output layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(X)

    # Tạo model
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=output)
    return model

"""## Model for downstream task"""

if __name__ == '__main__':
    import sys

    # Loại bỏ các tham số không liên quan của Jupyter
    if "-f" in sys.argv:
        idx = sys.argv.index("-f")
        sys.argv = sys.argv[:1]  # Giữ lại chỉ tên file script

    import argparse
    parser = argparse.ArgumentParser(description="Fine-tuning script with BioBert and early stopping")
    parser.add_argument("--working_path", type=str, default="./BioBert", help="working path")
    parser.add_argument("--data_path", type=str, default="./BioBert/data/", help="data path")
    parser.add_argument("--model_name", type=str, default="dmis-lab/biobert-v1.1", help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--num_epoch", type=int, default=10, help="number of epochs")
    parser.add_argument("--features", type=str, default='TAKC', help="Which features")
    parser.add_argument("--max_len", type=int, default=512, help="Max length")
    parser.add_argument("--use_aim", action="store_true", help="Whether to use aim embeddings")
    parser.add_argument("--saved_folder", type=str,default='FPSRS', help="Whether to use aim embeddings")
    args = parser.parse_args()

    if args.use_aim:
        features_name = args.features + 'S'
    else:
        features_name = args.features


    # Run the main part of the script
    print(f"Using model: {args.model_name}")
    print(f"Using batch size: {args.batch_size}")
    print(f"Max Length: {args.max_len}")
    print(f"use_aim: {args.use_aim}")
    print(f"Features: {features_name}")

    """# Data preparation"""

    data_train = pd.read_csv(args.data_path + "train_set.csv", encoding="ISO-8859-1")
    data_validate = pd.read_csv(args.data_path + "val_set.csv", encoding="ISO-8859-1")
    data_test = pd.read_csv(args.data_path + "test_set.csv", encoding="ISO-8859-1")
    data_aims = pd.read_csv(args.data_path + "journal_category.csv", encoding="ISO-8859-1")

    data_train.fillna("", inplace=True)
    data_validate.fillna("", inplace=True)
    data_test.fillna("", inplace=True)
    data_aims.fillna("", inplace=True)

    num_class = len(data_aims)

    """## Feature selection"""
    print("Feature selection")

    # Create a dictionary to map category labels from data_aims
    category_mapping = dict(zip(data_aims['Label'], data_aims['Categories']))

    # Map the category values using the dictionary
    data_train['Categories'] = data_train['Label'].map(category_mapping)
    data_validate['Categories'] = data_validate['Label'].map(category_mapping)
    data_test['Categories'] = data_test['Label'].map(category_mapping)


    """## Tokenization"""
    print("Tokenization")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    MAX_LEN = args.max_len

    def data_generator(data):
      for i in range(data.shape[0]):
            if args.features == 'TAKC':
              sentence = data.iloc[i, 0] + " " + data.iloc[i, 1] + " " + data.iloc[i, 2]+ " " + data.iloc[i, 4] # Tổ hợp TAKC
            elif args.features == 'TAC':
              sentence = data.iloc[i, 0] + " " + data.iloc[i, 1] + " " + data.iloc[i, 4] # Tổ hợp TAC
            elif args.features == 'TKC':
              sentence = data.iloc[i, 0] + " " + data.iloc[i, 2]+ " " + data.iloc[i, 4] # Tổ hợp TKC
            elif args.features == 'AKC':
              sentence = data.iloc[i, 1] + " " + data.iloc[i, 2]+ " " + data.iloc[i, 4] # Tổ hợp AKC
            elif args.features == 'TC':
              sentence = data.iloc[i, 0] + " " + data.iloc[i, 4] # Tổ hợp TC
            elif args.features == 'AC':
              sentence = data.iloc[i, 1] + " " + data.iloc[i, 4] # Tổ hợp AC
            else:
              sentence = data.iloc[i, 2]+ " " + data.iloc[i, 4] # Tổ hợp KC
            y = data.iloc[i, 3]

            encoded = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )

            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]

            y_one_hot = np.zeros(num_class)
            y_one_hot[y] = 1

            yield (input_ids, attention_mask), y_one_hot

    batch_size = args.batch_size

    train_dataset = tf.data.Dataset.from_generator(
          lambda: data_generator(data_train),  # Use lambda to pass arguments
          output_types=((tf.int32, tf.int32), tf.float32),
          output_shapes=(((MAX_LEN,), (MAX_LEN,)), (num_class,))
      ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
          lambda: data_generator(data_validate),  # Use lambda to pass arguments
          output_types=((tf.int32, tf.int32), tf.float32),
          output_shapes=(((MAX_LEN,), (MAX_LEN,)), (num_class,))
      ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(
          lambda: data_generator(data_test),  # Use lambda to pass arguments
          output_types=((tf.int32, tf.int32), tf.float32),
          output_shapes=(((MAX_LEN,), (MAX_LEN,)), (num_class,))
      ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


    def Tokenizer_aims(sentences, tokenizer=tokenizer):
      input_ids, input_masks = [],[]
      for sentence in tqdm(sentences):
          inputs = tokenizer.encode_plus(
              sentence,
              add_special_tokens=True,
              max_length=250,
              padding='max_length',
              truncation=True,
              return_attention_mask=True)
          input_ids.append(inputs['input_ids'])
          input_masks.append(inputs['attention_mask'])
      print("xong")
      return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int8')

    Aims = data_aims["Aims"].tolist()
    aims_input_ids, aims_input_masks=Tokenizer_aims(Aims)

    print("Finish encoding data")
    """# Model definition"""
    """## Load fine-tuned LM"""

    # Choose Model based on use_aim
    if args.use_aim:
        model = create_model_WithAim(aims_input_ids_in=aims_input_ids, aims_input_masks_in=aims_input_masks,num_classes=num_class, model_name=args.model_name) # Pass aims data to the model
    else:
        model = create_model_NoAim(num_classes=num_class,model_name=args.model_name)
    #model.to(device) # Remove this line, as model is a Keras model

    """# Training


    """## Optimizer and Loss function"""

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=1e-08)

    # Loss function
    loss = tf.keras.losses.categorical_crossentropy

    """## Training settings"""

    max_epochs = args.num_epoch

    # Metric
    top3_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = "top3"
    top5_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = "top5"
    top10_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=10)
    top10_acc.__name__ = "top10"


    def get_latest_checkpoint(checkpoint_dir):
        checkpoint_sort = [f for f in os.listdir(checkpoint_dir) if f.endswith(".weights.h5")]
        if not checkpoint_sort:
            return None  # No checkpoints found

        def extract_epoch(filename):
            match = re.search(r'Epoch_(\d+)', filename)
            return int(match.group(1)) if match else -1  # Default to -1 if no epoch found

        sorted_checkpoints = sorted(checkpoint_sort, key=extract_epoch, reverse=True)
        latest_checkpoint = sorted_checkpoints[0]
        latest_epoch = extract_epoch(latest_checkpoint)
        return latest_checkpoint, latest_epoch if sorted_checkpoints else None


    checkpoint_dir = args.working_path +  args.saved_folder + "/" + features_name + "/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_name, initial_epoch = get_latest_checkpoint(checkpoint_dir)

    if checkpoint_name:
        print(f"Latest checkpoint found: {checkpoint_name}")
        try:
            model.load_weights(os.path.join(checkpoint_dir, checkpoint_name))
            print(f"Resuming training from epoch {initial_epoch +1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch.")
            initial_epoch = 0
    else:
        print("No checkpoints found. Starting training from scratch.")
        initial_epoch = 0

    # Create the result file
    result_file_path = os.path.join(checkpoint_dir, "results.txt")

    # Check if the results file exists
    if os.path.exists(result_file_path):
        # If it exists, open in append mode ('a')
        with open(result_file_path, "a") as f:
            f.write(f"Resuming training on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        # If it doesn't exist, open in write mode ('w') and write the headers
        with open(result_file_path, "w") as f:
            f.write(f"Training started on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Name: {args.model_name}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Features: {features_name}\n")
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Max Length: {args.max_len}\n")

            f.write(f"================================================\n")
            f.write(
                f"Epoch,Train Loss,Val Loss,Train Acc@1,Train Acc@3,Train Acc@5,Train Acc@10,Val Acc@1,Val Acc@3,Val Acc@5,Val Acc@10\n")


    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_dir, monitor='val_accuracy', mode='max', patience=3):
            super(CustomCallback, self).__init__()
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)
            self.epoch_start_time = None
            self.monitor = monitor
            self.mode = mode
            self.best_metric = None
            self.best_epoch = None
            self.patience = patience  # Số epoch chờ đợi trước khi early stopping
            self.wait = 0  # Số epoch đã chờ đợi

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - self.epoch_start_time

            # Kiểm tra và lưu best weights
            current_metric = logs.get(self.monitor)
            if current_metric is not None:
                if self.best_metric is None or (self.mode == 'max' and current_metric > self.best_metric) or \
                  (self.mode == 'min' and current_metric < self.best_metric):
                    self.best_metric = current_metric
                    self.best_epoch = epoch + 1
                    # Change the filename to include .weights.h5
                    self.model.save_weights(os.path.join(self.log_dir, 'best_weights.weights.h5'))
                    self.wait = 0  # Reset wait counter
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.model.stop_training = True  # Early stopping
                        print(f"Early stopping at epoch {epoch + 1}")
                        # Khôi phục best weights
                        self.model.load_weights(os.path.join(self.log_dir, 'best_weights.weights.h5'))
                        print(f"Restoring best weights from epoch {self.best_epoch}")

            # Lưu checkpoint
            self.model.save_weights(os.path.join(self.log_dir, f'Epoch_{epoch+1}.weights.h5'))

            # Ghi kết quả vào file txt
            with open(os.path.join(self.log_dir, 'results.txt'), 'a') as f:
                f.write(f'Epoch {epoch + 1}:\n')
                f.write(f'  Loss: {logs["loss"]:.4f}\n')
                f.write(f'  Time per epoch: {epoch_time:.2f} seconds\n')
                f.write(f'  Other metrics: {logs}\n\n')
                if self.best_epoch is not None:
                  f.write(f'  Best {self.monitor}: {self.best_metric:.4f} at epoch {self.best_epoch}\n\n')
    custom_callback = CustomCallback(checkpoint_dir)
    """## Model fit"""

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy", top3_acc, top5_acc, top10_acc])
    print(model.summary())

    history = model.fit(train_dataset,
                    epochs=args.num_epoch,
                    steps_per_epoch=len(data_train)//batch_size,
                    callbacks=[custom_callback],
                    verbose=1,
                    validation_data=val_dataset,
                    validation_steps=len(data_validate) // batch_size,  # Number of validation steps
                    initial_epoch=initial_epoch
                    )

    #Test
    model.load_weights(os.path.join(checkpoint_dir, 'best_weights.weights.h5'))
    loss, accuracy, top3, top5, top10= model.evaluate(test_dataset, verbose=1, steps=len(data_test) // batch_size)

    with open(os.path.join(checkpoint_dir, 'results.txt'), "a") as f:
      f.write(f"================================================\n")
      f.write(f"Testing Loss: {loss:.6f}\n") # Use loss directly
      f.write(f"Testing Acc@1: {accuracy:.4f}\n") # Use accuracy directly
      f.write(f"Testing Acc@3: {top3:.4f}\n") # Use top3 directly
      f.write(f"Testing Acc@5: {top5:.4f}\n") # Use top5 directly
      f.write(f"Testing Acc@10: {top10:.4f}\n") # Use top10 directly
      f.write(f"================================================\n")
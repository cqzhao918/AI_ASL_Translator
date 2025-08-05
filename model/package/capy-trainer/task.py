import pandas as pd
import os
import tqdm
import numpy as np
import json
import tensorflow as tf
import gc
from os import mkdir
import argparse

import math
import pickle

# import matplotlib.pyplot as plt
from keras import layers, models

# import tensorflow_addons as tfa
from google.cloud import storage
import io
import os
from tensorflow.python.lib.io import file_io

import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wandb_key", dest="wandb_key", default="16", type=str, help="WandB API Key"
)
args = parser.parse_args()

wandb.login(key=args.wandb_key)

client = storage.Client()
bucket = client.bucket("capy-data")

# blobs = bucket.list_blobs(prefix='data/preprocessed_dfs/preprocessed_dfs/')

# import wandb
# wandb.login()

FRAME_LEN = 128 * 2
# Read character to prediction index
blob_json = bucket.blob(
    "data/preprocessed_dfs/preprocessed_dfs/character_to_prediction_index.json"
)
with blob_json.open("r") as f:
    char_to_num = json.loads(f.read())

pad_token = "^"
pad_token_idx = 59

char_to_num[pad_token] = pad_token_idx
num_to_char = {j: i for i, j in char_to_num.items()}
INPUT_SHAPE = [256, 276]


def load_npy_data(npy_path, npy_file):
    blob_df = bucket.blob(os.path.join(npy_path, npy_file))
    npy_data = blob_df.download_as_bytes()
    loaded_array = np.load(io.BytesIO(npy_data), allow_pickle=True)
    return loaded_array


npyPath = "data/preprocessed_dfs/preprocessed_dfs/mean_std"
RHM = load_npy_data(npyPath, "rh_mean.npy")
LHM = load_npy_data(npyPath, "lh_mean.npy")
RPM = load_npy_data(npyPath, "rp_mean.npy")
LPM = load_npy_data(npyPath, "lp_mean.npy")
FACEM = load_npy_data(npyPath, "face_mean.npy")

RHS = load_npy_data(npyPath, "rh_std.npy")
LHS = load_npy_data(npyPath, "lh_std.npy")
RPS = load_npy_data(npyPath, "rp_std.npy")
LPS = load_npy_data(npyPath, "lp_std.npy")
FACES = load_npy_data(npyPath, "face_std.npy")


@tf.function()
def resize_pad(x):
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(
            x,
            ([[0, FRAME_LEN - tf.shape(x)[0]], [0, 0], [0, 0]]),
            constant_values=float("NaN"),
        )
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x


@tf.function()
def pre_process1(face, rhand, lhand, rpose, lpose):
    print(type(face), face.shape)
    face = (resize_pad(face) - FACEM) / FACES
    rhand = (resize_pad(rhand) - RHM) / RHS
    lhand = (resize_pad(lhand) - LHM) / LHS
    rpose = (resize_pad(rpose) - RPM) / RPS
    lpose = (resize_pad(lpose) - LPM) / LPS

    x = tf.concat([face, rhand, lhand, rpose, lpose], axis=1)
    s = tf.shape(x)
    x = tf.reshape(x, (s[0], s[1] * s[2]))
    x = tf.where(tf.math.is_nan(x), 0.0, x)
    return x


# Deconde tfrecords
def decode_fn(record_bytes):
    schema = {
        "face": tf.io.VarLenFeature(tf.float32),
        "rhand": tf.io.VarLenFeature(tf.float32),
        "lhand": tf.io.VarLenFeature(tf.float32),
        "rpose": tf.io.VarLenFeature(tf.float32),
        "lpose": tf.io.VarLenFeature(tf.float32),
        "phrase": tf.io.VarLenFeature(tf.int64),
    }
    x = tf.io.parse_single_example(record_bytes, schema)

    face = tf.reshape(tf.sparse.to_dense(x["face"]), (-1, 40, 3))
    rhand = tf.reshape(tf.sparse.to_dense(x["rhand"]), (-1, 21, 3))
    lhand = tf.reshape(tf.sparse.to_dense(x["lhand"]), (-1, 21, 3))
    rpose = tf.reshape(tf.sparse.to_dense(x["rpose"]), (-1, 5, 3))
    lpose = tf.reshape(tf.sparse.to_dense(x["lpose"]), (-1, 5, 3))
    phrase = tf.sparse.to_dense(x["phrase"])

    return face, rhand, lhand, rpose, lpose, phrase


def pre_process_fn(lip, rhand, lhand, rpose, lpose, phrase):
    phrase = tf.pad(
        phrase,
        [[0, MAX_PHRASE_LENGTH - tf.shape(phrase)[0]]],
        constant_values=pad_token_idx,
    )
    return pre_process1(lip, rhand, lhand, rpose, lpose), phrase


MAX_PHRASE_LENGTH = 500
tffiles_dir = [
    file.name
    for file in bucket.list_blobs(
        prefix="data/preprocessed_dfs/preprocessed_dfs/test_tfrecords"
    )
]
tffiles = [
    os.path.join("gs://capy-data", tffile)
    for tffile in tffiles_dir
    if ".tfrecord" in tffile
]
print("path", tffiles[:2])
# tffiles = [f"C:/Users/chuqi/ac215/capy_data_test/test_tfds/{file_id}.tfrecord" for file_id in os.listdir('C:/Users/chuqi/ac215/capy_data_test/test_npy')]
val_len = 1
train_batch_size = 32
val_batch_size = 32
#
# tffiles = ['gs://capy-data/data/preprocessed_dfs/preprocessed_dfs/test_tfrecords/preprocessed__fZbAxSSbX4_1-5-rgb_front.npy.tfrecord',
# 'gs://capy-data/data/preprocessed_dfs/preprocessed_dfs/test_tfrecords/preprocessed__fZbAxSSbX4_2-5-rgb_front.npy.tfrecord']

train_dataset = (
    tf.data.TFRecordDataset(tffiles[val_len:])
    .prefetch(tf.data.AUTOTUNE)
    .shuffle(5000)
    .map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .map(pre_process_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(train_batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
val_dataset = (
    tf.data.TFRecordDataset(tffiles[:val_len])
    .prefetch(tf.data.AUTOTUNE)
    .map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .map(pre_process_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(val_batch_size)
    .prefetch(tf.data.AUTOTUNE)
)


print("train:", train_dataset)
print("train type:", type(train_dataset))

print("val:", val_dataset)
print("val type:", type(val_dataset))

val = next(iter(val_dataset))
print(val[0].shape)

train = next(iter(train_dataset))
print(train[0].shape)


#%%Build model
class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(
            1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False
        )

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn


class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer="glorot_uniform",
        name="",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D(
            (dilation_rate * (kernel_size - 1), 0), name=name + "_pad"
        )
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size,
            strides=1,
            dilation_rate=dilation_rate,
            padding="valid",
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            name=name + "_dwconv",
        )
        self.supports_masking = True

    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x


def Conv1DBlock(
    channel_size,
    kernel_size,
    dilation_rate=1,
    drop_rate=0.0,
    expand_ratio=2,
    se_ratio=0.25,
    activation="swish",
    name=None,
):
    """
    efficient conv1d block, @hoyso48
    """
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + "_expand_conv",
        )(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(
            kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + "_dwconv",
        )(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + "_bn")(x)

        x = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size, use_bias=True, name=name + "_project_conv"
        )(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1), name=name + "_drop"
            )(x)

        if channels_in == channel_size:
            x = tf.keras.layers.add([x, skip], name=name + "_add")
        return x

    return apply


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim**-0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(
            tf.keras.layers.Reshape(
                (-1, self.num_heads, self.dim * 3 // self.num_heads)
            )(qkv)
        )
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(
            tf.keras.layers.Permute((2, 1, 3))(x)
        )
        x = self.proj(x)
        return x


def TransformerBlock(
    dim=256, num_heads=6, expand=4, attn_dropout=0.2, drop_rate=0.2, activation="swish"
):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(
            x
        )
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(
            x
        )
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x

    return apply


def positional_encoding(maxlen, num_hid):
    depth = num_hid / 2
    positions = tf.range(maxlen, dtype=tf.float32)[..., tf.newaxis]
    depths = tf.range(depth, dtype=tf.float32)[np.newaxis, :] / depth
    angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
    angle_rads = tf.linalg.matmul(positions, angle_rates)
    pos_encoding = tf.concat(
        [tf.math.sin(angle_rads), tf.math.cos(angle_rads)], axis=-1
    )
    return pos_encoding


def positional_encoding2(maxlen, num_hid):
    depth = num_hid / 2
    positions = tf.range(maxlen, dtype=tf.float32)[..., tf.newaxis]
    depths = tf.range(depth, dtype=tf.float32)[np.newaxis, :] / depth
    angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
    angle_rads = tf.linalg.matmul(positions, angle_rates)
    pos_encoding = np.zeros((maxlen, num_hid))
    pos_encoding[:, 0::2] = np.sin(angle_rads)
    pos_encoding[:, 1::2] = np.cos(angle_rads)
    return pos_encoding


# %% Build Loss function
def CTCLoss(labels, logits):
    label_length = tf.reduce_sum(tf.cast(labels != pad_token_idx, tf.int32), axis=-1)
    logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=pad_token_idx,
        logits_time_major=False,
    )
    loss = tf.reduce_mean(loss)
    return loss


def get_model(dim=384):
    inp = tf.keras.Input(INPUT_SHAPE)
    x = tf.keras.layers.Masking(mask_value=0.0)(inp)
    x = tf.keras.layers.Dense(dim, use_bias=False, name="stem_conv")(x)
    pe = tf.cast(positional_encoding(INPUT_SHAPE[0], dim), dtype=x.dtype)
    x = x + pe
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name="stem_bn")(x)
    num_blocks = 6
    drop_rate = 0.4
    for i in range(num_blocks):
        x = Conv1DBlock(dim, 11, drop_rate=drop_rate)(x)
        x = Conv1DBlock(dim, 5, drop_rate=drop_rate)(x)
        x = Conv1DBlock(dim, 3, drop_rate=drop_rate)(x)
        x = TransformerBlock(dim, expand=2)(x)

    x = tf.keras.layers.Dense(dim * 2, activation="relu", name="top_conv")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    # x = LateDropout(0.7)(x)
    x = tf.keras.layers.Dense(len(char_to_num), name="classifier")(x)

    model = tf.keras.Model(inp, x)

    loss = CTCLoss

    # Adam Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = tf.optimizers.RectifiedAdam(sma_threshold=4)
    # optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)

    model.compile(loss=loss, optimizer=optimizer)

    return model


tf.keras.backend.clear_session()
model = get_model()
# model(batch[0])


def num_to_char_fn(y):
    return [num_to_char.get(x, "") for x in y]


@tf.function()
def decode_phrase(pred):
    x = tf.argmax(pred, axis=1)
    diff = tf.not_equal(x[:-1], x[1:])
    adjacent_indices = tf.where(diff)[:, 0]
    x = tf.gather(x, adjacent_indices)
    mask = x != pad_token_idx
    x = tf.boolean_mask(x, mask, axis=0)
    return x


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    output_text = []
    for result in pred:
        result = "".join(num_to_char_fn(decode_phrase(result).numpy()))
        output_text.append(result)
    return output_text


# A callback class to output a few transcriptions during training
class CallbackEval(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        # model.save_weights("model.keras")
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = "".join(num_to_char_fn(label.numpy()))
                targets.append(label)
        print("-" * 100)
        # for i in np.random.randint(0, len(predictions), 2):
        print(f"Target    : {targets}")
        print(f"Prediction: {predictions}, len: {len(predictions)}")
        print("-" * 100)


#         for i in range(32):
#             print(f"Target    : {targets[i]}")
#             print(f"Prediction: {predictions[i]}, len: {len(predictions[i])}")
#             print("-" * 100)

# Callback function to check transcription on the val set.
validation_callback = CallbackEval(val_dataset.take(1))


N_EPOCHS = 51
N_WARMUP_EPOCHS = 10
LR_MAX = 1e-3
WD_RATIO = 0.05
WARMUP_METHOD = "exp"


def lrfn(
    current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=N_EPOCHS
):

    if current_step < num_warmup_steps:
        if WARMUP_METHOD == "log":
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        return (
            max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )
            * lr_max
        )


# Learning rate for encoder
LR_SCHEDULE = [
    lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.50)
    for step in range(N_EPOCHS)
]
# Learning Rate Callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(
    lambda step: LR_SCHEDULE[step], verbose=0
)

# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio=WD_RATIO):
        self.step_counter = 0
        self.wd_ratio = wd_ratio

    def on_epoch_begin(self, epoch, logs=None):
        model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
        print(
            f"learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}"
        )


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="code/baseline2/out2/model.{epoch:05d}.keras",
    save_weights_only=True,
    period=50,
)


# load weight
weight_path = (
    "gs://capy-data/data/preprocessed_dfs/preprocessed_dfs/model_weights/model.h5"
)
print(
    "model weight",
    [
        file.name
        for file in bucket.list_blobs(
            prefix="data/preprocessed_dfs/preprocessed_dfs/model_weights"
        )
    ],
)
blob = bucket.blob("data/preprocessed_dfs/preprocessed_dfs/model_weights/model.h5")
# model.load_weights(weight_path, by_name=False)

# Training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    #     epochs=N_EPOCHS,
    epochs=1,
    callbacks=[
        checkpoint_callback,
        validation_callback,
        lr_callback,
        WeightDecayCallback(),
    ],
)
model.save("basemodel.keras")

# Initialize a W&B run
wandb.init(
    project="capy-train",
    config={
        "learning_rate": LR_MAX,
        "epochs": 1,
        "batch_size": 32,
        "model_name": model.name,
    },
    name=model.name,
)

# Wandb training
training_results = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[WandbMetricsLogger()],
    # callbacks = [WandbMetricsLogger(log_freq=1)],
    verbose=1,
)

wandb.run.finish()

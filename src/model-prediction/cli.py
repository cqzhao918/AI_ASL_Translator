"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --test
"""

import os
import requests
import zipfile
import tarfile
import argparse
from glob import glob
import numpy as np
import base64
from google.cloud import storage
from google.cloud import aiplatform
import tensorflow as tf
import pyarrow.parquet as pq 

import pandas as pd
import json
from multiprocessing import cpu_count
from tensorflow.python.lib.io import file_io


client = storage.Client()
bucket = client.bucket('capy-data')

# blob_df = bucket.blob("data/WLASL-data/wlasl_test_new.csv")
# train_df = pd.read_csv(blob_df.open())

print("\n\n... LOAD SIGN TO PREDICTION INDEX MAP FROM JSON FILE ...\n")
# Read character to prediction index
blob_json = bucket.blob("data/WLASL-data/sign_to_prediction_index_map.json")
with blob_json.open("r") as f:
    json_file = json.loads(f.read())

s2p_map = {k.lower():v for k,v in json_file.items()}
p2s_map = {v:k for k,v in json_file.items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)
# print(s2p_map)
# train_df['label'] = train_df.sign.map(encoder)

ROWS_PER_FRAME = 543
MAX_LEN = 384
CROP_LEN = MAX_LEN
NUM_CLASSES  = 250
PAD = -100.
NOSE=[
    1,2,98,327
]
LNOSE = [98]
RNOSE = [327]
LIP = [ 0, 
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513,505,503,501]
RPOSE = [512,504,502,500]

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE #+POSE

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6*NUM_NODES

# print(NUM_NODES)
# print(CHANNELS)

def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis,  keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None,...]
        else:
            x = inputs
        
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5,x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2) #N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        
        x = (x - mean)/std

        if self.max_len is not None:
            x = x[:,:self.max_len]
        length = tf.shape(x)[1]
        x = x[...,:2]

        dx = tf.cond(tf.shape(x)[1]>1,lambda:tf.pad(x[:,1:] - x[:,:-1], [[0,0],[0,1],[0,0],[0,0]]),lambda:tf.zeros_like(x))

        dx2 = tf.cond(tf.shape(x)[1]>2,lambda:tf.pad(x[:,2:] - x[:,:-2], [[0,0],[0,2],[0,0],[0,0]]),lambda:tf.zeros_like(x))

        x = tf.concat([
            tf.reshape(x, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx2, (-1,length,2*len(self.point_landmarks))),
        ], axis = -1)
        
        x = tf.where(tf.math.is_nan(x),tf.constant(0.,x.dtype),x)
        
        return x

########Model#########
class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn

class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)
      
    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(self._train_counter < self.start_step, lambda:inputs, lambda:self.dropout(inputs, training=training))
        if training:
            self._train_counter.assign_add(1)
        return x

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, 
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer='glorot_uniform',
        name='', **kwargs):
        super().__init__(name=name,**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

def Conv1DBlock(channel_size,
          kernel_size,
          dilation_rate=1,
          drop_rate=0.0,
          expand_ratio=2,
          se_ratio=0.25,
          activation='swish',
          name=None):
    '''
    efficient conv1d block, @hoyso48
    '''
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
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim*expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

def get_model(max_len=MAX_LEN, dropout_step=0, dim=192):
    inp = tf.keras.Input((max_len,CHANNELS))
    #x = tf.keras.layers.Masking(mask_value=PAD,input_shape=(max_len,CHANNELS))(inp) #we don't need masking layer with inference
    x = inp
    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    if dim == 384: #for the 4x sized model
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

    x = tf.keras.layers.Dense(dim*2,activation=None,name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES,name='classifier')(x)
    return tf.keras.Model(inp, x)

models_path = [
              'gs://capy-data/data/model_weights/islr-fp16-192-8-seed42-foldall-full.h5', #comment out other weights to check single model score
            #    'C:/Users/chuqi/ac215/kaggle-data/aslfr-isolated/islr-fp16-192-8-seed42-foldall-last.h5',
#                '/kaggle/input/islr-models/islr-fp16-192-8-seed44-foldall-last.h5',
               #'/kaggle/input/islr-models/islr-fp16-192-8-seed45-foldall-last.h5',
              ]
models_path_local = []
for model_path in models_path: 
    model_file = file_io.FileIO(model_path, mode='rb')
    model_name = model_path.split('/')[-1]
    temp_model_location = f'./{model_name}'
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(model_file.read())
    temp_model_file.close()
    model_file.close()
    models_path_local.append(model_name)

models = [get_model() for _ in models_path_local]
              
# models = [get_model() for _ in models_path]
for model,path in zip(models,models_path_local):
    model.load_weights(path)
# models[0].summary()

def load_target(wlasl_train_df, file_data_name):
    parquet_id = file_data_name.split("/")[-1]
    path = 'wlasl_parquet_deploy/'+parquet_id
#     print(path)
#     print(wlasl_train_df[wlasl_train_df['path']==path])
    print(wlasl_train_df)
    target = wlasl_train_df[wlasl_train_df['path']==path]['sign'].iloc[0]
    return target


def load_relevant_data_subset(file_data_name):
    data_columns = ['x', 'y', 'z']
#     data = pd.read_parquet('C:/Users/chuqi/ac215/kaggle-data/aslfr-isolated/islr-5fold' + pq_path, columns=data_columns)
#     file_data_name = [pq_path + file_name for file_name in os.listdir(pq_path)]
    # print("File_name", file_data_name)
    # blob_df = bucket.blob(file_data_name)
    # data = pd.read_parquet(blob_df, columns=data_columns)
    table = pq.read_table(file_data_name)
    data = table.to_pandas()[data_columns]
    # print(data)
    
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    # print(data.shape)
    return data.astype(np.float32)

def prune_model(model, sparisity=1):
#     model = models[1]
    layer_with_weights = []
    lw = []
    for layer in model.layers:
        if bool(layer.trainable_weights):
            layer_with_weights.append(layer.get_weights())
            lw.append(layer.name)

    w = layer_with_weights
    w_flat = []
    #masks_flat = []
    for item in w:
        for elem in item:
            for x in list(elem.flatten()):
                w_flat.append(x)
    w_flat_abs = [abs(x) for x in w_flat]
    prune_length = int(len(w_flat_abs)*sparisity)
    w_flat_abs_prune = sorted(w_flat_abs)[:prune_length][-1]

    weights_pruned = []
    for item in w:
        wp = []
        for i in range(len(item)):
            pruned = np.where(np.abs(item[i])<=w_flat_abs_prune, 0 ,item[i])
            wp.append(pruned)
        weights_pruned.append(wp)
        #w_pruned = np.where(np.abs(w[i])<= w_flat_abs_prune, 0, w[i])
       #weights_pruned.append(w_pruned)

    wd = dict(zip(lw,weights_pruned))
    for layer in model.layers:
        if bool(layer.trainable_weights):
            layer.set_weights(wd[layer.name])
            
    return model

class TFModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – a preprocessing model
        – the ISLR model 
    """
    def __init__(self, islr_models):
        """
        Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
        """
        super(TFModel, self).__init__()

        # Load the feature generation and main models
        self.prep_inputs = Preprocess()
        self.islr_models   = islr_models
        
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = self.prep_inputs(inputs)
#         x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = [model(x) for model in self.islr_models]
        outputs = tf.keras.layers.Average()(outputs)[0]
        return {'outputs': outputs}

class TFLiteModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – a preprocessing model
        – the ISLR model 
    """

    def __init__(self, islr_models):
        """
        Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
        """
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.prep_inputs = Preprocess()
        self.islr_models   = islr_models
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = [model(x) for model in self.islr_models]
        outputs = tf.keras.layers.Average()(outputs)[0]
        return {'outputs': outputs}


def prediction(sparisity=1, QUANT=False, PRUNE=False):
    TARGET, PRED = [], []
    
    # Load target csv
    # blob_df = bucket.blob("data/WLASL-data/wlasl_test_new.csv")
    # wlasl_test_df = pd.read_csv(blob_df.open())
    # wlasl_test_df = pd.read_csv(r"./wlasl/wlasl_test_new.csv")
    
    # Load Model
    model =  models[0]
    tflite_keras_model = TFModel(islr_models=models)
    if QUANT == True and PRUNE==False:
        tflite_keras_model = TFLiteModel(islr_models=models)
    elif QUANT == True and PRUNE==True:
        pruned_model = prune_model(model, sparisity)
        tflite_keras_model = TFLiteModel(islr_models=[pruned_model])
    
    # Save Model
    keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
    keras_model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    keras_model_converter.target_spec.supported_types = [tf.float16]
    tflite_model = keras_model_converter.convert()
    
    
    # Load all files
    tffiles_dir = [file.name for file in bucket.list_blobs(prefix='data/WLASL-data/wlasl_parquet_deploy') if '.parquet' in file.name]
    tffiles = [os.path.join('gs://capy-data',tffile) for tffile in tffiles_dir if '.parquet' in tffile]

    pq_path = tffiles
    # file_data_names = [pq_path + file_name for file_name in os.listdir(pq_path)]
    file_data_names = tffiles
    for file_data_name in file_data_names:
        # Transformer Prediction 
#         if QUANT == True:
        #print(file_data_name)
        demo_output = tflite_keras_model(load_relevant_data_subset(file_data_name))["outputs"]
        pred_value = decoder(np.argmax(demo_output.numpy(), axis=-1))
#         else:
#             demo_output = tflite_keras_model(load_relevant_data_subset(file_data_name))
#             pred_value = decoder(np.argmax(demo_output.numpy(), axis=-1))
        
        # Target
        # target = load_target(wlasl_test_df, file_data_name)
    
        # print(f"Target: {target}")
        # print(f"Prediction: {pred_value}")
        # print("-"*20)
        
        # TARGET.append(target)
        PRED.append(pred_value)
    return PRED

def decoder_accuracy(TARGET,PRED):
    correct_predition_count = sum(1 for true, pred in zip(TARGET, PRED) if true == pred)
    return correct_predition_count/len(TARGET)


def main(args=None):
    if args.test:
        print("Making Prediction: ")

        PRED = prediction(sparisity=1, QUANT=False, PRUNE=False)
        # accuracy = decoder_accuracy(TARGET,PRED)
        print(f"Prediction: {PRED}%")


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Model Inference CLI")

    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Make inference on test set",
    )
    
    args = parser.parse_args()

    main(args)

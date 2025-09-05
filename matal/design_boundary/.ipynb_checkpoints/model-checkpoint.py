from datetime import datetime
import pickle
import json
import copy

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Activation, BatchNormalization, GaussianNoise
from tensorflow.keras import activations, regularizers
from tensorflow.keras.utils import plot_model
from tqdm.keras import TqdmCallback

import matplotlib as mpl
import matplotlib.pyplot as plt

from matal.utils import auto_log, get_sid, obj_to_hash, RobustJSONEncoder
from matal.configs import BATCH_DIR, PROJ_CODE, MODEL_DIR


def get_activation_layer(a, **kwargs):
    if a.lower in ['leaky_relu', 'leakyrelu']:
        return LeakyReLU(**kwargs)
    else:
        return Activation(tf.keras.activations.get(a), **kwargs)


def get_encoder(in_model, n_bottleneck=2, name='encoder', input_noise=0.0, layer_scales=[2, 1, 0.5], 
                l1_coeff: float = 0.0, l2_coeff: float = 0.0, kernel_l1_coeff: float = None, kernel_l2_coeff: float = None,
                bias_l1_coeff: float = None, bias_l2_coeff: float = None,
                batch_norm=True, activation='leaky_relu', out_activation='leaky_relu'):
    enc = in_model
    n_inputs = in_model.shape.as_list()[1]
    
    enc = GaussianNoise(input_noise, name=f'{name}_gn')(enc)
    
    kernel_l1_coeff = l1_coeff if kernel_l1_coeff is None else kernel_l1_coeff
    kernel_l2_coeff = l2_coeff if kernel_l2_coeff is None else kernel_l2_coeff
    bias_l1_coeff = l1_coeff if bias_l1_coeff is None else bias_l1_coeff
    bias_l2_coeff = l2_coeff if bias_l2_coeff is None else bias_l2_coeff
    
    def get_regularizer_kw():
        return dict(kernel_regularizer=regularizers.L1L2(l1=kernel_l1_coeff, l2=kernel_l2_coeff),
                    bias_regularizer=regularizers.L1L2(l1=bias_l1_coeff, l2=bias_l2_coeff))

    for li, ls in enumerate(layer_scales, 1):
        enc = Dense(round(n_inputs * ls), name=f'{name}_l{li}',
                    **get_regularizer_kw(),
                   )(enc)
        if batch_norm:
            enc = BatchNormalization(name=f'{name}_l{li}_bn')(enc)
        if activation is not None:
            enc = get_activation_layer(activation, name=f'{name}_l{li}_ac')(enc)
    
    if out_activation is None:
        enc = Dense(n_bottleneck, name=f'{name}_out', **get_regularizer_kw())(enc)
    else:
        enc = Dense(n_bottleneck, name=f'{name}_out_preact', **get_regularizer_kw())(enc)
        enc = get_activation_layer(out_activation, name=f'{name}_l{li}_out')(enc)
        
    return enc

def get_classifier(in_model, n_outputs=1, name='classifier', output_name='classifier_output', layer_scales=[4, ], 
                   l1_coeff: float = 0.0, l2_coeff: float = 0.0, kernel_l1_coeff: float = None, kernel_l2_coeff: float = None,
                   bias_l1_coeff: float = None, bias_l2_coeff: float = None,
                   batch_norm=True, activation='leaky_relu', out_activation='leaky_relu'):
    est = in_model
    
    kernel_l1_coeff = l1_coeff if kernel_l1_coeff is None else kernel_l1_coeff
    kernel_l2_coeff = l2_coeff if kernel_l2_coeff is None else kernel_l2_coeff
    bias_l1_coeff = l1_coeff if bias_l1_coeff is None else bias_l1_coeff
    bias_l2_coeff = l2_coeff if bias_l2_coeff is None else bias_l2_coeff
    
    def get_regularizer_kw():
        return dict(kernel_regularizer=regularizers.L1L2(l1=kernel_l1_coeff, l2=kernel_l2_coeff),
                    bias_regularizer=regularizers.L1L2(l1=bias_l1_coeff, l2=bias_l2_coeff))
    
    for li, ls in enumerate(layer_scales, 1):
        est = Dense(round(n_outputs * ls), name=f'{name}_l{li}', **get_regularizer_kw())(est)
        if batch_norm:
            est = BatchNormalization(name=f'{name}_l{li}_bn')(est)
        if activation is not None:
            est = get_activation_layer(activation, name=f'{name}_l{li}_ac')(est)
    
    if out_activation is None:
        est = Dense(n_outputs, name=output_name, **get_regularizer_kw())(est)
    else:
        est = Dense(n_outputs, name=f'{name}_out_preact', **get_regularizer_kw())(est)
        est = get_activation_layer(out_activation, name=output_name)(est)

    return est


def build_model(params):
    inp = Input(shape=(params['n_inputs'],), name=f'input')
    
    encoder = get_encoder(inp, name='enc', **params['encoder'])
    outputs = [get_classifier(encoder, name='cls_' + y, **p) for y, p in params['classfiers'].items()]
    model = Model(inputs=inp, outputs=outputs, name=params['model_name'])
    model.compile(loss=tf.keras.losses.get(params['loss']), 
                  optimizer=tf.keras.optimizers.get(params['optimizer']).__class__(**params['optimizer_kwargs']), 
                  metrics=params['metrics'])
    
    return model

def save_model(model, history=None, history_filter_str='_accuracy', sep='__',
               params_hash=None, params=None, data_tag=None, prefix=None):
    model_prefix = prefix if prefix else ''
    model_prefix += datetime.now().strftime('%y%m%d')
    
    if params_hash:
        model_prefix += (sep + params_hash)
    if data_tag:
        model_prefix += (sep + data_tag)
        
    model.save(MODEL_DIR / (model_prefix + '.keras'))
    model.save(MODEL_DIR / (model_prefix + '.keras_dir'))
    
    if history:
        with open(MODEL_DIR / (model_prefix + '.hist.json'), 'w') as f:
            json.dump(history.history, f, indent=2, cls=RobustJSONEncoder)
            
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for k, v in history.history.items():
            if history_filter_str is not None and history_filter_str not in k: continue
            ax.plot(history.history[k], label=k)
        ax.legend()
        fig.savefig(MODEL_DIR / (model_prefix + '.hist.pdf'))
        plt.close(fig)
        
    if params:
        with open(MODEL_DIR / (model_prefix + '.params.json'), 'w') as f:
            json.dump(params, f, indent=2)
    
    return model_prefix

def load_model(model_prefix):
    try:
        model = keras.models.load_model(MODEL_DIR / f'{model_prefix}.keras_dir')
    except:
        try:
            model = keras.models.load_model(MODEL_DIR / f'{model_prefix}.keras')
        except:
            model = None
    
    return model
    
    
class KerasCombinedModel:
    def __init__(self):
        self.enc_layers = []
        self.cls_layers = {}
    
    def load_keras_model(self, model, enc_prefix='enc_', cls_prefix='cls_', input_name='input', sep='_'):
        for l in model.layers:
            if l.name == input_name or l.name.startswith(enc_prefix):
                self.enc_layers.append(l)
            elif l.name.startswith(cls_prefix):
                name_segs = l.name.split(sep)
                cls_name = name_segs[1]
                if cls_name in self.cls_layers.keys():
                    self.cls_layers[cls_name].append(l)
                else:
                    self.cls_layers[cls_name] = [l]
            elif l.name in self.cls_layers.keys():
                self.cls_layers[l.name].append(l)
            else:
                print(f'Unknown layer:', l.name, l)
        self.cls_names = list(self.cls_layers.keys())
        
        return self
    
    def predict(self, X, cls_name, encode_first=True):
        if encode_first:
            X = self.encode(X)
        else:
            X = np.array(X)
        for l in self.cls_layers[cls_name]:
            X = l(X)
        return np.array(X)
    
    def encode(self, X):
        X = np.array(X)
        for l in self.enc_layers:
            X = l(X)
        return np.array(X)

    @staticmethod
    def _encode_keras_layers(layers):
        tups = [(layer.__class__, layer.get_config(), layer.get_weights()) for layer in layers]
        return tups
    
    @staticmethod
    def _decode_keras_layers(tups, prev_layer=None):
        layers = []
        for tup in tups:
            keras_class, config, weights = tup
            auto_log(f'Class: {keras_class.__name__}, config: {config}, weights: {[tf.shape(a).numpy().tolist() for a in weights]}')
            layer = keras_class(**config)
            if prev_layer:
                # layer.build(prev_layer.output_shape)
                layer([np.ones(prev_layer.output_shape[1:])])
            layer.set_weights(weights)
            layers.append(layer)
            prev_layer = layer
        return layers
    
    def __getstate__(self):
        d = copy.copy(self.__dict__)
        
        d['__saved_keras__enc_layers'] = self._encode_keras_layers(self.enc_layers)
        d['__saved_keras__cls_layers'] = {k: self._encode_keras_layers(layers) for k, layers in self.cls_layers.items()}
        del d['enc_layers']
        del d['cls_layers']
        
        return d

    def __setstate__(self, d):
        for k, v in d.items():
            if k == '__saved_keras__enc_layers':
                self.enc_layers = self._decode_keras_layers(v)
            
            elif k == '__saved_keras__cls_layers':
                self.cls_layers = {cls_name: self._decode_keras_layers(layers, prev_layer=self.enc_layers[-1]) for cls_name, layers in v}
            
            else:
                setattr(self, k, v)
    
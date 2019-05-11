#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'wf'

import nltk
import re
from urllib.parse import unquote
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
def GeneSeg(payload):
    payload=payload.lower()#变小写
    payload=unquote(unquote(payload))#解码
    payload,num=re.subn(r'\d+',"0",payload)#数字泛化为"0"
    #替换url为”http://u
    payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)
def init_session():
    # 设置按需使用GPU
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.operation_timeout_in_ms = -1
    # config.operation_timeout_in_ms = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)  # 最多占gpu资源的70%
    config.gpu_options.allow_growth = True
    ktf.set_session(tf.InteractiveSession(config=config))#创建一个会话，当上下文管理器退出时会话关闭和资源释放自动完成。
def close_session():
    ktf.clear_session()
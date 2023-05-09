#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

"""
@author: nl8590687
若干声学模型模型的定义
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization, LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Lambda, Activation, Conv2D, MaxPooling2D, TimeDistributed
from tensorflow.keras import backend as K
import numpy as np
from utils.ops import ctc_decode_delete_tail_blank

class BaseModel:
    """
    定义声学模型类型的接口基类
    """

    def __init__(self):
        # 输入数据形状
        self.input_shape = None
        # 输出数据形状
        self.output_shape = None
        # 模型
        self.model = None
        # 模型基类
        self.model_base = None
        # 模型名称
        self._model_name = None

    # 返回模型和模型基类的元组
    def get_model(self) -> tuple:
        return self.model, self.model_base

    # 返回用于训练的模型
    def get_train_model(self) -> Model:
        return self.model

    # 返回用于评估的模型
    def get_eval_model(self) -> Model:
        return self.model_base

    # 打印模型的摘要信息
    def summary(self) -> None:
        self.model.summary()

    # 返回模型名称
    def get_model_name(self) -> str:
        return self._model_name

    # 加载模型权重
    def load_weights(self, filename: str) -> None:
        self.model.load_weights(filename)

    # 保存模型权重
    def save_weights(self, filename: str) -> None:
        # 保存模型
        self.model.save_weights(filename + '.model.h5')
        # 保存模型基类
        self.model_base.save_weights(filename + '.model.base.h5')

        # 将模型名称写入文件
        f = open('epoch_'+self._model_name+'.txt', 'w')
        f.write(filename)
        f.close()
        
    # 获取损失函数
    def get_loss_function(self):
        raise Exception("method not implemented")
    # 模型前向传递
    def forward(self, x):
        raise Exception("method not implemented")


def ctc_lambda_func(args):
    """
    定义CTC loss函数
    :param args: 元组，包含y_pred(预测的输出序列)、labels(标签序列)、input_length(输入序列长度)和label_length(标签序列长度)
    """
    y_pred, labels, input_length, label_length = args
    # 将y_pred转换成(batch_size, time_steps, num_classes)的形状
    y_pred = y_pred[:, :, :]
    # 计算CTC loss
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class SpeechModel251BN(BaseModel):
    """
    定义CNN+CTC模型，使用函数式模型
    
    简要：

    输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）\\
    隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2 \\
    隐藏层：全连接层 \\
    输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数， \\
    CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

    参数： \\
        input_shape: tuple，默认值(1600, 200, 1) \\
        output_shape: tuple，默认值(200, 1428)
    """
    def __init__(self, input_shape: tuple = (1600, 200, 1), output_size: int = 1428) -> None:
        # 父类初始化
        super().__init__()
        # 将输入形状保存到类中
        self.input_shape = input_shape
        # 定义池化大小
        self._pool_size = 8
        # 根据池化大小计算输出形状
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        # 定义模型名称
        self._model_name = 'SpeechModel251bn'
        # 调用 _define_model 方法，返回构建的模型和基础模型
        self.model, self.model_base = self._define_model(self.input_shape, self.output_shape[1])

    def _define_model(self, input_shape, output_size) -> tuple:
        # 预测的文本序列中字符的最大数量
        label_max_string_length = 64

        # 参数输入层
        input_data = Input(name='the_input', shape=input_shape)

        # 32个卷积核，大小为3×3， use_bias 是否使用偏置项，padding='same' 保留边界处的卷积结果，kernel_initializer='he_normal' 表示使用He正态分布来初始化卷积核的权重
        layer_h = Conv2D(32, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv0')(input_data)  # 卷积层
        # epsilon 用于防止除0错误
        layer_h = BatchNormalization(epsilon=0.0002, name='BN0')(layer_h) # 规范层
        # 即使函数输出为 0 或更大，而小于输入时，将输入保持不变。
        # ReLU 是一种常用的激活函数，在深度学习中经常被用作非线性函数，因为它具有许多好的性质，如简单性、易于计算和避免梯度消失等。该激活函数对于许多问题都表现良好，因此在卷积神经网络中经常被使用。
        layer_h = Activation('relu', name='Act0')(layer_h)

        layer_h = Conv2D(32, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv1')(layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN1')(layer_h)
        layer_h = Activation('relu', name='Act1')(layer_h)

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h) # 池化层

        layer_h = Conv2D(64, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv2')(layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN2')(layer_h)
        layer_h = Activation('relu', name='Act2')(layer_h)

        layer_h = Conv2D(64, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv3')(layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN3')(layer_h)
        layer_h = Activation('relu', name='Act3')(layer_h)

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h) # 池化层

        layer_h = Conv2D(128, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv4')(layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN4')(layer_h)
        layer_h = Activation('relu', name='Act4')(layer_h)

        layer_h = Conv2D(128, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv5')(layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN5')(layer_h)
        layer_h = Activation('relu', name='Act5')(layer_h)

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h) # 池化层

        layer_h = Conv2D(128, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv6')(layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN6')(layer_h)
        layer_h = Activation('relu', name='Act6')(layer_h)

        layer_h = Conv2D(128, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv7')(layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN7')(layer_h)
        layer_h = Activation('relu', name='Act7')(layer_h)

        layer_h = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h) # 池化层

        layer_h = Conv2D(128, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv8')(layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN8')(layer_h)
        layer_h = Activation('relu', name='Act8')(layer_h)

        layer_h = Conv2D(128, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv9')(layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN9')(layer_h)
        layer_h = Activation('relu', name='Act9')(layer_h)

        layer_h = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h) # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()

        layer_h = Reshape((self.output_shape[0], input_shape[1] // self._pool_size * 128), name='Reshape0')(layer_h)  # Reshape层

        layer_h = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal', name='Dense0')(layer_h)  # 全连接层

        layer_h = Dense(output_size, use_bias=True, kernel_initializer='he_normal', name='Dense1')(layer_h) # 全连接层
        y_pred = Activation('softmax', name='Activation0')(layer_h)

        model_base = Model(inputs = input_data, outputs = y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        return model, model_base

    def get_loss_function(self) -> dict:
        return {'ctc': lambda y_true, y_pred: y_pred}

    def forward(self, data_input):
        batch_size = 1 
        in_len = np.zeros((batch_size,), dtype=np.int32)

        in_len[0] = self.output_shape[0]

        x_in = np.zeros((batch_size,) + self.input_shape, dtype=np.float64)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.model_base.predict(x=x_in)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        if tf.__version__[0:2] == '1.':
            r1 = r[0][0].eval(session=tf.compat.v1.Session())
        else:
            r1 = r[0][0].numpy()
        
        speech_result = ctc_decode_delete_tail_blank(r1[0])
        return speech_result


class SpeechModel251(BaseModel):
    """
    定义CNN+CTC模型，使用函数式模型

    输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）\\
    隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2 \\
    隐藏层：全连接层 \\
    输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数， \\
    CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

    参数： \\
        input_shape: tuple，默认值(1600, 200, 1) \\
        output_shape: tuple，默认值(200, 1428)
    """
    def __init__(self, input_shape: tuple = (1600, 200, 1), output_size: int = 1428) -> None:
        super().__init__()
        self.input_shape = input_shape
        self._pool_size = 8
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        self._model_name = 'SpeechModel251'
        self.model, self.model_base = self._define_model(self.input_shape, self.output_shape[1])

    def _define_model(self, input_shape, output_size) -> tuple:
        label_max_string_length = 64

        input_data = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        layer_h3 = Dropout(0.05)(layer_h3)  # 随机中断部分神经网络连接，防止过拟合

        layer_h4 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)  # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5) # 池化层
        layer_h6 = Dropout(0.1)(layer_h6)
        
        layer_h7 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8) # 池化层
        layer_h9 = Dropout(0.15)(layer_h9)
        
        layer_h10 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9)  # 卷积层
        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11) # 池化层

        layer_h12 = Dropout(0.2)(layer_h12)
        layer_h13 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.2)(layer_h13)
        layer_h14 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h13)  # 卷积层
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14) # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()

        layer_h16 = Reshape((self.output_shape[0], input_shape[1] // self._pool_size * 128))(layer_h15)  # Reshape层
        layer_h16 = Dropout(0.3)(layer_h16)  # 随机中断部分神经网络连接，防止过拟合
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(output_size, use_bias=True, kernel_initializer='he_normal')(layer_h17)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(layer_h18)

        model_base = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        return model, model_base

    def get_loss_function(self) -> dict:
        return {'ctc': lambda y_true, y_pred: y_pred}

    def forward(self, data_input):
        batch_size = 1 
        in_len = np.zeros((batch_size,), dtype=np.int32)

        in_len[0] = self.output_shape[0]

        x_in = np.zeros((batch_size,) + self.input_shape, dtype=np.float64)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.model_base.predict(x = x_in)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        if tf.__version__[0:2] == '1.':
            r1 = r[0][0].eval(session=tf.compat.v1.Session())
        else:
            r1 = r[0][0].numpy()
        
        speech_result = ctc_decode_delete_tail_blank(r1[0])
        return speech_result


class SpeechModel25(BaseModel):
    """
    定义CNN+CTC模型，使用函数式模型

    输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）\\
    隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2 \\
    隐藏层：全连接层 \\
    输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数， \\
    CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

    参数： \\
        input_shape: tuple，默认值(1600, 200, 1) \\
        output_shape: tuple，默认值(200, 1428)
    """
    def __init__(self, input_shape: tuple = (1600, 200, 1), output_size: int = 1428) -> None:
        super().__init__()
        self.input_shape = input_shape
        self._pool_size = 8
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        self._model_name = 'SpeechModel25'
        self.model, self.model_base = self._define_model(self.input_shape, self.output_shape[1])

    def _define_model(self, input_shape, output_size) -> tuple:
        label_max_string_length = 64

        input_data = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        layer_h3 = Dropout(0.05)(layer_h3)  # 随机中断部分神经网络连接，防止过拟合

        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)  # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5) # 池化层
        layer_h6 = Dropout(0.1)(layer_h6)
        
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层
        layer_h9 = Dropout(0.15)(layer_h9)
        
        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9)  # 卷积层
        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11) # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()

        layer_h12 = Reshape((self.output_shape[0], input_shape[1] // self._pool_size * 128))(layer_h12)  # Reshape层
        layer_h12 = Dropout(0.3)(layer_h12)  # 随机中断部分神经网络连接，防止过拟合
        layer_h13 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h12)  # 全连接层
        layer_h13 = Dropout(0.3)(layer_h13)
        layer_h14 = Dense(output_size, use_bias=True, kernel_initializer='he_normal')(layer_h13)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(layer_h14)

        model_base = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        return model, model_base

    def get_loss_function(self) -> dict:
        return {'ctc': lambda y_true, y_pred: y_pred}

    def forward(self, data_input):
        batch_size = 1 
        in_len = np.zeros((batch_size,), dtype=np.int32)

        in_len[0] = self.output_shape[0]

        x_in = np.zeros((batch_size,) + self.input_shape, dtype=np.float64)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.model_base.predict(x=x_in)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        if tf.__version__[0:2] == '1.':
            r1 = r[0][0].eval(session=tf.compat.v1.Session())
        else:
            r1 = r[0][0].numpy()
        
        speech_result = ctc_decode_delete_tail_blank(r1[0])
        return speech_result


class SpeechModel24(BaseModel):
    """
    定义CNN+CTC模型，使用函数式模型

    输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）\\
    隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2 \\
    隐藏层：全连接层 \\
    输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数， \\
    CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

    参数： \\
        input_shape: tuple，默认值(1600, 200, 1) \\
        output_shape: tuple，默认值(200, 1428)
    """
    def __init__(self, input_shape :tuple=(1600, 200, 1), output_size: int = 1428) -> None:
        super().__init__()
        self.input_shape = input_shape
        self._pool_size = 8
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        self._model_name = 'SpeechModel24'
        self.model, self.model_base = self._define_model(self.input_shape, self.output_shape[1])

    def _define_model(self, input_shape, output_size) -> tuple:
        label_max_string_length = 64

        input_data = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.1)(layer_h1)
        layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        layer_h3 = Dropout(0.2)(layer_h3)  # 随机中断部分神经网络连接，防止过拟合

        layer_h4 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)  # 卷积层
        layer_h4 = Dropout(0.2)(layer_h4)
        layer_h5 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5) # 池化层

        layer_h6 = Dropout(0.3)(layer_h6)
        layer_h7 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.3)(layer_h7)
        layer_h8 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8) # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()
        # output_shape = (input_shape[0] // self._pool_size, output_size)
        # layer_h10 = Reshape((output_shape[0], input_shape[1] // self._pool_size * 128))(layer_h9)  # Reshape层
        layer_h10 = Reshape((self.output_shape[0], input_shape[1] // self._pool_size * 128))(layer_h9)  # Reshape层
        layer_h10 = Dropout(0.3)(layer_h10)  # 随机中断部分神经网络连接，防止过拟合
        layer_h11 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h10) # 全连接层
        layer_h11 = Dropout(0.3)(layer_h11)
        layer_h12 = Dense(output_size, use_bias=True, kernel_initializer='he_normal')(layer_h11) # 全连接层
        y_pred = Activation('softmax', name='Activation0')(layer_h12)

        model_base = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        return model, model_base

    def get_loss_function(self) -> dict:
        return {'ctc': lambda y_true, y_pred: y_pred}

    def forward(self, data_input):
        batch_size = 1 
        in_len = np.zeros((batch_size,), dtype=np.int32)

        in_len[0] = self.output_shape[0]

        x_in = np.zeros((batch_size,) + self.input_shape, dtype=np.float64)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.model_base.predict(x=x_in)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        if tf.__version__[0:2] == '1.':
            r1 = r[0][0].eval(session=tf.compat.v1.Session())
        else:
            r1 = r[0][0].numpy()
        
        speech_result = ctc_decode_delete_tail_blank(r1[0])
        return speech_result

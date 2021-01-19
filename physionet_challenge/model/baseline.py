#!/usr/bin/env python

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend as K


def focal_loss(gamma=2., alpha=.25, eps=1e-5):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = K.clip(tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred)), eps, 1-eps)
		pt_0 = K.clip(tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred)), eps, 1-eps)

		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

	return focal_loss_fixed


def BaselineMultibranch(n_inputs=146, n_classes=111):
    inp = Input((n_inputs,))
    dense_1 = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))

    dense_2_output = dense_1(inp)
    branches = []
    for i in range(n_classes):
        branch_1 = Dense(32, activation="relu")
        dropout = Dropout(.2)
        branch_2 = Dense(1, activation="sigmoid")
        branches.append(branch_2(dropout(branch_1(dense_2_output))))
    output = Concatenate()(branches)

    neural_model = Model(inputs=inp, outputs=output)
    neural_model.compile(
        optimizer=Adam(lr=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return neural_model

def BaselineMultibranchBig(n_inputs=146, n_classes=111):
    inp = Input((n_inputs,))
    dropout = Dropout(.2)
    dense_1 = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))
    dense_2 = Dense(32, activation="relu", kernel_regularizer=l2(1e-4))
    dense_1_output = dense_1(dropout(inp))
    dense_2_output = dense_2(dense_1_output)

    branches = []
    for i in range(n_classes):
        branch_1 = Dense(16, activation="relu")
        branch_2 = Dense(8, activation="relu")
        branch_3 = Dense(1, activation="sigmoid")
        branches.append(branch_3(branch_2(branch_1(dense_2_output))))
    output = Concatenate()(branches)

    neural_model = Model(inputs=inp, outputs=output)
    neural_model.compile(
        optimizer=Adam(lr=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return neural_model

def BaselineMultibranchFocal(n_inputs=146, n_classes=111):
    inp = Input((n_inputs,))
    dense_1 = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))

    dense_2_output = dense_1(inp)
    branches = []
    for i in range(n_classes):
        branch_1 = Dense(32, activation="relu")
        dropout = Dropout(.2)
        branch_2 = Dense(1, activation="sigmoid")
        branches.append(branch_2(dropout(branch_1(dense_2_output))))
    output = Concatenate()(branches)

    neural_model = Model(inputs=inp, outputs=output)
    neural_model.compile(
        optimizer=Adam(lr=1e-4),
        loss=focal_loss(),
        metrics=["accuracy"]
    )

    return neural_model

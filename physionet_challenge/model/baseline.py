#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Input, Concatenate

def BaselineMultibranch(n_inputs=14, n_classes=111):
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
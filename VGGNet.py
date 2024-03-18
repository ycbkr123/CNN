# 데이터셋 불러오기, 훈련셋과 테스트셋 분류
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) =\
keras.datasets.fashion_mnist.load_data()

# 데이터셋과 검증셋 분류
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2)

# 모델 설계
model = keras.models.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=32, activation='relu'),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
    keras.layers.MaxPool2D(strides=(2, 2)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=64, activation='relu'),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=64, activation='relu'),
    keras.layers.MaxPool2D(strides=(1, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=128, activation='relu'),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=128, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('bset_cnn_model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model.summary()

# 모델 학습
history = model.fit(train_input, train_target, epochs=50,
                    validation_data=(val_input, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

# 성능 측정
model.evaluate(test_input, test_target)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

plt.imshow(train_input[0].reshape(28,28), cmap='gray_r')
plt.show()
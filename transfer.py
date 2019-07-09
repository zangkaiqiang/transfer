import tensorflow as tf
from tensorflow import keras

from prepare import train_generator, val_generator
from model import IMG_SHAPE
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=1,
                help="# of GPUs to use for training")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]


def mobilenetv2():
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    return base_model


def resnet50():
    weights_path = 'input/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    return tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                          include_top=False,
                                          weights=weights_path)


def train():
    base_model = resnet50()
    model = tf.keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    if G > 1:
        model = keras.utils.multi_gpu_model(model, gpus=G)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    epochs = 10
    steps_per_epoch = train_generator.n
    validation_steps = val_generator.n

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  workers=4,
                                  validation_data=val_generator,
                                  validation_steps=validation_steps)

    return history


if __name__ == '__main__':
    train()

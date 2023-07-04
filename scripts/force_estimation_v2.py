# -*- coding: utf-8 -*-

from model import *

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras


image_height = 360
image_width = 512
input_image_shape = [image_height, image_width]


class ForceEstimationModel(tf.keras.Model):

    def __init__(self, *args, **kargs):
        super(ForceEstimationModel, self).__init__(*args, **kargs)
        tracker_names = ['loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        xs, y_labels = data
        xs, y_labels = augment(xs, y_labels)

        with tf.GradientTape() as tape:
            y_pred = self(xs, training=True)
            loss = self.compute_loss(y_labels, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        xs, y_labels = data

        y_pred = self(xs, training=False)
        loss = self.compute_loss(y_labels, y_pred)

        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, y_labels, y_pred):
        loss = tf.reduce_mean(tf.square(y_labels - y_pred))
        return loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


def model_resnet_decoder(input_shape, name='resnet_decoder'):
    feature_input = tf.keras.Input(shape=(input_shape))
    x = feature_input
    x = res_block(x, [1024, 512], 3, strides=[1, 1], name='resb1')
    x = res_block(x, [256, 128], 3, strides=[1, 1], name='resb2')
    x = upsample(x, (24, 32))
    x = res_block(x, [64, 64], 3, strides=[1, 1], name='resb3')
    x = res_block(x, [32, 32], 3, strides=[1, 1], name='resb4')

    x = tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)
    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def augment(xs, ys):
    """
    apply the same transform to xs and ys
    """
    img = xs
    fmap = ys

    # color transformation on xs
    # brightness_max_delta=0.2
    # contrast_lower=0.8
    # contrast_upper=1.2
    hue_max_delta = 0.05
    # img = tf.image.random_brightness(img, max_delta=brightness_max_delta)
    # img = tf.image.random_contrast(img, lower=contrast_lower, upper=contrast_upper)
    img = tf.image.random_hue(img, max_delta=hue_max_delta)
    batch_sz = tf.shape(xs)[0]
    height = tf.shape(xs)[1]
    # width = tf.shape(xs)[2]
    shift_fmap = 2.0
    shift_height = tf.cast(height * 4 / 40, tf.float32)
    shift_width = tf.cast(height * 4 / 40, tf.float32)
    angle_factor = 0.2
    rnds = tf.random.uniform(shape=(batch_sz, 2))
    img = tfa.image.translate(img, translations=tf.stack([shift_height, shift_width], axis=0)*rnds)
    fmap = tfa.image.translate(fmap, translations=tf.stack([shift_fmap, shift_fmap], axis=0)*rnds)
    rnds = tf.random.uniform(shape=(batch_sz,))
    rnds = rnds - 0.5
    img = tfa.image.rotate(img, angles=angle_factor*rnds)
    fmap = tfa.image.rotate(fmap, angles=angle_factor*rnds)

    return img, fmap


def model_rgb_to_fmap_res50(input_shape=input_image_shape, input_noise_stddev=0.3):
    input_shape = input_shape + [3]
    image_input = tf.keras.Input(shape=input_shape)

    x = image_input

    # augmentation layers
    # geometrical augmentation is applied in train_step()
    # because the same conversion needs to be applied to the labels as well.
    x = tf.keras.layers.RandomZoom(0.05)(x)
    x = tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 1.0))(x)
    x = tf.keras.layers.RandomContrast(factor=0.3)(x)
    x = tf.keras.layers.GaussianNoise(input_noise_stddev)(x)

    # encoder
    resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape)
    encoded_img = resnet50(x)

    # decoder
    decoded_img = model_resnet_decoder((12, 16, 2048))(encoded_img)

    model = ForceEstimationModel(inputs=[image_input], outputs=[decoded_img], name='model_resnet')
    model.summary()

    return model

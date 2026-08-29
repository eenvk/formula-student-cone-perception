#novkovic

"""Lightweight U-Net model for cone segmentation."""

import tensorflow as tf

from segmentation.segmentation_config import BASE_FILTERS, DROPOUT_RATE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH


def conv_block(inputs, filters):
    """Apply two convolutional layers with ReLU activation."""

    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x)

    return x


def encoder_block(inputs, filters):
    """Apply a convolutional block followed by max pooling."""

    features = conv_block(inputs, filters)
    pooled = tf.keras.layers.MaxPooling2D(pool_size=2)(features)

    return features, pooled


def decoder_block(inputs, skip_features, filters):
    """Upsample features, concatenate the skip connection, and apply convolutions."""

    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters)

    return x


def build_unet():
    """Build a lightweight U-Net for binary cone segmentation."""

    inputs = tf.keras.layers.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))

    skip_1, pooled_1 = encoder_block(inputs, BASE_FILTERS)
    skip_2, pooled_2 = encoder_block(pooled_1, BASE_FILTERS * 2)
    skip_3, pooled_3 = encoder_block(pooled_2, BASE_FILTERS * 4)

    bottleneck = conv_block(pooled_3, BASE_FILTERS * 8)
    bottleneck = tf.keras.layers.Dropout(DROPOUT_RATE)(bottleneck)

    decoded_3 = decoder_block(bottleneck, skip_3, BASE_FILTERS * 4)
    decoded_2 = decoder_block(decoded_3, skip_2, BASE_FILTERS * 2)
    decoded_1 = decoder_block(decoded_2, skip_1, BASE_FILTERS)

    outputs = tf.keras.layers.Conv2D(1, kernel_size=1, activation="sigmoid")(decoded_1)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cone_unet")
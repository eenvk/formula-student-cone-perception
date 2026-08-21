#novkovic

import tensorflow as tf


def conv_block(inputs, num_filters):
    """Apply two convolutional layers with ReLU activation."""

    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(inputs)

    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    return x


def encoder_block(inputs, num_filters):
    """
    Apply a convolutional block followed by max pooling.

    Returns both the feature map used for the skip connection
    and the pooled feature map passed to the next encoder level.
    """

    features = conv_block(
        inputs,
        num_filters
    )

    pooled = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )(features)

    return features, pooled


def decoder_block(
        inputs,
        skip_features,
        num_filters
):
    """
    Upsample the input, concatenate the encoder skip features,
    and apply a convolutional block.
    """

    x = tf.keras.layers.Conv2DTranspose(
        num_filters,
        kernel_size=2,
        strides=2,
        padding="same"
    )(inputs)

    x = tf.keras.layers.Concatenate()(
        [x, skip_features]
    )

    x = conv_block(
        x,
        num_filters
    )

    return x


def build_unet(
        input_shape=(256, 256, 3),
        num_classes=5
):
    """Build a U-Net model for multi-class semantic segmentation."""

    inputs = tf.keras.layers.Input(
        shape=input_shape
    )

    # Encoder
    skip1, pool1 = encoder_block(
        inputs,
        32
    )

    skip2, pool2 = encoder_block(
        pool1,
        64
    )

    skip3, pool3 = encoder_block(
        pool2,
        128
    )

    skip4, pool4 = encoder_block(
        pool3,
        256
    )

    # Bottleneck
    bottleneck = conv_block(
        pool4,
        512
    )

    # Decoder
    decoder1 = decoder_block(
        bottleneck,
        skip4,
        256
    )

    decoder2 = decoder_block(
        decoder1,
        skip3,
        128
    )

    decoder3 = decoder_block(
        decoder2,
        skip2,
        64
    )

    decoder4 = decoder_block(
        decoder3,
        skip1,
        32
    )

    # Pixel-wise classification
    outputs = tf.keras.layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation="softmax"
    )(decoder4)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="unet"
    )

    return model


if __name__ == "__main__":
    model = build_unet()
    model.summary()
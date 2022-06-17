import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
from dotenv import dotenv_values, load_dotenv
import os
import json
import re


def softmax_cross_entropy_with_masking(y_true, y_pred):
    p = y_pred
    pi = y_true

    # mask output where prob < 0.05 (likely illegal moves)
    zero = tf.fill(tf.shape(pi), 0.05)
    where = tf.less(pi, zero)
    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss


def conv_layer(x, filters, kernel_size, reg_const):
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        data_format="channels_last",
        padding="same",
        use_bias=False,
        activation="linear",
        kernel_regularizer=regularizers.l2(reg_const),
    )(x)

    x = layers.BatchNormalization(axis=3)(x)

    x = layers.Activation(lambda z: tf.nn.gelu(z, approximate=True))(x)

    return x


def residual_block(input_block, filters, kernel_size, reg_const):
    x = conv_layer(input_block, filters, kernel_size, reg_const)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        data_format="channels_last",
        padding="same",
        use_bias=False,
        activation="linear",
        kernel_regularizer=regularizers.l2(reg_const),
    )(x)

    x = layers.BatchNormalization(axis=3)(x)

    x = layers.add([input_block, x])

    x = layers.Activation(lambda z: tf.nn.gelu(z, approximate=True))(x)

    return x


if __name__ == "__main__":
    with open("constants.jsonc", "r") as f:
        constants = json.loads(re.sub("//.*", "", f.read(), flags=re.MULTILINE))

    main_input = keras.Input(shape=constants["GAME_STATE_SHAPE"], name="main_input")
    x = conv_layer(
        main_input,
        constants["NUM_FILTERS"],
        constants["KERNEL_SIZE"],
        constants["REG_CONST"],
    )

    for _ in range(constants["NUM_HIDDEN_RES_BLOCK"]):
        x = residual_block(
            x,
            constants["NUM_FILTERS"],
            constants["KERNEL_SIZE"],
            constants["REG_CONST"],
        )

    vh = layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        data_format="channels_last",
        padding="same",
        use_bias=False,
        activation="linear",
        kernel_regularizer=regularizers.l2(constants["REG_CONST"]),
    )(x)
    vh = layers.BatchNormalization(axis=3)(vh)
    vh = layers.Activation(lambda z: tf.nn.gelu(z, approximate=True))(vh)
    vh = layers.Flatten()(vh)
    vh = layers.Dense(
        128,
        use_bias=False,
        activation="linear",
        kernel_regularizer=regularizers.l2(constants["REG_CONST"]),
    )(vh)
    vh = layers.Activation(lambda z: tf.nn.gelu(z, approximate=True))(vh)
    vh = layers.Dense(
        1,
        use_bias=False,
        activation="tanh",
        kernel_regularizer=regularizers.l2(constants["REG_CONST"]),
        name="value_head",
    )(vh)

    ph = conv_layer(x, 128, (3, 3), constants["REG_CONST"])
    ph = layers.Conv2D(
        name="policy_head",
        filters=constants["MOVE_SHAPE"][-1],
        kernel_size=(1, 1),
        data_format="channels_last",
        padding="same",
        use_bias=False,
        activation="linear",
        kernel_regularizer=regularizers.l2(constants["REG_CONST"]),
    )(ph)

    model = keras.Model(inputs=main_input, outputs=[vh, ph])
    model.compile(
        loss={"value_head": "mean_squared_error", "policy_head": "mean_squared_error"},
        metrics={
            "value_head": "binary_accuracy",
            "policy_head": "categorical_accuracy",
        },
        optimizer=optimizers.SGD(
            learning_rate=constants["LEARNING_RATE"], momentum=constants["MOMENTUM"]
        ),
        loss_weights={"value_head": 0.5, "policy_head": 0.5},
    )
    print(model.summary())
    keras.utils.plot_model(model, "models\\graph.png", show_shapes=True)
    model.save(constants["NET_PATH"])


# hidden_cnn_layers = []
# for i in range(1 + 4):
#     hidden_cnn_layers.append({"filters": 48, "kernel_size": (3, 3)})

# net = NeuralNet.ResidualCnn(hidden_layers=hidden_cnn_layers).model
# import tensorflow as tf

# tf.keras.utils.plot_model(net, "models\\graph.png", show_shapes=True)
# print(net.summary())
# NeuralNet.save_model(net, "models\\CaroZero")

import tensorflow as tf
from tensorflow import keras


def softmax_cross_entropy_with_masking(y_true, y_pred):
    mask = tf.equal(y_true, 0.0)
    neg = tf.fill(tf.shape(y_true), -100.0)
    pi = tf.where(mask, neg, y_true)
    p = tf.where(mask, neg, y_pred)

    pi = tf.nn.softmax(tf.reshape(pi, [-1]))
    p = tf.reshape(p, [-1])

    batch_size = tf.shape(y_true)[0]
    loss = tf.math.divide(
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)),
        tf.cast(batch_size, dtype=tf.float32),
    )

    return loss


def load(path):
    model = keras.models.load_model(
        path,
        custom_objects={
            "softmax_cross_entropy_with_masking": softmax_cross_entropy_with_masking
        },
    )
    return model


def save(model, path):
    model.save(path)

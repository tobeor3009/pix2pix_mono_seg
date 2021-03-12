import tensorflow as tf
from tensorflow.keras import backend as K

AXIS = [1, 2]


def f1_loss_for_training(y_true, y_pred):

    tp = tf.math.reduce_sum(y_true * y_pred, axis=AXIS)
    fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=AXIS)
    fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=AXIS)

    p = tp / (tp + fp + K.epsilon())  # Precision = tp / tp + fp
    r = tp / (tp + fn + K.epsilon())  # Recall = tp / (tp+fn)

    f1 = (2 * p * r) / (p + r + K.epsilon())

    return 1 - K.mean(f1)


def f1_score(y_true, y_pred):

    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    tp = tf.math.reduce_sum(y_true * y_pred, axis=AXIS)
    fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=AXIS)
    fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=AXIS)

    p = tp / (tp + fp + K.epsilon())  # Precision = tp / tp + fp
    r = tp / (tp + fn + K.epsilon())  # Recall = tp / (tp+fn)

    f1 = (2 * p * r) / (p + r + K.epsilon())

    return K.mean(f1)


def dice_loss_for_training(y_true, y_pred):
    # example input tensorsize y_true.shape = y_pred.shape = (4,512,512,1)
    # K means keras_background
    epsilon = K.epsilon()  # K.epsilon = 1e-7
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=AXIS)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_true_sum = tf.math.reduce_sum(y_true, axis=AXIS)
    y_pred_sum = tf.math.reduce_sum(y_pred, axis=AXIS)  # add tensorwise
    # tensor shape = (4)
    dice_loss = (2. * intersection + epsilon) / (y_true_sum +
                                                 y_pred_sum + epsilon)  # 0 < loss <=1 shape : (4)

    return 1 - K.mean(dice_loss)


def dice_score(y_true, y_pred):

    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    # example input tensorsize y_true.shape = y_pred.shape = (4,512,512,1)
    # K means keras_background
    epsilon = K.epsilon()  # K.epsilon = 1e-7
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=AXIS)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_true_sum = tf.math.reduce_sum(y_true, axis=AXIS)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_pred_sum = tf.math.reduce_sum(y_pred, axis=AXIS)

    # tensor shape = (4)
    dice_loss = (2. * intersection + epsilon) / \
        (y_true_sum + y_pred_sum + epsilon)  # 0 < loss <=1

    return K.mean(dice_loss)


def jaccard_coef_loss_for_training(y_true, y_pred):
    # example input tensorsize y_true.shape = y_pred.shape = (4,512,512,1)
    # K means keras_background
    epsilon = K.epsilon()  # K.epsilon = 1e-7
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=AXIS)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_true_sum = tf.math.reduce_sum(y_true, axis=AXIS)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_pred_sum = tf.math.reduce_sum(y_pred, axis=AXIS)
    # tensor shape = (4)
    jaccard_coef = (intersection + epsilon) / \
        (y_true_sum + y_pred_sum - intersection + epsilon)

    return 1 - K.mean(jaccard_coef)


def jaccard_coef_score(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    # example input tensorsize y_true.shape = y_pred.shape = (4,512,512,1)
    # K means keras_background
    epsilon = K.epsilon()  # K.epsilon = 1e-7
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=AXIS)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_true_sum = tf.math.reduce_sum(y_true, axis=AXIS)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_pred_sum = tf.math.reduce_sum(y_pred, axis=AXIS)
    # tensor shape = (4)
    jaccard_coef = (intersection + epsilon) / \
        (y_true_sum + y_pred_sum - intersection + epsilon)

    return K.mean(jaccard_coef)

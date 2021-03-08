import tensorflow as tf
from tensorflow.keras import backend as K

axis = [1, 2]


def f1_loss_for_training(y_true, y_pred):

    tp = tf.math.reduce_sum(y_true * y_pred, axis=axis)
    fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=axis)
    fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=axis)

    p = tp / (tp + fp + K.epsilon())  # Precision = tp / tp + fp
    r = tp / (tp + fn + K.epsilon())  # Recall = tp / (tp+fn)

    f1 = (2 * p * r) / (p + r + K.epsilon())

    return 1 - K.mean(f1)


def f1_score(y_true, y_pred):

    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    tp = tf.math.reduce_sum(y_true * y_pred, axis=axis)
    fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=axis)
    fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=axis)

    p = tp / (tp + fp + K.epsilon())  # Precision = tp / tp + fp
    r = tp / (tp + fn + K.epsilon())  # Recall = tp / (tp+fn)

    f1 = (2 * p * r) / (p + r + K.epsilon())

    return K.mean(f1)


def dice_loss_for_training(y_true, y_pred):
    # example input tensorsize y_true.shape = y_pred.shape = (4,512,512,1)
    # K means keras_background
    epsilon = K.epsilon()  # K.epsilon = 1e-7
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=axis)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_true_sum = tf.math.reduce_sum(y_true, axis=axis)
    y_pred_sum = tf.math.reduce_sum(y_pred, axis=axis)  # add tensorwise
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
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=axis)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_true_sum = tf.math.reduce_sum(y_true, axis=axis)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_pred_sum = tf.math.reduce_sum(y_pred, axis=axis)

    # tensor shape = (4)
    dice_loss = (2. * intersection + epsilon) / \
        (y_true_sum + y_pred_sum + epsilon)  # 0 < loss <=1

    return K.mean(dice_loss)


def jaccard_coef_loss_for_training(y_true, y_pred):
    # example input tensorsize y_true.shape = y_pred.shape = (4,512,512,1)
    # K means keras_background
    epsilon = K.epsilon()  # K.epsilon = 1e-7
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=axis)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_true_sum = tf.math.reduce_sum(y_true, axis=axis)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_pred_sum = tf.math.reduce_sum(y_pred, axis=axis)
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
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=axis)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_true_sum = tf.math.reduce_sum(y_true, axis=axis)
    # add tensorwise, shape example : (4,512,512,1) => (4)
    y_pred_sum = tf.math.reduce_sum(y_pred, axis=axis)
    # tensor shape = (4)
    jaccard_coef = (intersection + epsilon) / \
        (y_true_sum + y_pred_sum - intersection + epsilon)

    return K.mean(jaccard_coef)
# def compute_msd(trajectory, t_step):
#     diffs = trajectory- t_step
#     sqdist = np.square(diffs).sum(axis=1)
#     msds = sqdist.mean()
#     msds_std = sqdist.std()

#     return msds

# def dice_coef(y_true, y_pred):
#     smooth = 0.0001
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth) # 1>= loss >0

# def jaccard_coef(y_true, y_pred):
#     smooth = 0.0001
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth) # 1>= loss >0


# def f1_loss_for_training(y_true, y_pred):

#     tp = y_true * y_pred
#     tp = K.sum(tp * y_pred) / (K.sum(y_pred) + K.epsilon())

#     tn = (1-y_true) * (1-y_pred)
#     tn = K.sum(tn * (1-y_pred)) / (K.sum(1-y_pred) + K.epsilon())

#     fp = (1-y_true) * y_pred
#     fp = K.sum(fp * y_pred) / (K.sum(y_pred)  + K.epsilon())

#     fn = y_true * (1-y_pred)
#     fn = K.sum(fn * (1-y_pred)) / (K.sum(1-y_pred) + K.epsilon())

#     p = tp / (tp + fp + K.epsilon()) # Precision = tp / tp + fp
#     r = tp / (tp + fn + K.epsilon()) # Recall = tp / (tp+fn)

#     f1 = (2*p*r) / (p+r+K.epsilon())

#     return 1 - f1

# def f1_score(y_true, y_pred):

#     y_true = K.round(y_true)
#     y_pred = K.round(y_pred)

#     tp = y_true * y_pred
#     tp = K.sum(tp * y_pred) / (K.sum(y_pred) + K.epsilon())

#     tn = (1-y_true) * (1-y_pred)
#     tn = K.sum(tn * (1-y_pred)) / (K.sum(1-y_pred) + K.epsilon())

#     fp = (1-y_true) * y_pred
#     fp = K.sum(fp * y_pred) / (K.sum(y_pred)  + K.epsilon())

#     fn = y_true * (1-y_pred)
#     fn = K.sum(fn * (1-y_pred)) / (K.sum(1-y_pred) + K.epsilon())

#     p = tp / (tp + fp + K.epsilon()) # Precision = tp / tp + fp
#     r = tp / (tp + fn + K.epsilon()) # Recall = tp / (tp+fn)

#     f1 = (2*p*r) / (p+r+K.epsilon())

#     return f1

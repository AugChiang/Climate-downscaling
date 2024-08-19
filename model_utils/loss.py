import tensorflow as tf

# custimez loss here
class wL2(tf.keras.losses.Loss):
    def __init__(self, LOSS_GAMMA = 1.0):
        super().__init__()
        self.gamma = LOSS_GAMMA

    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred)
        y_true = tf.squeeze(y_true)
        se = tf.math.square(y_pred-y_true)
        wse = self.gamma*se + (1-self.gamma)*tf.math.multiply(se, y_true)
        wmse = tf.reduce_mean(wse)
        return wmse
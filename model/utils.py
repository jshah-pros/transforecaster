import numpy as np
import tensorflow as tf

# Helper function for creaeting positional encodings
def positional_encoding(length, depth):
    """
    length: width of one input i.e. number of time lags in one input sequence (n_steps_in)
    depth: dimension of embedding i.e. number of features used to represent a single time lag (word token in NLP) - it will be 1 for our time series.
    
    This code is taken from tensorflow's official documentation.
    """
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)

    angle_rates = 1 / (10000 ** depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis = -1) 

    return tf.cast(pos_encoding, dtype = tf.float32)

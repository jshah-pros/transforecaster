from . import layers
import tensorflow as tf

class Transforecaster(tf.keras.Model):
    """
    This code is taken from tensorflow's official documentation and modified according to requirements.
    """
    def __init__(self, *, num_layers, d_model, num_heads, dff, n_features, n_steps_out, dropout_rate = 0.1):
        """
        Removed input_vocab_size and output_vocab_size parameters, added n_features parameter for final dense layer and n_steps_out for inference loop.
        """
        super().__init__()
        self.encoder = layers.Encoder(num_layers = num_layers, d_model = d_model, num_heads = num_heads, dff = dff, dropout_rate = dropout_rate)
        self.decoder = layers.Decoder(num_layers = num_layers, d_model = d_model, num_heads = num_heads, dff = dff, dropout_rate = dropout_rate)
        self.final_layer = tf.keras.layers.Dense(n_features)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x  = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits
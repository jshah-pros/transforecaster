from . import utils
import tensorflow as tf

# Layer for positional encoding + embeddings
class PositionalEmbedding(tf.keras.layers.Layer):
    """
    This code is taken from tensorflow's official documentation and modified according to requirements.
    Instead of Embedding layer, I am using Dense layer.
    """
    def __init__(self, d_model):
        """
        Removed vocab_size parameter as it was used in Embedding layer and I have replaced that with Dense.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Dense(units = d_model) 
        self.pos_encoding = utils.positional_encoding(length = 512, depth = d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    """
    This code is taken from tensorflow's official documentation.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    """
    This code is taken from tensorflow's official documentation.
    """
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query = x,
            key = context,
            value = context,
            return_attention_scores = True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    """
    This code is taken from tensorflow's official documentation.
    """
    def call(self, x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    """
    This code is taken from tensorflow's official documentation.
    """
    def call(self, x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    """
    This code is taken from tensorflow's official documentation.
    """
    def __init__(self, d_model, dff, dropout_rate = 0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
              tf.keras.layers.Dense(dff, activation = 'relu'),
              tf.keras.layers.Dense(d_model),
              tf.keras.layers.Dropout(dropout_rate)
            ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x
    

class EncoderLayer(tf.keras.layers.Layer):
    """
    This code is taken from tensorflow's official documentation.
    """
    def __init__(self,*, d_model, num_heads, dff, dropout_rate = 0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(num_heads = num_heads, key_dim = d_model, dropout = dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    

class Encoder(tf.keras.layers.Layer):
    """
    This code is taken from tensorflow's official documentation and modified according to requirements.
    """
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate = 0.1):
        """
        Removed vocab_size parameter.
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(d_model = d_model)
        self.enc_layers = [EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout_rate = dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    """
    This code is taken from tensorflow's official documentation.
    """
    def __init__(self, *, d_model, num_heads, dff, dropout_rate = 0.1):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(num_heads = num_heads, key_dim = d_model, dropout = dropout_rate)
        self.cross_attention = CrossAttention(num_heads = num_heads, key_dim = d_model, dropout = dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x
    

class Decoder(tf.keras.layers.Layer):
    """
    This code is taken from tensorflow's official documentation and modified according to requirements.
    """
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate = 0.1):
        """
        Removed vocab_size parameter.
        """
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(d_model = d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [DecoderLayer(d_model = d_model, num_heads = num_heads,dff = dff, dropout_rate = dropout_rate) for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x
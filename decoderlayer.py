import tensorflow as tf

"""
Whats so special about causal self attention layers ?

So in a normal Attention layer each token which is generated is often based
on each and every token which was generated before it, in a causal self attention
layer we generate the next token based on only the last token, 

this gives us several efficiency benefits, and also during inference (usage) its also
faster and takes lesser memory
"""
class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__ (self, dropout_rate=None, **kwargs):
        super().__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.normalisation = tf.keras.layers.LayerNormalization()

    def call (self, x):
        print("shape:",tf.shape(x))
        attn = self.multi_head_attention(
                query = x,
                value = x,
                use_causal_mask = True
                )
        x = self.add([x, attn])
        return self.normalisation(x)

class CrossAttention (tf.keras.layers.Layer):
    def __init__ (self, dropout_rate=None, **kwargs):
        super().__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(**kwargs)

        self.add = tf.keras.layers.Add()
        self.normalisation = tf.keras.layers.LayerNormalization()

        self.last_attention_scores = -1

    def call (self, x, y, **kwargs):
        attn, attention_scores = self.multi_head_attention(
                query = x,
                value = y,
                return_attention_scores = True
                )

        self.last_attention_scores = attention_scores

        x = self.add([x, attn])
        return self.normalisation(x)

class FeedForward (tf.keras.layers.Layer):
    def __init__ (self, units, dropout_rate = 0.1):
        super().__init__()
        self.sequential = tf.keras.Sequential([
                tf.keras.layers.Dense(units = 2*units, activation = 'relu'),
                tf.keras.layers.Dense(units=units),
                tf.keras.layers.Dropout(dropout_rate)
            ])

        self.add = tf.keras.layers.Add()
        self.normalisation = tf.keras.layers.LayerNormalization()

    def call (self, x):
        x = self.add([x, self.seq(x)])
        return self.normalisation(x)

class Decoder (tf.keras.layers.Layer):
    def __init__ (self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()

        self.self_attention = CausalSelfAttention (num_heads=num_heads, key_dim=units)
        self.cross_attention = CrossAttention (num_heads=num_heads, key_dim=units, dropout_rate=dropout_rate)
        self.feed_forward   = FeedForward (units=units, dropout_rate=dropout_rate)

    def call (self, inputs, training=False):
        in_seq, out_seq = inputs

        out_seq = self.self_attention(out_seq)
        out_seq = self.cross_attention(out_seq, in_seq)
        out_seq = self.feed_forward(out_seq)

        self.last_attention_scores = self.cross_attention.last_attention_scores
        
        return out_seq


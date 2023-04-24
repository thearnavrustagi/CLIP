import tensorflow as tf
from constants import VOCAB_SIZE


"""
The sequential embedding layer is our principle layer for embedding the 
input sentence with N dimensional tensors

vocab_size : int
the number of words you are accounting for in the given program

max_length : int
the maximum possible length of any given sentence

depth : int
the depth, or dimensions during embedding

"""
class SeqEmbedding (tf.keras.layers.Layer):
    def __init__ (self, vocab_size:int, max_length:int, depth:int):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(
                    input_dim = vocab_size,
                    output_dim = depth,
                    # this is to tell the layer that 0 is a special padding character
                    mask_zero = True,
                )

        self.position_embeddings = tf.keras.layers.Embedding(
                    input_dim = max_length,
                    output_dim = depth,
                )

        self.add = tf.keras.layers.Add()

    """
    Okay this code is interesting

    So in essence this layer is creating embeddings which account for 
    firstly the POSITION of each word
    secondly the VALUE of each word

    so first of all it creates an embedding for the sequence of the words
    using the token_embeddings layer, where it embeds each word with a special
    N dimensional vector,

    next it creates a list of all the indexes of the word, and creates N dimensional
    embeddings for it,

    to create an output tensor accounting for both of these properties of the word,
    it adds both the embeddings, thus overall the returning tensor has both the properties
    of the word in it
    """
    def call (self, seq):
        seq = self.token_embeddings(seq)

        x = tf.range(tf.shape(seq)[1])
        x = x[tf.newaxis, :]
        x = self.position_embeddings(seq)

        return self.add([seq, x])

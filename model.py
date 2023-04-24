import tensorflow as tf
from dataset import flickr8k
from seqembedding import SeqEmbedding
from decoderlayer import Decoder
from output import TokenOutput
from utils import initialise_tokenizer, initialise_mobilenet, load_image
import einops

class Captioner(tf.keras.Model):
    @classmethod
    def add_method (cls, fn):
        setattr(cls, fn.__name__, fn)
        return fn

    def __init__ (self, tokenizer, feature_extractor, output_layer, num_layers=1, units=256, max_length =50, num_heads =1, dropout_rate = 0.1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.words_to_index = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
        self.indeX_to_words = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert = True)

        self.seq_embedding = SeqEmbedding(vocab_size = tokenizer.vocabulary_size(), depth=units, max_length=max_length)
        self.decoder_layers = [
                    Decoder(units, num_heads=num_heads, dropout_rate=dropout_rate)
                    for n in range(num_layers)
                ]

        self.output_layer = output_layer

    def call(self, inputs):
        image, txt = inputs
        print(image.shape)

        if image.shape[-1] == 3:
            image = self.feature_extractor(image)

        image = einops.rearrange(image, 'b h w c -> b (h w) c')
        print(image.shape)

        if txt.dtype == tf.string:
            txt = tokenizer(txt)

        txt = self.seq_embedding(txt)

        for decoder in self.decoder_layers:
            print('shap[e into decoder',image.shape,txt.shape)
            txt = decoder(inputs=(image, txt))

        txt = output_layer(txt)
        return txt
    
    def simple_gen(self, image, temperature=1):
        initial = self.words_to_index([['[START]']])
        img_features = self.feature_extractor(image[tf.newaxis, ...])
        print(tf.shape(img_features))

        tokens = initial
        for n in range(50):
            preds = self((img_features, tokens)).numpy()
            preds = preds[ :,-1,: ]
            
            if temperature == 0:
                next = tf.argmax(preds, axis=-1)[:,tf.newaxis]
            else:
                next = tf.random.categorical(preds/temperature, num_samples=1)
            tokens = tf.concat([tokens,next], axis=1)

            if next[0] == self.word_to_index('[END]'):
                break
        words = index_to_word(tokens[0, 1:-1])
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return result.numpy().decode()

def masked_loss (labels, preds):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

    mask = (label != 0) & (loss < 1e8)
    mask = tf.cast(mask, loss.dtype)

    loss = loss*mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

    return loss

def masked_acc (labels, preds):
    mask = tf.cast(labels!=0,tf.float32)
    preds = tf.argmax(preds, axis = -1)
    labels = tf.cast(labels, tf.int64)
    match = tf.cast(preds == labels, mask.dtype)
    acc = tf.reduce_sum(match*mask)/tf.reduce_sum(mask)

    return acc

class GenerateText(tf.keras.callbacks.Callback):
  def __init__(self):
    image_url = 'https://tensorflow.org/images/surf.jpg'
    image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
    self.image = load_image(image_path)

  def on_epoch_end(self, epochs=None, logs=None):
    print()
    print()
    for t in (0.0, 0.5, 1.0):
      result = self.model.simple_gen(self.image, temperature=t)
      print(result)
    print()


if __name__ == "__main__":
    tokenizer, mobilenet = initialise_tokenizer(), initialise_mobilenet()
    train_raw, test_raw = flickr8k()
    tokenizer.adapt(train_raw.map(lambda fp, text : text).unbatch().batch(1024))

    output_layer = TokenOutput(tokenizer)

    model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,
                  units=256, dropout_rate=0.5, num_layers=2, num_heads=2)

    image_url = 'https://tensorflow.org/images/surf.jpg'
    image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
    image = load_image(image_path)

    for t in (0,0.5,1):
        result = model.simple_gen(image,t)
        print("[temperature] :",t,"\n[result]",result)


    callbacks = [ GenerateText(),
    tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True)]

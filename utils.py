import tensorflow as tf
import tqdm
import re
import string
import einops
import tqdm

from constants import IMAGE_SHAPE, VOCAB_SIZE

def load_image (image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize (img, IMAGE_SHAPE[:-1])

    return img

def preprocess_text (s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
    s = tf.strings.join (['[START]' ,s ,'[END]'], separator=' ')
    return s


def initialise_tokenizer () -> tf.keras.layers.TextVectorization :
    tokenizer = tf.keras.layers.TextVectorization(
            max_tokens = VOCAB_SIZE,
            standardize = preprocess_text,
            ragged = True
            )

    return tokenizer

def initialise_mobilenet ():
    mobilenet = tf.keras.applications.MobileNetV3Small(
            input_shape = IMAGE_SHAPE,
            include_top = False,
            include_preprocessing = True 
            )
    mobilenet.trainable = False
    
    return mobilenet

def preprocess (tokenizer, train_raw):
    tokenizer.adapt(train_raw.map(lambda fp, text : text).unbatch().batch(1024))

    print(tokenizer.get_vocabulary()[:10])
    print(tokenizer([['A man eating a submarine']]))

    words_to_index = tf.keras.layers.StringLookup (
            mask_token = '',
            vocabulary = tokenizer.get_vocabulary()
            )

    index_to_word = tf.keras.layers.StringLookup (
            mask_token = "",
            vocabulary = tokenizer.get_vocabulary(),
            invert = True
            )

    return words_to_index, index_to_word

def match_shapes (images, captions):
    caption_shape = einops.parse_shape(captions, 'b c')
    captions = einops.rearrange(captions, 'b c -> (b c)')
    images = einops.repeat(
            images, 'b ... -> (b c) ...',
            c = caption_shape['c'])

    return images, captions

def prepare_txt (tokenizer ,imgs, txts):
    tokens = tokenizer(txts)

    input_tokens = tokens[ ... , :-1]
    label_tokens = tokens[ ... , 1:]
    
    return (imgs, input_tokens), label_tokens


def prepare_dataset (dataset, tokenizer, batch_size = 32, shuffle_buffer = 1000):
    dataset = (dataset
            .shuffle(shuffle_buffer)
            .map(lambda path, caption : (load_image(path), caption))
            .apply(tf.data.experimental.ignore_errors())
            .batch (batch_size))

    def to_tensor(inputs, labels):
        (images, in_tok), out_tok = inputs, labels
        return (images, in_tok.to_tensor()), out_tok.to_tensor()

    return (dataset
        .map(match_shapes, tf.data.AUTOTUNE)
        .unbatch()
        .shuffle(shuffle_buffer)
        .batch(batch_size)
        .map(lambda x,y : prepare_txt(tokenizer, x, y), tf.data.AUTOTUNE)
        .map(to_tensor, tf.data.AUTOTUNE)
        )

def save_dataset(ds, save_path, image_model, tokenizer, shards=10, batch_size=32):
  # Load the images and make batches.

  ds = (ds
        .map(lambda path, caption: (load_image(path), caption))
        .apply(tf.data.experimental.ignore_errors())
        .batch(batch_size))

  # Run the feature extractor on each batch
  # Don't do this in a .map, because tf.data runs on the CPU. 
  def gen():
    for (images, captions) in tqdm.tqdm(ds): 
      feature_maps = image_model(images)

      feature_maps, captions = match_shapes(feature_maps, captions)
      yield feature_maps, captions

  # Wrap the generator in a new tf.data.Dataset.
  new_ds = tf.data.Dataset.from_generator(
      gen,
      output_signature=(
          tf.TensorSpec(shape=image_model.output_shape),
          tf.TensorSpec(shape=(None,), dtype=tf.string)))

  # Apply the tokenization 
  new_ds = (new_ds
            .map(lambda x,y : prepare_txt(tokenizer, x, y), tf.data.AUTOTUNE)
            .unbatch()
            .shuffle(1000))

  # Save the dataset into shard files.
  def shard_func(i, item):
    return i % shards
  new_ds.enumerate().save(save_path, shard_func=shard_func)

def load_dataset(save_path, batch_size=32, shuffle=1000, cycle_length=2):
  def custom_reader_func(datasets):
    datasets = datasets.shuffle(1000)
    return datasets.interleave(lambda x: x, cycle_length=cycle_length)

  ds = tf.data.Dataset.load(save_path, reader_func=custom_reader_func)

  def drop_index(i, x):
    return x

  ds = (ds
        .map(drop_index, tf.data.AUTOTUNE)
        .shuffle(shuffle)
        .padded_batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))
  return ds

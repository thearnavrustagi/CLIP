import tensorflow as tf
#from image_captioning import masked_acc, masked_loss
import sys

from utils import load_image

image_path = './test.jpeg' if len(sys.argv) == 1 else sys.argv[1]
model_path = './model'

@tf.keras.utils.register_keras_serializable()
def standardize(s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
    s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
    return s

def masked_loss(labels, preds):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

  mask = (labels != 0) & (loss < 1e8)
  mask = tf.cast(mask, loss.dtype)

  loss = loss*mask
  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss



def masked_acc(labels, preds):
  mask = tf.cast(labels!=0, tf.float32)
  preds = tf.argmax(preds, axis=-1)
  labels = tf.cast(labels, tf.int64)
  match = tf.cast(preds == labels, mask.dtype)
  acc = tf.reduce_sum(match*mask)/tf.reduce_sum(mask)
  return acc



if __name__ == "__main__":
    model = tf.keras.models.load_model(model_path,custom_objects={'masked_acc':masked_acc, 'masked_loss':masked_loss})
    model.summary()

    print(model.simple_gen(load_image(image_path), temperature=0.5))

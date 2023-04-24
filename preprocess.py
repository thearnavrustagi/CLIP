from dataset import flickr8k, conceptual_captions
from constants import IMAGE_SHAPE
from utils import load_image
from utils import initialise_tokenizer, initialise_mobilenet
from utils import preprocess, save_dataset
import pickle

def main (train_raw, test_raw):
    mobilenet, tokenizer = initialise()
    words_to_index, index_to_word = preprocess(tokenizer, train_raw)
    save_dataset(train_raw, 'train_cache', mobilenet, tokenizer)
    save_dataset(test_raw, 'test_cache', mobilenet, tokenizer)

    save_object(tokenizer,'tokenizer')

def save_object (obj,name):
    obj_json = obj.to_json()
    with open(f'{name}.json', 'w+') as handle:
        handle.write(json.dumps(obj_json, ensure_ascii = False ))

def initialise():
    mobilenet = initialise_mobilenet()
    tokenizer = initialise_tokenizer()

    return mobilenet, tokenizer

if __name__ == "__main__":
    choose = 'flickr8k'
    train_raw, test_raw = None, None

    if choose == 'flickr8k':
      train_raw, test_raw = flickr8k()
    else:
      train_raw, test_raw = conceptual_captions(num_train=10000, num_val=5000)

    for ex_path, ex_captions in train_raw.take(1):
      print(ex_path)
      print(ex_captions)

    main(train_raw, test_raw)



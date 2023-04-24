import tensorflow as tf
from utils import load_dataset, initialise_tokenizer, initialise_mobilenet
from utils import load_image, preprocess
from output import TokenOutput
from

def main ():
    train_raw = load_dataset('train_cache')
    test_raw = load_dataset('test_cache')
    tokenizer, mobilenet = initialise_tokenizer(), initialise_mobilenet()
    with open('tokenizer.json') as f:
        tokenizer_from_json(json.load(f))

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



if __name__ == "__main__":
    main()

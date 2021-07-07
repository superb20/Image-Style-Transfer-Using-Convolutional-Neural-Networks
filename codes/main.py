import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

# Generated image size
RESIZE_HEIGHT = 607

NUM_ITER = 3000

# Weights of the different loss components
CONTENT_WEIGHT = 8e-4 # 8e-4
STYLE_WEIGHT = 8e-1 # 8e-4

# The layer to use for the content loss.
CONTENT_LAYER_NAME = "block5_conv2" # "block2_conv2"

# List of layers to use for the style loss.
STYLE_LAYER_NAMES = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

def get_result_image_size(image_path, result_height):
    image_width, image_height = keras.preprocessing.image.load_img(image_path).size
    result_width = int(image_width * result_height / image_height)
    return result_height, result_width

def preprocess_image(image_path, target_height, target_width):
    img = keras.preprocessing.image.load_img(image_path, target_size = (target_height, target_width))
    arr = keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis = 0)
    arr = vgg19.preprocess_input(arr)
    return tf.convert_to_tensor(arr)

def get_model():
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights = 'imagenet', include_top = False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in VGG19 (as a dict).
    return keras.Model(inputs = model.inputs, outputs = outputs_dict)

def get_optimizer():
    return keras.optimizers.Adam(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 8.0, decay_steps = 445, decay_rate = 0.98
            # initial_learning_rate = 2.0, decay_steps = 376, decay_rate = 0.98
        )
    )

def compute_loss(feature_extractor, combination_image, content_features, style_features):
    combination_features = feature_extractor(combination_image)
    loss_content = compute_content_loss(content_features, combination_features)
    loss_style = compute_style_loss(style_features, combination_features, combination_image.shape[1] * combination_image.shape[2])

    return CONTENT_WEIGHT * loss_content + STYLE_WEIGHT * loss_style

# A loss function designed to maintain the 'content' of the original_image in the generated_image
def compute_content_loss(content_features, combination_features):
    original_image = content_features[CONTENT_LAYER_NAME]
    generated_image = combination_features[CONTENT_LAYER_NAME]

    return tf.reduce_sum(tf.square(generated_image - original_image)) / 2

def compute_style_loss(style_features, combination_features, combination_size):
    loss_style = 0

    for layer_name in STYLE_LAYER_NAMES:
        style_feature = style_features[layer_name][0]
        combination_feature = combination_features[layer_name][0]
        loss_style += style_loss(style_feature, combination_feature, combination_size) / len(STYLE_LAYER_NAMES)

    return loss_style

# The "style loss" is designed to maintain the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of feature maps from the style reference image and from the generated image
def style_loss(style_features, combination_features, combination_size):
    S = gram_matrix(style_features)
    C = gram_matrix(combination_features)
    channels = style_features.shape[2]
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (combination_size ** 2))

def gram_matrix(x):
   x = tf.transpose(x, (2, 0, 1))
   features = tf.reshape(x, (tf.shape(x)[0], -1))
   gram = tf.matmul(features, tf.transpose(features))
   return gram

def save_result(generated_image, result_height, result_width, name):
    img = deprocess_image(generated_image, result_height, result_width)
    keras.preprocessing.image.save_img(name, img)

# Util function to convert a tensor into a valid image
def deprocess_image(tensor, result_height, result_width):
    tensor = tensor.numpy()
    tensor = tensor.reshape((result_height, result_width, 3))

    # Remove zero-center by mean pixel
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.680

    # 'BGR'->'RGB'
    tensor = tensor[:, :, ::-1]
    return np.clip(tensor, 0, 255).astype("uint8")

if __name__ == "__main__":
    # Prepare content, stlye images
    path = os.path.abspath(os.getcwd())
    content_image_path = keras.utils.get_file(path + '\dataset\paris.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
    style_image_path = keras.utils.get_file(path + '\dataset\starry_night.jpg', 'https://i.imgur.com/9ooB60I.jpg')
    result_height, result_width = get_result_image_size(content_image_path, RESIZE_HEIGHT)
    print("result resolution: (%d, %d)" % (result_height, result_width))

    # Preprocessing
    content_tensor = preprocess_image(content_image_path, result_height, result_width)
    style_tensor = preprocess_image(style_image_path, result_height, result_width)
    generated_image = tf.Variable(tf.random.uniform(style_tensor.shape, dtype=tf.dtypes.float32))
    # generated_image = tf.Variable(preprocess_image(content_image_path, result_height, result_width))

    # Build model
    model = get_model()
    optimizer = get_optimizer()
    print(model.summary())

    content_features = model(content_tensor)
    style_features = model(style_tensor)

    # Optimize result image
    for iter in range(NUM_ITER):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, generated_image, content_features, style_features)

        grads = tape.gradient(loss, generated_image)

        print("iter: %4d, loss: %8.f" % (iter, loss))
        optimizer.apply_gradients([(grads, generated_image)])

        if (iter + 1) % 100 == 0:
            name = "result/generated_at_iteration_%d.png" % (iter + 1)
            save_result(generated_image, result_height, result_width, name)

    name = "result/result_%d_%f_%f.png" % (NUM_ITER, CONTENT_WEIGHT, STYLE_WEIGHT)
    save_result(generated_image, result_height, result_width, name)

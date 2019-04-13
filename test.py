import PIL.Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from keras.applications import vgg19
from keras import backend as K
from scipy.misc import imsave
from PIL import Image
import imageio

# Variables declaration
base_image_path = "arimauro.jpg"
style_reference_image_path = "vincent-van-gogh2.jpg"
iterations = 20

# Weights to compute the final loss
total_variation_weight = 1
style_weight = 2
content_weight = 5

# Dimensions of the generated picture.
width, height = load_img(base_image_path).size
resized_width = 400
resized_height = int(width * resized_width / height)

# Preprocessing image to make it compatible with the VGG19 model
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(resized_width, resized_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# Function to convert a tensor into an image
def deprocess_image(x):
    x = x.reshape((resized_width, resized_height, 3))

    # Remove zero-center by mean pixel. Necessary when working with VGG model
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Format BGR->RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# The gram matrix of an image tensor is the inner product between the vectorized feature map in a layer.
# It is used to compute the style loss, minimizing the mean squared distance between the feature correlation map of the style image
# and the input image
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# The style_loss_per_layer represents the loss between the style of the style reference image and the generated image.
# It depends on the gram matrices of feature maps from the style reference image and from the generated image.
def style_loss_per_layer(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = resized_width * resized_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# The total_style_loss represents the total loss between the style of the style reference image and the generated image,
# taking into account all the layers considered for the style transfer, related to the style reference image.
def total_style_loss(feature_layers):
    loss = K.variable(0.)
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss_per_layer(style_reference_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl
    return loss

# The content loss maintains the features of the content image in the generated image.
def content_loss(layer_features):
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    return K.sum(K.square(combination_features - base_image_features))

# The total variation loss mantains the generated image loclaly coherent,
# smoothing the pixel variations among neighbour pixels.
def total_variation_loss(x):
    a = K.square(x[:, :resized_width - 1, :resized_height - 1, :] - x[:, 1:, :resized_height - 1, :])
    b = K.square(x[:, :resized_width - 1, :resized_height - 1, :] - x[:, :resized_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def total_loss():
    loss = K.variable(0.)

    # contribution of content_loss
    feature_layers_content = outputs_dict['block5_conv2']
    loss += content_weight * content_loss(feature_layers_content)

    # contribution of style_loss
    feature_layers_style = ['block1_conv1', 'block2_conv1',
                            'block3_conv1', 'block4_conv1',
                            'block5_conv1']
    loss += total_style_loss(feature_layers_style) * style_weight

    # contribution of variation_loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

# Evaluate the loss and the gradients respect to the generated image. It is called in the Evaluator, necessary to
# compute the gradients and the loss as two different functions (limitation of the L-BFGS algorithm) without
# excessive losses in performance
def eval_loss_and_grads(x):
    x = x.reshape((1, resized_width, resized_height, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# Save generated pictures
def save(filename, generated):
    imageio.imwrite(filename, generated)

# Get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# Placeholder for generated image
combination_image = K.placeholder((1, resized_width, resized_height, 3))

# Combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

# Build the VGG19 network with our 3 images as input
# the model is loaded with pre-trained ImageNet weights
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

# Get the outputs of each key layer, through unique names.
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
loss = total_loss()

# Get the gradients of the generated image
grads = K.gradients(loss, combination_image)
outputs = [loss]
outputs += grads

f_outputs = K.function([combination_image], outputs)

# Evaluator returns the loss and the gradient in two separate functions, but the calculation of the two variables
# are dependent. This reduces the computation time, since otherwise it would be calculated separately.
class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = preprocess_image(base_image_path)

# The oprimizer is fmin_l_bfgs
for i in range(iterations):
    print('Iteration: ', i)
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x.flatten(),
                                     fprime=evaluator.grads,
                                     maxfun=25)

    print('Current loss value:', min_val)

    # Save current generated image
    img = deprocess_image(x.copy())
    fname = 'img' + np.str(i) + '.png'
    lol = Image.fromarray(img)
    lol.save(fname);

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image
import vgg16

# this just returns a numpy array that represents the image from the filename path specified
def load_image(filename, max_size=None):
    #open the image from path
    imagejohn = PIL.Image.open(filename)
    # if we have not set the max_size meme that sets either the height or width to the john
    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = max_size / np.max(imagejohn.size)

        # Scale the image's height and width.
        size = np.array(imagejohn.size) * factor

        # The size is now floaty boi it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        imagejohn = imagejohn.resize(size, PIL.Image.LANCZOS)

    # Convert to numpy floating-point array.
    return np.float32(image)

#I just copied this as all its doing is converting to jpeg so fat meme image is a floaty boi array
def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

#average of the square of the error in both matricies, gonna use these johns in the loss functions
def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))

def create_content_loss(session, model, content_image, layer_ids):
    # session is our tf.session
    # model is the vgg16 johzn that we are using for our style transfer inference stuff
    # content_image is our numpy array that contains pixel values
    # layer_ids are the layers we are using in our model (in our case vgg16) to meme about

    #feed_dict is python dictionary john stuff
    feed_dict = model.create_feed_dict(image = content_image)

    #get refs to tensors in layers specified by layer_ids
    layers = model.get_layer_tensors(layer_ids)

    #calculate the output values of the layers when feeding content image to model (vgg16)
    values = session.run(layers, feed_dict = feed_dict)

    #set models graph bc we wanna add some computational nodes
    with model.graph.as_default():
        #this is a list of loss function values because we are calculating losses at each layers
        layer_losses = [];

        #remember in zip just makes it so you can iterate over two lists
        for value,layer in zip(values, layers):
            #values from layer in vgg16 model make sure you dont modify  with const guard
            value_guard = tf.constant(value);

            #loss function for this is just MSE
            loss_value = mean_squared_error(layer, value_guard)

            #add loss function here to list of loss functions we initialized earlier
            layer_losses.append(loss_value)

        #combined loss for all the layers is average but might be weighted different
        total_content_loss = tf.reduce_mean(layer_losses)

        return total_content_loss

#gram matrix for style loss. We use a gram matrix because it contains second order stats for our feature maps
def gram_matrix(tensor):
    shape = tensor.get_shape()
    #num of feature channels from john
    channels = int(shape[3])

    #flatten to 2d matrix
    matrix = tf.reshape(tensor, shape = [-1, channels])

    #gram matrix is matrix transposed with self
    #this means you can see if the john has any feature similarities within the same picture
    # we use this john to figure out if the mixed image should take in patterns in image
    #we dot the original matrix with its transposed matrix
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram

def create_style_loss(session, model, style_image, layer_ids):
    #essentially doing the same thing as content loss but with gram matrix instead of raw tensor values
    feed_dict = model.create_feed_dict(image = style_image)
    layers = model.get_layer_tensors(layer_ids) 

    with model.graph.as_default():
        layer_losses = []
        gram_layers = [gram_matrix(layer) for layer in layers]
        #get output value of gram_layer tensor with the style image passed in
        values = session.run(gram_layers, feed_dict = feed_dict)
        #iterate through values and layers
        for value, layer in zip(values, gram_layers):
            value_const = tf.constant(value)
            gram_value = mean_squared_error(layer, value_const)
            layer_losses.append(gram_value)
        #combined loss for all the layers is average but might be weighted different
        total_style_loss = tf.reduce_mean(layer_losses)
    return total_style_loss

#total variation denoising implementation: essentially just shifts the rows and subtracts the diff
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
    return loss

def style_transfer(content_image, style_image, content_layer_ids, style_layer_ids,
                    weight_content = 1.5, weight_style = 10.0,
                    weight_denoise = 0.3, num_iterations = 120,
                    step_size = 10.0):
    #instantiate VGG16 model everytime otherwise when we add nodes it will eat up too much RAM
    model = vgg16.VGG16()
    #make your tf session
    session = tf.InteractiveSession(graph = model.graph)

    content_loss = create_content_loss(session = session, model = model, content_image = content_image
                                        layer_ids = content_layer_ids)

    style_loss = create_style_loss(session = session, model = model, style_image = style_image,
                                    layer_ids = style_layer_ids)
    denoise_loss = create_denoise_loss(model)

    #adjust levels of loss functions
    content_adjusted = tf.Variable(1e-10, name = 'content_adjusted')
    style_adjusted = tf.Variable(1e-10, name = 'style_adjusted')
    denoise_adjusted = tf.Variable(1e-10, name = 'denoise_adjusted')

    # Initialize the adjustment values for the loss-functions.
    session.run([content_adjusted.initializer,
                 style_adjusted.initializer,
                 denoise_adjusted.initializer])

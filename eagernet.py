from IPython.display import Image, display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image
import vgg16


#this stuff is how you print to a text file with slices along the height dimension of a 3D tensor
# Generate some test data
# data = np.arange(200).reshape((4,5,10))
#
# # Write the array to disk
# with open('test.txt', 'w') as outfile:
#     # I'm writing a header here just for the sake of readability
#     # Any line starting with "#" will be ignored by numpy.loadtxt
#     outfile.write('# Array shape: {0}\n'.format(data.shape))
#
#     # Iterating through a ndimensional array produces slices along
#     # the last axis. This is equivalent to data[i,:,:] in this case
#     for data_slice in data:
#
#         # The formatting string indicates that I'm writing out
#         # the values in left-justified columns 7 characters in width
#         # with 2 decimal places.
#         np.savetxt(outfile, data_slice, fmt='%-7.2f')
#
#         # Writing out a break to indicate different slices...
#         outfile.write('# New slice\n')

# this just returns a numpy array that represents the image from the filename path specified
def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = max_size / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS)

    # Convert to numpy floating-point array.
    return np.float32(image)

#I just copied this as all its doing is converting to jpeg so fat meme image is converted from floaty boi array
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
            tensor = session.run(layer, feed_dict = feed_dict)
            tf.print(tensor, output_stream='file://text.txt')
            print(session.run(layer, feed_dict = feed_dict))
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
        gram_layers = [gram_matrix(layer) for layer in layers]
        #get output value of gram_layer tensor with the style image passed in
        values = session.run(gram_layers, feed_dict = feed_dict)
        layer_losses = []
        #iterate through values and layers
        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(gram_layer, value_const)
            layer_losses.append(loss)
        #combined loss for all the layers is average but might be weighted different
        total_style_loss = tf.reduce_mean(layer_losses)
    return total_style_loss

#total variation denoising implementation: essentially just shifts the rows and subtracts the diff
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
    return loss

def style_transfer(content_image, style_image, model,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):
    #instantiate VGG16 model everytime otherwise when we add nodes it will eat up too much RAM
    # model = vgg16.VGG16()
    #make your tf session
    session = tf.InteractiveSession(graph = model.graph)

    content_loss = create_content_loss(session=session,
                                       model=model,
                                       content_image=content_image,
                                       layer_ids=content_layer_ids)

    style_loss = create_style_loss(session=session,
                                   model=model,
                                   style_image=style_image,
                                   layer_ids=style_layer_ids)
    denoise_loss = create_denoise_loss(model)

    #adjust levels of loss functions
    content_adjusted = tf.Variable(1e-10, name = 'content_adjusted')
    style_adjusted = tf.Variable(1e-10, name = 'style_adjusted')
    denoise_adjusted = tf.Variable(1e-10, name = 'denoise_adjusted')

    # Initialize the adjustment values for the loss-functions.
    session.run([content_adjusted.initializer,
                 style_adjusted.initializer,
                 denoise_adjusted.initializer])
    #creating tf operations to update adjusted values with normalized adjusted values
    update_content_adj = content_adjusted.assign(1.0 / (content_loss + 1e-10))
    update_style_adj = style_adjusted.assign(1.0 / (style_loss + 1e-10))
    update_denoise_adj = denoise_adjusted.assign(1.0 / (denoise_loss + 1e-10))

    #weighted loss function we use for our mixed image
    loss_combined = weight_content * content_adjusted * content_loss + \
                    weight_style * style_adjusted * style_loss + \
                    weight_denoise * denoise_adjusted * denoise_loss

    # tensorflow operation to generate gradient for weighted loss function from model input image
    gradient = tf.gradients(loss_combined, model.input)

    # list of tensor for tensor flow session that we will run in each slow optimization iteration :(
    run_tensors = [gradient, update_content_adj, update_style_adj, \
                update_denoise_adj]

    #initialize mixed image with noise
    mixed_image = np.random.rand(*content_image.shape) + 128

    for i in range(num_iterations):
        feed_dict = model.create_feed_dict(image = mixed_image)

        grad, content_adjusted, style_adjusted, denoise_adjusted \
        = session.run(run_tensors, feed_dict = feed_dict)

        grad = np.squeeze(grad)

        #scale step size based on gradient
        scaled_step = step_size / (np.std(grad) + 1e-8)

        #gradient descent : update image based on gradient
        mixed_image -= grad * scaled_step

        # ensure the pixel values of numpy array mixed_image have 0 - 255 value
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        fname = 'img' + np.str(i) + '.jpeg'
        save_image(mixed_image, fname)
        #printing progress john each iteration
        print(".", end = "")

        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)

            # Print adjustment weights for loss-functions.
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(content_adjusted, style_adjusted, denoise_adjusted))

    session.close()
    return mixed_image

# all the stuff we care about in terms of function calls
tf.enable_eager_execution()
content_filename = 'willy_wonka_old.jpg'
content_image = load_image(content_filename, max_size=None)
style_filename = 'style7.jpg'
style_image = load_image(style_filename, max_size=300)
content_layer_ids = [4]
style_layer_ids = list(range(13))
test = vgg16.VGG16()
layers = test.get_layer_tensors(content_layer_ids)
session = tf.InteractiveSession(graph = test.graph)
feed_dict = test.create_feed_dict(image = content_image)
for layer in layers:
    tensor = session.run(layer, feed_dict = feed_dict)
    data = tensor
    with open('test.txt', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
            outfile.write('# New 4D slice \n')
            for data_meme_slice in data_slice:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                np.savetxt(outfile, data_meme_slice, fmt='%-7.2f')

                # Writing out a break to indicate different slices...
                outfile.write('# New slice\n')
    #np.savetxt('text.txt',tensor, delimiter = ',')
session.close()
# img = style_transfer(content_image=content_image,
#                      style_image=style_image,
#                      model = test,
#                      content_layer_ids=content_layer_ids,
#                      style_layer_ids=style_layer_ids,
#                      weight_content=1.5,
#                      weight_style=10.0,
#                      weight_denoise=0.3,
#                      num_iterations=60,
#                      step_size=10.0)
# save_image(img, 'finalimg.jpeg')

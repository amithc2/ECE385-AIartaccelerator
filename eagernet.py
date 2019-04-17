from IPython.display import Image, display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image
import vgg16

# this just returns a numpy array that represents the image from the filename path specified KEEP THIS
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


# all the stuff we care about in terms of function calls also its fine that it doesnt print mat mul because we do gram matrix in C file not in text
tf.enable_eager_execution()
content_filename = 'willy_wonka_old.jpg'
content_image = load_image(content_filename, max_size=None)
style_filename = 'style7.jpg'
style_image = load_image(style_filename, max_size=300)
content_layer_ids = [4]
style_layer_ids = list(range(13))
test = vgg16.VGG16()
session = tf.InteractiveSession(graph = test.graph)
layers = test.get_layer_tensors(content_layer_ids)
feed_dict = test.create_feed_dict(image = content_image)
values = session.run(layers, feed_dict = feed_dict)
for layer, value in zip(layers, values):
    tensor = session.run(layer, feed_dict = feed_dict)
    data = tensor
    with open('content.txt', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        #outfile.write('# Array shape: {0}\n'.format(data.shape))
        outfile.write('%d %d %d %d' % (data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
        outfile.write('\n')
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
            j = 0
            outfile.write('# New Content Layer Tensor \n')
            for data_meme_slice in data_slice:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                #outfile.write('# Dimension index: ' + np.str(j) + '\n')
                np.savetxt(outfile, data_meme_slice, fmt='%-15.7f')

                # Writing out a break to indicate different slices...
                #outfile.write('\n')
        outfile.write('done')
        outfile.write('\n')
for layer, value in zip(layers, values):
    data = value
    with open('content.txt', 'a') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        #outfile.write('# Array shape: {0}\n'.format(data.shape))
        outfile.write('%d %d %d %d' % (data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
        outfile.write('\n')
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        k = 0
        for data_slice in data:
            j = 0
            outfile.write('# New Content Value Tensor '+ np.str(k) + '\n')
            for data_meme_slice in data_slice:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                #outfile.write('# Dimension index: ' + np.str(j) + '\n')
                np.savetxt(outfile, data_meme_slice, fmt='%-15.7f')

                # Writing out a break to indicate different slices...
               # outfile.write('\n')
        outfile.write('done')
        outfile.write('\n')
# for style stuff
layers = test.get_layer_tensors(style_layer_ids)
feed_dict = test.create_feed_dict(image = style_image)
values = session.run(layers, feed_dict = feed_dict)
for layer, value in zip(layers, values):
    data = session.run(layer, feed_dict = feed_dict)
    with open('style.txt', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        #outfile.write('# Array shape: {0}\n'.format(data.shape))
        outfile.write('%d %d %d %d' % (data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
        outfile.write('\n')
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
            j = 0
            outfile.write('# New Style Layer Tensor \n')
            for data_meme_slice in data_slice:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
               # outfile.write('# Dimension index: ' + np.str(j) + '\n')
                np.savetxt(outfile, data_meme_slice, fmt='%-15.7f')

                # Writing out a break to indicate different slices...
                #outfile.write('\n')
        outfile.write('done')
        outfile.write('\n')
for layer, value in zip(layers, values):
    data = value
    with open('style.txt', 'a') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        #outfile.write('# Array shape: {0}\n'.format(data.shape))
        outfile.write('%d %d %d %d' % (data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
        outfile.write('\n')
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        k = 0
        for data_slice in data:
            j = 0
            outfile.write('# New Style Value Tensor'+ np.str(k) + '\n')
            for data_meme_slice in data_slice:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                #outfile.write('# Dimension index: ' + np.str(j) + '\n')
                np.savetxt(outfile, data_meme_slice, fmt='%-15.7f')

                # Writing out a break to indicate different slices...
                #outfile.write('\n')
        outfile.write('done')
        outfile.write('\n')
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

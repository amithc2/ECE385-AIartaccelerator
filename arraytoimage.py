import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
def preProcess(image):
    image = imresize(image, (224,224))
    mean = [123.68, 116.779, 103.939]
    images = image-mean
    return images

# img1 = imread('laska.png', mode = 'RGB')
# data = preProcess(img1)
# print(img1.shape)
# with open('input.txt', 'w') as outfile:
#     # I'm writing a header here just for the sake of readability
#     # Any line starting with "#" will be ignored by numpy.loadtxt
#     #outfile.write('# Array shape: {0}\n'.format(data.shape))
#     data_split = np.split(data, 3, axis = 2)
#     outfile.write('%d %d %d' % (data.shape[0], data.shape[1], data.shape[2]))
#     outfile.write('\n')
#     for i in range(data.shape[2]):
#             np.savetxt(outfile, data[...,i], fmt='%-15.7f')


with open('result.txt', 'r') as inFile:
    img1d = np.zeros(224*224*3, dtype = np.float)
    imgArray = np.zeros((224, 224, 3), dtype = np.float)
    i = 0
    for val in inFile.read().split():
        img1d[i] = val
        i = i + 1

    for k in range(3):
        for i in range(224):
            for j in range(224):
                imgArray[i][j][k] = img1d[j + (i*224) + (k*224*224)]

    # for i in range(224):
    #     for j in range(224):
    #         imgArray[i, j, 0] = float(inFile.read().split())
    #         imgArray[i, j, 1] = float(inFile.read().split())
    #         imgArray[i, j, 2] = float(inFile.read().split())

imsave('mixedImage.png', imgArray)

import numpy as np


weights = np.load('vgg16_weights.npz')
keys = sorted(weights.keys())
print(keys)
print(weights[keys[0]])


with open('convlayer1_1.txt', 'w') as outfile:
    convTensor = weights[keys[0]]
    convBias = weights[keys[1]]
    print(np.shape(convTensor))
    #outfile.write(format(convTensor.shape))
    outfile.write('3 3 3 64')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    outfile.write('\n')
    outfile.write('64')
    outfile.write('\n')
    for i in range(64):
        outfile.write("%f\n " % convBias[i])
outfile.close()




with open('convlayer1_2.txt', 'w') as outfile:
    convTensor = weights[keys[2]]
    convBias = weights[keys[3]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 64 64')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('64')
    outfile.write('\n')
    for i in range(64):
        outfile.write("%f\n " % convBias[i])
outfile.close()

with open('convlayer2_1.txt', 'w') as outfile:
    convTensor = weights[keys[4]]
    convBias = weights[keys[5]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 64 128')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('128')
    outfile.write('\n')
    for i in range(128):
        outfile.write("%f\n " % convBias[i])
outfile.close()

with open('convlayer2_2.txt', 'w') as outfile:
    convTensor = weights[keys[6]]
    convBias = weights[keys[7]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 128 128')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('128')
    outfile.write('\n')
    for i in range(128):
        outfile.write("%f\n " % convBias[i])
outfile.close()


with open('convlayer3_1.txt', 'w') as outfile:
    convTensor = weights[keys[8]]
    convBias = weights[keys[9]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 128 256')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('256')
    outfile.write('\n')
    for i in range(256):
        outfile.write("%f\n " % convBias[i])
outfile.close()

with open('convlayer3_2.txt', 'w') as outfile:
    convTensor = weights[keys[10]]
    convBias = weights[keys[11]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 256 256')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('256')
    outfile.write('\n')
    for i in range(256):
        outfile.write("%f\n " % convBias[i])
outfile.close()


with open('convlayer3_3.txt', 'w') as outfile:
    convTensor = weights[keys[12]]
    convBias = weights[keys[13]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 256 256')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('256')
    outfile.write('\n')
    for i in range(256):
        outfile.write("%f\n " % convBias[i])
outfile.close()

with open('convlayer4_1.txt', 'w') as outfile:
    convTensor = weights[keys[14]]
    convBias = weights[keys[15]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 256 512')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
    #    outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('512')
    outfile.write('\n')
    for i in range(512):
        outfile.write("%f\n " % convBias[i])
outfile.close()

with open('convlayer4_2.txt', 'w') as outfile:
    convTensor = weights[keys[16]]
    convBias = weights[keys[17]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 512 512')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('512')
    outfile.write('\n')
    for i in range(512):
        outfile.write("%f\n " % convBias[i])
outfile.close()

with open('convlayer4_3.txt', 'w') as outfile:
    convTensor = weights[keys[18]]
    convBias = weights[keys[19]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 512 512')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('512')
    outfile.write('\n')
    for i in range(512):
        outfile.write("%f\n " % convBias[i])
outfile.close()

with open('convlayer5_1.txt', 'w') as outfile:
    convTensor = weights[keys[20]]
    convBias = weights[keys[21]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 512 512')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('512')
    outfile.write('\n')
    for i in range(512):
        outfile.write("%f\n " % convBias[i])
outfile.close()

with open('convlayer5_2.txt', 'w') as outfile:
    convTensor = weights[keys[22]]
    convBias = weights[keys[23]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 512 512')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('512')
    outfile.write('\n')
    for i in range(512):
        outfile.write("%f\n " % convBias[i])
outfile.close()

with open('convlayer5_3.txt', 'w') as outfile:
    convTensor = weights[keys[24]]
    convBias = weights[keys[25]]
    print(np.shape(convTensor))
    #outfile.write('# Weight matrix shape: {0}\n'.format(convTensor.shape))
    outfile.write('3 3 512 512')
    outfile.write('\n')
    for i in range(convTensor.shape[3]):
        for j in range(convTensor.shape[2]):
            for k in range(convTensor.shape[0]):
                np.savetxt(outfile, convTensor[k,...,j,i], fmt='%-7.9f')
        #outfile.write('# New filter\n')
    #outfile.write('# Bias matrix shape: {0}\n'.format(convBias.shape))
    outfile.write('\n')
    outfile.write('512')
    outfile.write('\n')
    for i in range(512):
        outfile.write("%f\n " % convBias[i])
outfile.close()

# with open('fc6.txt', 'w') as outfile:
#     fcTensor = weights[keys[26]]
#     fcBias = weights[keys[27]]
#     print(np.shape(fcTensor))
#     outfile.write('# Weight matrix shape: {0}\n'.format(fcTensor.shape))
#     fcLayer = np.split(fcTensor, 4096, axis = 1)
#     for dataSlice in fcLayer:
#         for dataminiSlice in dataSlice:
#             np.savetxt(outfile, dataminiSlice, fmt='%-7.2f', newline = " ")
#         outfile.write('# New filter\n')
#     outfile.write('# Bias matrix shape: {0}\n'.format(fcBias.shape))
#     for i in range(4096):
#         outfile.write("%f\n " % fcBias[i])
# outfile.close()
#
# with open('fc7.txt', 'w') as outfile:
#     fcTensor = weights[keys[28]]
#     fcBias = weights[keys[29]]
#     print(np.shape(fcTensor))
#     outfile.write('# Weight matrix shape: {0}\n'.format(fcTensor.shape))
#     fcLayer = np.split(fcTensor, 4096, axis = 1)
#     for dataSlice in fcLayer:
#         for dataminiSlice in dataSlice:
#             np.savetxt(outfile, dataminiSlice, fmt='%-7.2f',newline = " ")
#         outfile.write('# New filter\n')
#     outfile.write('# Bias matrix shape: {0}\n'.format(fcBias.shape))
#     for i in range(4096):
#         outfile.write("%f\n " % fcBias[i])
# outfile.close()
#
# with open('fc8.txt', 'w') as outfile:
#     fcTensor = weights[keys[30]]
#     fcBias = weights[keys[31]]
#     print(np.shape(fcTensor))
#     outfile.write('# Weight matrix shape: {0}\n'.format(fcTensor.shape))
#     fcLayer = np.split(fcTensor, 1000, axis = 1)
#     for dataSlice in fcLayer:
#         for dataminiSlice in dataSlice:
#             np.savetxt(outfile, dataminiSlice, fmt='%-7.2f', newline = " ")
#         outfile.write('# New filter\n')
#     outfile.write('# Bias matrix shape: {0}\n'.format(fcBias.shape))
#     for i in range(1000):
#         outfile.write("%f\n " % fcBias[i])
# outfile.close()

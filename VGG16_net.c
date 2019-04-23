#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// IMPLEMENTATION OF VGG16 IN C:

// Dense Layer
// float* denseLayer(int output_shape, float* input, float* weight, float* bias, int h1, int w1, int h2, int w2){
//   int i, j;
//   int size;
//   float* matMul;
//   matMul = matrixMultiplier(input, weight, h1, w1, h2, w2);
//   for(i = 0; i < h1; i++){
//     for(j = 0; j < w2; j++){
//       matMul[(w2*i)+j] = matMul[(w2*i)+j] + bias[j];
//     }
//   }
//   return matMul;
// }


float** getWeights(char* layerFile){
  // variable declarations

  FILE* weightsFile;
  float* getLayer;
  float* getBias;
  float** LayerBias;
  float val;
  int dimVal1, dimVal2, dimVal3, dimVal4;
  int i, length;

  // return both the getLayer and getBias pointers in an array
  LayerBias = (float**)malloc(sizeof(float*)*2);
  // open the desired file
  if((weightsFile = fopen(layerFile, "r")) == NULL){
    printf("Content file not found!");
    return NULL;
  }
  // get the dimensions of the 3d kernel
  fscanf(weightsFile, "%d", &dimVal1);
  fscanf(weightsFile, "%d", &dimVal2);
  fscanf(weightsFile, "%d", &dimVal3);
  fscanf(weightsFile, "%d", &dimVal4);


  length = dimVal1*dimVal2*dimVal3*dimVal4;
  getLayer = (float*)malloc(sizeof(float)*length);

  for(i = 0; i < length; i++){
    fscanf(weightsFile, "%f", &val);
    getLayer[i] = val;
  }

  LayerBias[0] = getLayer;

  // get length of 1d bias matrix
  fscanf(weightsFile, "%d", &length);
  getBias = (float*)malloc(sizeof(float)*length);

  for(i = 0; i < length; i++){
    fscanf(weightsFile, "%f", &val);
    getBias[i] = val;
  }
  // return array
  LayerBias[1] = getBias;
  return LayerBias;
}

// matrixMultiplier
float* matrixMultiplier(float* matrix1, float* matrix2, int h1, int w1, int h2, int w2){

  // declarations
  int i, j, k;
  float sum;
  sum = 0;
  // num columns of first matrix must equal num rows of second matrix
  if(w1 != h2){
    return NULL;
  }
  // result will have same num rows as first matrix, same num columns as second
  float* result = (float*)malloc(sizeof(float)*(h1*w2));

  // multiply matrices
  for(i = 0; i < h1; i++){
    for(j = 0; j < w2; j++){
      for(k = 0; k < h2; k++){
        sum = sum + matrix1[(i*w1)+k]*matrix2[(k*w2)+j];
      }
      //printf("%f ", sum);
      result[i*w2+j] = sum;
      sum = 0;
    }
  }

  return result;
}
// float* matrixIndexMultiplier(float* matrix1, float* matrix2, int h1, int w1, int h2, int w2){
//
//   // declarations
//   int i, j, k;
//   // num columns of first matrix must equal num rows of second matrix
//   if(w1 != h2){
//     return NULL;
//   }
//   // result will have same num rows as first matrix, same num columns as second
//   float* result = (float*)malloc(sizeof(float)*(h1*w2));
//
//   // multiply matrices
//   for(i = 0; i < h1; i++){
//     for(j = 0; j < w2; j++){
//       result[i*w2 + j] = matrix1[i*w2 + j]*matrix2[i*w2 + j];
//     }
//   }
//
//   return result;
// }

float* matrixIndexMultiplier(float* matrix1, float* matrix2, int h1, int w1, int h2, int w2, int depth){

  // declarations
  int i, j, k;
  // num columns of first matrix must equal num rows of second matrix
  if(w1 != h2){
    return NULL;
  }
  // result will have same num rows as first matrix, same num columns as second
  float* result = (float*)malloc(sizeof(float)*(h1*w2*depth));

  // multiply matrices
  for(k = 0; k < depth; k++){
    for(i = 0; i < h1; i++){
      for(j = 0; j < w2; j++){
          result[i + j * h1 + k*h1*w2] = matrix1[i + j * h1 + k*h1*w2] * matrix2[i + j * h1 + k*h1*w2];
          // result[i + (w2 * (j + (depth * k)))] = matrix1[i + (w2 * (j + (depth * k)))]*matrix2[i + (w2 * (j + (depth * k)))];
        }
      }
  }

  return result;
}
// HELPER FUNCTIONS FOR  VGG16
// sums up all the indicies in a matrix
// used for convolution kernels
float sum(float* input, int size){
  float sum = 0.0;
  unsigned i;
  for(i = 0; i < size; i++){
    sum += input[i];
  }
  return sum;
}
// this will be the pre processing function for vgg
// resize the image to 224x224
// vgg_mean computed from the training set to preprocess images
// vgg_mean= array value of : [123.68, 116.779, 103.939] in numpy this is float data type
// subtract each pixel in the image by the vgg_mean
// next  convert  RGB  to  BGR by right shifting
float* preprocess(char* inputImageFile){
  // RGB2BGR macro
  // Resize image
  // subtract vgg mean
  float* imgArray;
  float val;
  int dimVal1, dimVal2, dimVal3;
  int i, length;
  FILE* image;
  if((image = fopen(inputImageFile, "r")) == NULL){
    printf("Content file not found!");
    return NULL;
  }

  fscanf(image, "%d", &dimVal1);
  fscanf(image, "%d", &dimVal2);
  fscanf(image, "%d", &dimVal3);

  length = 224*224*3;
  imgArray = (float*)malloc(sizeof(float)*length);
  for(i = 0; i < length; i++){
    fscanf(image, "%f", &val);
    imgArray[i] = val;
  }


  return imgArray;
}

// here is the general description of the stack of conv layers
// all conv layers are  3x3 in  the vgg16 framework
// here is some psuedo code:
/*
* float* conv_layer(array type image, array type weight){
    int m = length of first dimension of image (ex.224) SUBTRACTED BY 2
    int n = length of second dimension of image (ex.224) SUBTRACTED BY 2

    float* new_image[m, n] = {0}->this isn't actually how you do it but just initializer list of zeroes
    float* patch = temporary image patch that is 3x3
    for(int i = 0; i < m; i++){
      for(int j = 0; j < n; j++){
        patch = im[i to i+3, j to  j+3] literally just hand write this its not that hard
        new_image[i, j] = sum(multiply(patch, weight)) -> once we  finish the multiply func we just call  that and sum up indicies
      }
    }
    return new_image;
  }
*/
// assuming the weights are going to be 3x3x3 filters
// float* conv_layer(float* input_image, float* weight, int rows, int cols){
//   int m = rows - 2;
//   int n = cols - 2;
//   int i, j, a, b;
//   int y = 0;
//   int z = 0;
//   int index = 0;
//
//   // this is my IMPLEMENTATION of zero-padding
//   float input_image_padded[(rows+2)*(cols+2)];
//   for(i = 0; i < (rows + 2); i++){
//     for(j = 0; j < (cols + 2); j++){
//       if(i > 0 && i < rows + 1 && j > 0 && j < cols + 1)
//         input_image_padded[i*(cols+2) + j]  = input_image[(i-1)*cols + (j-1)];
//       else
//         input_image_padded[i*(cols+2) + j] = 0;
//       printf("padded : %f\n", input_image_padded[i*(cols+2) + j]);
//     }
//   }
//   // actual convolution
//   float* filtered_image = (float*)malloc(sizeof(float)*(rows*cols));
//   float patch[m*n];
//   for(i = 0; i < rows; i++){
//     for(j = 0; j < cols; j++){
//       y = 0;
//       for(a = 0; a < 3; a++)
//         for(b = 0; b < 3; b++){
//           // printf("%f\n", input_image[(i+a)*(cols) + j + b]);
//           patch[y] = input_image_padded[(i+a)*(cols) + j + b];
//           y++;
//         }
//       float* matrixmult = matrixIndexMultiplier(patch, weight, 3, 3, 3, 3);
//       filtered_image[index] = sum(matrixmult, 9);
//       index++;
//       free(matrixmult);
//       z++;
//     }
//   }
//   return filtered_image;
// }

// assuming the weights are going to be 3x3 filters
// 3d to 1d indexing: Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
float* convFilter(float* input_image, float* weight, float bias, int rows, int cols, int depth, int stride){
  // variable declarations
  int m = rows - 2;
  int n = cols - 2;
  int i, j, k, a, b, c;
  int y = 0;
  int z = 0;
  int index = 0;

  // zero-padding
  float input_image_padded[(rows+2)*(cols+2)*(depth)];
  for(k = 0; k < (depth); k++){
    for(i = 0; i < (rows + 2); i++){
      for(j = 0; j < (cols + 2); j++){
        if(i > 0 && i < rows + 1 && j > 0 && j < cols + 1){
          input_image_padded[j + i*(cols+2) + k*(rows+2)*(cols+2)] = input_image[(j-1) + (i-1)*cols + k*(rows)*(cols)];
          //input_image_padded[i + (cols+2) * (j + (depth * k))] = input_image[(i - 1) + (cols * ((j - 1) + (depth * (k))))];
          // printf("unpadded : %f\n", input_image[(j-1) + (i-1)*cols + k*(rows)*(cols)]);
          // printf("%d\n",test );
        }
        else{
          input_image_padded[j + i*(cols+2) + k*(rows+2)*(cols+2)] = 0;
        }
        // printf("padded : %f\n", input_image_padded[j + i*(cols+2) + k*(rows+2)*(cols+2)]);
      }
    }
  }

  // actual convolution
  float* filtered_image = (float*)malloc(sizeof(float)*(rows*cols));

  float patch[m*n*depth];

  for(k = 0; k < 1; k++){
    for(i = 0; i < rows; i+=stride){
      for(j = 0; j < cols; j+=stride){
        y = 0;
        // our filter is 3x3xdepth since the depth of our filter and the input
        // must be equal
        // printf("new patch\n" );
        for(c = 0; c < depth; c++){
            // printf("new patch dim\n" );
          for(a = 0; a < 3; a++){
            for(b = 0; b < 3; b++){
              // int testmeme = j+b + (i+a)*(cols+2) + (k+c)*(rows+2)*(cols+2);
              // printf("patchy : %d b : %d j : %d i: %d a : %d c : %d testmeme : %d\n", y, b, j, i, a, c, testmeme);
              // printf(" input padded : %f\n",input_image_padded[2] );
              patch[y] = input_image_padded[j+b + (i+a)*(cols+2) + (k+c)*(rows+2)*(cols+2)];
              // printf("patch : %f\n", patch[y]);
              // input_image_padded[(i+a) + (cols * ((j+b) + (depth * (k+c))))];
              y++;
            }
          }
        }
        float* matrixmult = matrixIndexMultiplier(patch, weight, 3, 3, 3, 3, depth);
        filtered_image[index] = sum(matrixmult, 3*3*depth) + bias;
        index++;
        free(matrixmult);
        z++;
      }
    }
  }
  return filtered_image;
}

// ReLU to introduce non linearity after convolutional layer
// this should  be  a simple single for loop:
/*
  if  we   pass by pointer void:
  void ReLU(float* x){
    for(int i = 0; i < x.length or smt; i++){
      if(x[i] < 0)
        x[i] = 0;
    }
  }G
*/
void relu(float* x, int size){
  for(int i = 0; i < size; i++)
    if(x[i] < 0)
      x[i] = 0.0;
}

// you might think we should use average pooling instead of max pooling for this use
// and you are absolutely  right, but we just didn't do it here because of time constraints on
// this project. We will use pre-trained weights for this so we don't want avg pooling to backfire
// Here is some psuedo  code:
/*
  float* maxpool(float* x, int stride){
    for(int i = 0; i < x.length rows or whatever; i++){
      for(int j = 0; j < x cols; i++){
        x[]
      }
    }
  }
*/
float* maxpool(float* x, int stride, int rows, int cols, int depth){
  float* result = (float*)malloc(sizeof(float)*(depth*(rows*cols)/stride));
  float curr_max = x[0];
  int m = rows - 1;
  int n = cols - 1;
  int y = 0;
  for(int k = 0; k < 1; k++){
    for(int i = 0; i < rows; i+=stride){
      for(int j = 0; j < cols; j+=stride){
        float curr_max = x[i*cols + j];
        for(int c = 0; c < depth; c++){
          for(int a = 0; a < 2; a++){
            for(int b = 0; b < 2; b++){
              //printf("%f\n", x[(i+a)*(cols) + j + b]);
              if(curr_max < x[j+b + (i+a)*cols + (k+c)*rows*cols])
                curr_max = x[j+b + (i+a)*cols + (k+c)*rows*cols];
            }
          }
        }
        result[y] = curr_max;
        y++;
      }
    }
  }
  return result;
}

// softmax : this is used in the last layer for vgg16
// Here is some psuedo code:
/*
  float* softmax(float* x){
    float softmax_sum = 0.0;
    float* softmax_result = malloc(sizeof(x));
    for(size_t i = 0; i < sizeof(x)/sizeof(float); i++){
      softmax_result[i] = expf(x[i]);
      softmax_sum+=softmax_result[i];
    }
    then divide
  }
*/
float* softmax(float* x, int size){
  float softmax_sum = 0.0;
  float* softmax_result = malloc(sizeof(x)*size);
  for(int i = 0; i < size; i++){
    softmax_result[i] = expf(x[i]);
    softmax_sum += softmax_result[i];
  }
  for(int i = 0; i < size; i++){
    softmax_result[i] /= softmax_sum;
  }
  return softmax_result;
}

void createVGG16(float* inputImage){
  float* prevImage;
  float** weights;
  float* layerWeight;
  float* layerBias;
  int i, j, k, m, n, filterIndex;
  float* featureMap;
  float* currFeatureMap;
  float* convKernel;
  float* newFeatureMap;

  // preprocessing done in python script
  // preprocess();

  // float* convFilter(float* input_image, float* weight, float bias, int rows, int cols, int depth);

  // CONVOLUTION LAYER 1
  // Block 1
  convKernel = (float*)malloc(sizeof(float)*27);
  featureMap = (float*)malloc(sizeof(float)*224*224*64);
  weights = getWeights("convlayer1_1.txt");
  filterIndex = 0;
  n = 0;
  for(i = 0; i < 64; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 27; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = convFilter(inputImage, convKernel, layerBias[i], 224, 224, 3, 1);
    for(m = 0; m < 224*224; m++){
      featureMap[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 27;
  }

  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);

  // Block 2
  filterIndex = 0;
  n = 0;
  newFeatureMap = (float*)malloc(sizeof(float)*224*224*64);
  convKernel = (float*)malloc(sizeof(float)*576);
  weights = getWeights("convlayer1_2.txt");
  for(i = 0; i < 64; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 576; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = convFilter(featureMap, convKernel, layerBias[i], 224, 224, 64, 1);
    for(m = 0; m < 224*224; m++){
      newFeatureMap[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 576;
  }
  free(featureMap);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);

  // perform Max Pooling
  float* pooledOutput;
  pooledOutput = maxpool(newFeatureMap, 2, 224, 224, 64);
  free(newFeatureMap);


  // CONVOLUTION LAYER 2
  // Block 1
  filterIndex = 0;
  n = 0;
  featureMap = (float*)malloc(sizeof(float)*112*112*128);
  convKernel = (float*)malloc(sizeof(float)*576);
  weights = getWeights("convlayer2_1.txt");
    for(i = 0; i < 128; i++){
      layerWeight = weights[0];
      layerBias = weights[1];
      k = 0;
      for(j = filterIndex; j < filterIndex + 576; j++){
        convKernel[k] = layerWeight[j];
        k++;
      }
      currFeatureMap = convFilter(pooledOutput, convKernel, layerBias[i], 112, 112, 64, 1);
      for(m = 0; m < 112*112; m++){
        featureMap[n] = currFeatureMap[m];
        n++;
      }
      free(currFeatureMap);
      filterIndex = filterIndex + 576;
    }
    free(pooledOutput);
    free(weights[0]);
    free(weights[1]);
    free(weights);
    free(convKernel);

   // Block 2
   filterIndex = 0;
   n = 0;
   newFeatureMap = (float*)malloc(sizeof(float)*112*112*128);
   convKernel = (float*)malloc(sizeof(float)*1152);
   weights = getWeights("convlayer2_2.txt");
     for(i = 0; i < 128; i++){
       layerWeight = weights[0];
       layerBias = weights[1];
       k = 0;
       for(j = filterIndex; j < filterIndex + 1152; j++){
         convKernel[k] = layerWeight[j];
         k++;
       }
       currFeatureMap = convFilter(featureMap, convKernel, layerBias[i], 112, 112, 64, 1);
       for(m = 0; m < 112*112; m++){
         newFeatureMap[n] = currFeatureMap[m];
         n++;
       }
       free(currFeatureMap);
       filterIndex = filterIndex + 1152;
     }
     free(featureMap);
     free(weights[0]);
     free(weights[1]);
     free(weights);
     free(convKernel);

  pooledOutput = maxpool(newFeatureMap, 2, 112, 112, 128);
  free(newFeatureMap);

  // CONVOLUTION LAYER 3
  // Block 1
  filterIndex = 0;
  n = 0;
  featureMap = (float*)malloc(sizeof(float)*56*56*256);
  convKernel = (float*)malloc(sizeof(float)*1152);
  weights = getWeights("convlayer3_1.txt");
    for(i = 0; i < 256; i++){
      layerWeight = weights[0];
      layerBias = weights[1];
      k = 0;
      for(j = filterIndex; j < filterIndex + 1152; j++){
        convKernel[k] = layerWeight[j];
        k++;
      }
      currFeatureMap = convFilter(pooledOutput, convKernel, layerBias[i], 56, 56, 128, 1);
      for(m = 0; m < 56*56; m++){
        featureMap[n] = currFeatureMap[m];
        n++;
      }
      free(currFeatureMap);
      filterIndex = filterIndex + 1152;
    }
    free(pooledOutput);
    free(weights[0]);
    free(weights[1]);
    free(weights);
    free(convKernel);

    // Block 2
    filterIndex = 0;
    n = 0;
    newFeatureMap = (float*)malloc(sizeof(float)*56*56*256);
    convKernel = (float*)malloc(sizeof(float)*2304);
    weights = getWeights("convlayer3_2.txt");
      for(i = 0; i < 256; i++){
        layerWeight = weights[0];
        layerBias = weights[1];
        k = 0;
        for(j = filterIndex; j < filterIndex + 2304; j++){
          convKernel[k] = layerWeight[j];
          k++;
        }
        currFeatureMap = convFilter(featureMap, convKernel, layerBias[i], 56, 56, 256, 1);
        for(m = 0; m < 56*56; m++){
          newFeatureMap[n] = currFeatureMap[m];
          n++;
        }
        free(currFeatureMap);
        filterIndex = filterIndex + 2304;
      }
      free(featureMap);
      free(weights[0]);
      free(weights[1]);
      free(weights);
      free(convKernel);

    // Block 3
    filterIndex = 0;
    n = 0;
    featureMap = (float*)malloc(sizeof(float)*56*56*256);
    convKernel = (float*)malloc(sizeof(float)*2304);
    weights = getWeights("convlayer3_3.txt");
      for(i = 0; i < 256; i++){
        layerWeight = weights[0];
        layerBias = weights[1];
        k = 0;
        for(j = filterIndex; j < filterIndex + 2304; j++){
          convKernel[k] = layerWeight[j];
          k++;
        }
        currFeatureMap = convFilter(newFeatureMap, convKernel, layerBias[i], 56, 56, 256, 1);
        for(m = 0; m < 56*56; m++){
          featureMap[n] = currFeatureMap[m];
          n++;
        }
        free(currFeatureMap);
        filterIndex = filterIndex + 2304;
      }
      free(newFeatureMap);
      free(weights[0]);
      free(weights[1]);
      free(weights);
      free(convKernel);

      pooledOutput = maxpool(featureMap, 2, 56, 56, 256);
      free(featureMap);


    // CONVOLUTION LAYER 4
    // Block 1
    filterIndex = 0;
    n = 0;
    featureMap = (float*)malloc(sizeof(float)*28*28*512);
    convKernel = (float*)malloc(sizeof(float)*2304);
    weights = getWeights("convlayer4_1.txt");
      for(i = 0; i < 512; i++){
        layerWeight = weights[0];
        layerBias = weights[1];
        k = 0;
        for(j = filterIndex; j < filterIndex + 2304; j++){
          convKernel[k] = layerWeight[j];
          k++;
        }
        currFeatureMap = convFilter(pooledOutput, convKernel, layerBias[i], 28, 28, 256, 1);
        for(m = 0; m < 28*28; m++){
          featureMap[n] = currFeatureMap[m];
          n++;
        }
        free(currFeatureMap);
        filterIndex = filterIndex + 2304;
      }
      free(pooledOutput);
      free(weights[0]);
      free(weights[1]);
      free(weights);
      free(convKernel);


      // Block 2
      filterIndex = 0;
      n = 0;
      newFeatureMap = (float*)malloc(sizeof(float)*28*28*512);
      convKernel = (float*)malloc(sizeof(float)*4608);
      weights = getWeights("convlayer4_2.txt");
        for(i = 0; i < 512; i++){
          layerWeight = weights[0];
          layerBias = weights[1];
          k = 0;
          for(j = filterIndex; j < filterIndex + 4608; j++){
            convKernel[k] = layerWeight[j];
            k++;
          }
          currFeatureMap = convFilter(featureMap, convKernel, layerBias[i], 28, 28, 512, 1);
          for(m = 0; m < 28*28; m++){
            newFeatureMap[n] = currFeatureMap[m];
            n++;
          }
          free(currFeatureMap);
          filterIndex = filterIndex + 4608;
        }
        free(featureMap);
        free(weights[0]);
        free(weights[1]);
        free(weights);
        free(convKernel);

        // Block 3
        filterIndex = 0;
        n = 0;
        featureMap = (float*)malloc(sizeof(float)*28*28*512);
        convKernel = (float*)malloc(sizeof(float)*4608);
        weights = getWeights("convlayer4_3.txt");
          for(i = 0; i < 512; i++){
            layerWeight = weights[0];
            layerBias = weights[1];
            k = 0;
            for(j = filterIndex; j < filterIndex + 4608; j++){
              convKernel[k] = layerWeight[j];
              k++;
            }
            currFeatureMap = convFilter(newFeatureMap, convKernel, layerBias[i], 28, 28, 512, 1);
            for(m = 0; m < 28*28; m++){
              featureMap[n] = currFeatureMap[m];
              n++;
            }
            free(currFeatureMap);
            filterIndex = filterIndex + 4608;
          }
          free(newFeatureMap);
          free(weights[0]);
          free(weights[1]);
          free(weights);
          free(convKernel);

          pooledOutput = maxpool(featureMap, 2, 28, 28, 512);
          free(featureMap);

        // CONVOLUTION LAYER 5
        // Block 1
        filterIndex = 0;
        n = 0;
        featureMap = (float*)malloc(sizeof(float)*14*14*512);
        convKernel = (float*)malloc(sizeof(float)*4608);
        weights = getWeights("convlayer5_1.txt");
          for(i = 0; i < 512; i++){
            layerWeight = weights[0];
            layerBias = weights[1];
            k = 0;
            for(j = filterIndex; j < filterIndex + 4608; j++){
              convKernel[k] = layerWeight[j];
              k++;
            }
            currFeatureMap = convFilter(pooledOutput, convKernel, layerBias[i], 14, 14, 512, 1);
            for(m = 0; m < 14*14; m++){
              featureMap[n] = currFeatureMap[m];
              n++;
            }
            free(currFeatureMap);
            filterIndex = filterIndex + 4608;
          }
          free(pooledOutput);
          free(weights[0]);
          free(weights[1]);
          free(weights);
          free(convKernel);

          // Block 2
          filterIndex = 0;
          n = 0;
          newFeatureMap = (float*)malloc(sizeof(float)*14*14*512);
          convKernel = (float*)malloc(sizeof(float)*4608);
          weights = getWeights("convlayer5_2.txt");
            for(i = 0; i < 512; i++){
              layerWeight = weights[0];
              layerBias = weights[1];
              k = 0;
              for(j = filterIndex; j < filterIndex + 4608; j++){
                convKernel[k] = layerWeight[j];
                k++;
              }
              currFeatureMap = convFilter(featureMap, convKernel, layerBias[i], 14, 14, 512, 1);
              for(m = 0; m < 14*14; m++){
                newFeatureMap[n] = currFeatureMap[m];
                n++;
              }
              free(currFeatureMap);
              filterIndex = filterIndex + 4608;
            }
            free(featureMap);
            free(weights[0]);
            free(weights[1]);
            free(weights);
            free(convKernel);

          // Block 3
          filterIndex = 0;
          n = 0;
          featureMap = (float*)malloc(sizeof(float)*14*14*512);
          convKernel = (float*)malloc(sizeof(float)*4608);
          weights = getWeights("convlayer5_3.txt");
            for(i = 0; i < 512; i++){
              layerWeight = weights[0];
              layerBias = weights[1];
              k = 0;
              for(j = filterIndex; j < filterIndex + 4608; j++){
                convKernel[k] = layerWeight[j];
                k++;
              }
              currFeatureMap = convFilter(newFeatureMap, convKernel, layerBias[i], 14, 14, 512, 1);
              for(m = 0; m < 14*14; m++){
                featureMap[n] = currFeatureMap[m];
                n++;
              }
              free(currFeatureMap);
              filterIndex = filterIndex + 4608;
            }
            free(newFeatureMap);
            free(weights[0]);
            free(weights[1]);
            free(weights);
            free(convKernel);

            pooledOutput = maxpool(featureMap, 2, 14, 14, 512);
            free(featureMap);

}


// // convert featureMapArray to 1d tensor
// k = 0;
// for(i = 0; i < 64; i++){
//   currFeatureMap = featureMapArray[i]
//   for(j = 0; j < 224*224; j++){
//     featureMap[k] = currFeatureMap[j]
//     k++;
//   }
// }




// MAIN FUNCTION IS BEING USED SIMPLY FOR TESTING PLEASE REMOVE LATER:
int main(){
  //  test for relu helper function
  float test[4] = {-.54, 54.6, 67.3, -.34};
  relu(test, 4);
  for(int i=0; i < 4; i++)
    printf("%f\n", test[i]);

  // test for sum helper function
  float test_sum = sum(test, 4);
  printf("(%f)\n", test_sum);

  // test for softmax
  float softmax_test[6] = { 8, 14, 16,  8, 14,  1};
  float* softmax_result_test = softmax(softmax_test, 6);
  for(int i=0; i < 6; i++)
    printf("%f\n", softmax_result_test[i]);
  free(softmax_result_test);

  //test for maxpool function
  printf("maxpool test\n");
  float maxpool_test[36*2] = {1, 4, 4, 1, 2, 2,
                            0, 4, 1, 2, 4, 2,
                            3, 1, 0, 3, 3, 0,
                            2, 0, 3, 1, 3, 4,
                            0, 0, 4, 0, 1, 1,
                            2, 0, 3, 1, 2, 1,
                            1, 4, 4, 1, 2, 2,
                            0, 4, 1, 2, 4, 2,
                            3, 1, 0, 3, 3, 0,
                            2, 0, 3, 1, 3, 4,
                            0, 0, 4, 0, 1, 1,
                            2, 0, 3, 1, 2, 1};
  float* maxpool_result_test = maxpool(maxpool_test, 2, 6, 6, 2);
  for(int i = 0; i < 9; i++)
    printf("result: %f\n", maxpool_result_test[i]);
  free(maxpool_result_test);

  // test for conv_layer
  printf("conv layer test\n");
  float test_conv_layer[5*5*3] = {0, 0, 1, 0, 2, // first index
                                  1, 0, 2, 0, 1,
                                  1, 0, 2, 2, 0,
                                  2, 0, 0, 2, 0,
                                  2, 1, 2, 2, 0,
                                  2, 1, 2, 1, 1, // second index
                                  2, 1, 2, 0, 1,
                                  0, 2, 1, 0, 1,
                                  1, 2, 2, 2, 2,
                                  0, 1, 2, 0, 1,
                                  2, 1, 1, 2, 0, // third index
                                  1, 0, 0, 1, 0,
                                  0, 1, 0, 0, 0,
                                  1, 0, 2, 1, 0,
                                  2, 2, 1, 1, 1};
  float test_conv_weight[3*3*3] = {-1, 0, 1,
                                   0, 0, 1,
                                   1, -1, 1,
                                   -1, 0, 1,
                                   1, -1, 1,
                                   0, 1, 0,
                                   -1, 1, 1,
                                   1, 1, 0,
                                   0, -1, 0};
  float test_bias = 1.0;
  float* conv_layer_test = convFilter(test_conv_layer, test_conv_weight, test_bias, 5, 5, 3, 2);
  for(int i = 0; i < 9; i++)
    printf("result: %f\n", conv_layer_test[i]);
  free(conv_layer_test);

  return 0;
}

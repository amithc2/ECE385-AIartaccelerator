#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// IMPLEMENTATION OF VGG16 IN C:
typedef struct layerBlock{
  float* blockOutput;
  float* beforeRelu;
  float* beforePool;
  float* beforeConv;
  int height;
  int width;
  int depth;
} layerBlock;

typedef struct layers{
  layerBlock Conv1_1;
  layerBlock Conv1_2;
  layerBlock Conv2_1;
  layerBlock Conv2_2;
  layerBlock Conv3_1;
  layerBlock Conv3_2;
  layerBlock Conv3_3;
  layerBlock Conv4_1;
  layerBlock Conv4_2;
  layerBlock Conv4_3;
  layerBlock Conv5_1;
  layerBlock Conv5_2;
  layerBlock Conv5_3;
} layers;


float** getWeights(char* layerFile){
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
  fclose(weightsFile);
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


float* rotatedMatrixMultiplier(float* matrix1, float* filter, int h1, int w1, int h2, int w2, int depth){

  // declarations
  int i, j, k;
  int rotI, rotJ, rotK;
  // num columns of first matrix must equal num rows of second matrix
  if(w1 != h2){
    return NULL;
  }
  // result will have same num rows as first matrix, same num columns as second
  float* result = (float*)malloc(sizeof(float)*(h1*w2*depth));

  // multiply matrices
  for(k = 0; k < depth; k++){
    rotI = 2;
    rotJ = 2;
    for(i = 0; i < h1; i++){
      for(j = 0; j < w2; j++){
          result[i + j * h1 + k*h1*w2] = matrix1[i + j * h1 + k*h1*w2] * filter[rotI + rotJ * h1 + k*h1*w2];
          // result[i + (w2 * (j + (depth * k)))] = matrix1[i + (w2 * (j + (depth * k)))]*matrix2[i + (w2 * (j + (depth * k)))];
          rotJ--;
        }
      rotI--;
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
  fclose(image);
  return imgArray;
}

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
  int size;
  size = (rows+2)*(cols+2)*depth;
  // zero-padding
  float* input_image_padded;
  input_image_padded = (float*)malloc(sizeof(float)*size);
  for(k = 0; k < (depth); k++){
    for(i = 0; i < (rows + 2); i++){
      for(j = 0; j < (cols + 2); j++){
        if(i > 0 && i < rows + 1 && j > 0 && j < cols + 1){
          input_image_padded[j + i*(cols+2) + k*(rows+2)*(cols+2)] = input_image[(j-1) + (i-1)*cols + k*(rows)*(cols)];

        }
        else{
          input_image_padded[j + i*(cols+2) + k*(rows+2)*(cols+2)] = 0;
        }
      }
    }
  }

  // actual convolution
  float* filtered_image = (float*)malloc(sizeof(float)*(rows*cols));
  //input_image_padded[ cols + (rows+1)*(cols+2) + (rows+2)*(cols+2)*(depth-1)] = 0;

  float* patch;
  patch = (float*)malloc(sizeof(float)*(3*3*depth));

  for(k = 0; k < 1; k++){
    for(i = 0; i < rows; i+=stride){
      for(j = 0; j < cols; j+=stride){
        y = 0;

        for(c = 0; c < depth; c++){

          for(a = 0; a < 3; a++){
            for(b = 0; b < 3; b++){

              patch[y] = input_image_padded[j+b + (i+a)*(cols+2) + (k+c)*(rows+2)*(cols+2)];
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
  //printf("%f\n", filtered_image[0]);
  free(patch);
  free(input_image_padded);
  return filtered_image;
}

// ReLU to introduce non linearity after convolutional layer
// this should  be  a simple single for loop:
float* relu(float* x, int size){
  int i;
  for(i = 0; i < size; i++){
    if(x[i] < 0)
      x[i] = 0.0;


  }


  return x;
}

// you might think we should use average pooling instead of max pooling for this use
// and you are absolutely  right, but we just didn't do it here because of time constraints on
// this project. We will use pre-trained weights for this so we don't want avg pooling to backfire
// Here is some psuedo  code:

float* maxpool(float* x, int stride, int rows, int cols, int depth){
  float* result = (float*)malloc(sizeof(float)*(depth*(rows*cols)));
  float curr_max;
  int m = rows - 1;
  int n = cols - 1;
  int y = 0;
  for(int k = 0; k < depth; k++){
    for(int i = 0; i < rows; i+=stride){
      for(int j = 0; j < cols; j+=stride){
          curr_max = x[i*cols + j + k*rows*cols];

          for(int a = 0; a < 2; a++){
            for(int b = 0; b < 2; b++){
              //printf("%f\n", x[(i+a)*(cols) + j + b]);
              if(curr_max < x[j+b + (i+a)*cols + (k)*(rows*cols)])
                curr_max = x[j+b + (i+a)*cols + k*rows*cols];
            }
          }
        result[y] = curr_max;
        y++;
      }
    }
  }
  return result;
}

// backprop for maxpool
// essentially inputs that are not the "max" are given 0 since they don't affect the output
float* backMax(float* dL, float* result, float* x, int stride, int rows, int cols, int depth){
  int cnt = 0;
  float* dX = (float*)malloc(sizeof(float)*(depth*(rows*cols)));
  int numOccurences;
  float dLval;
  float curr_max;
  int y = 0;
  for(int k = 0; k < depth; k++){
    for(int i = 0; i < rows; i+=stride){
      for(int j = 0; j < cols; j+=stride){
          dLval = dL[y];
          curr_max = result[y];
          cnt++;
          y++;
          numOccurences = 0;
          for(int a = 0; a < 2; a++){
            for(int b = 0; b < 2; b++){
              if(x[j+b + (i+a)*cols + (k)*(rows*cols)] < curr_max){
                dX[j+b + (i+a)*cols + (k)*(rows*cols)] = 0.0;
              }
              else if(x[j+b + (i+a)*cols + (k)*(rows*cols)] == curr_max){
                numOccurences++;
                if(numOccurences == 1){
                  dX[j+b + (i+a)*cols + (k)*(rows*cols)] = 1.0;
                }
                else{
                  dX[j+b + (i+a)*cols + (k)*(rows*cols)] = 0.0;
                }
              }
              else{
                dX[j+b + (i+a)*cols + (k)*(rows*cols)] = 1.0;
              }
              //printf("dX[index] is %f\n", dX[j+b + (i+a)*cols + (k)*(rows*cols)]);
            }
          }
      }
    }
  }
  return dX;
}


  //backReluOut = backRelu(backMaxOut, reluSave, 14*14*512);
float* backRelu(float* dL, float* x, int rows, int cols, int depth){
  float* dX;
  float dLval;
  dX = (float*)malloc(sizeof(float)*depth*rows*cols);
  int numLessthanZero;
  int i, j, k, a, b, y;
  for(k = 0; k < depth; k++){
    for(i = 0; i < rows; i+=2){
      for(j = 0; j < cols; j+=2){
        numLessthanZero = 0;
        for(a = 0; a < 2; a++){
          for(b = 0; b < 2; b++){
            dLval = dL[j+b + (i+a)*cols + (k)*(rows*cols)];
            if(x[j+b + (i+a)*cols + (k)*(rows*cols)] < 0.0){
              dX[j+b + (i+a)*cols + (k)*(rows*cols)] = 0.0;
              numLessthanZero++;
            }
            else{
              dX[j+b + (i+a)*cols + (k)*(rows*cols)] = dLval;
            }
          }
        }

        if(numLessthanZero == 4){
          dX[j + (i)*cols + (k)*(rows*cols)] = dL[j + (i)*cols + (k)*(rows*cols)];
        }

      }
    }
  }

  return dX;
}

float* rotateFilter(float* filter, int depth){
  float temp;
  int i, j, k;
  float* newFilter;
  for(k = 0; k < depth; k++){
    for (i = 0; i < 3 / 2; i++) {
        for (j = i; j < 3 - i - 1; j++) {

            // Swap elements of each cycle
            // in clockwise direction
            temp = filter[j+(i*3)+(9*k)];
            filter[j+(i*3)+(9*k)] = filter[((2 - j)*3)+i+(9*k)];
            filter[((2 - j)*3)+i+(9*k)] = filter[((2 - i)*3)+(2 - j)+(9*k)];
            filter[((2 - i)*3)+(2 - j)+(9*k)] = filter[(j*3)+(2 - i)+(9*k)];
            filter[(j*3)+(2 - i)+(9*k)] = temp;
        }
    }

    for (i = 0; i < 3 / 2; i++) {
        for (j = i; j < 3 - i - 1; j++) {

            // Swap elements of each cycle
            // in clockwise direction
            temp = filter[j+(i*3)+(9*k)];
            filter[j+(i*3)+(9*k)] = filter[((2 - j)*3)+i+(9*k)];
            filter[((2 - j)*3)+i+(9*k)] = filter[((2 - i)*3)+(2 - j)+(9*k)];
            filter[((2 - i)*3)+(2 - j)+(9*k)] = filter[(j*3)+(2 - i)+(9*k)];
            filter[(j*3)+(2 - i)+(9*k)] = temp;
        }
    }
  }


  newFilter = (float*)malloc(sizeof(float)*9*depth);
  for(i = 0; i < (9*depth); i++){
    newFilter[i] = filter[i];
  }
  return newFilter;
}

float* backConv(float* dL, float* filter, int rows, int cols, int depth, int stride){

  // the backpropagation step for convolution consists of performing the full convolution between
  // dL (the derivative of the loss function with respect to the output of the convolution) and
  // the 180 degree rotated filter
  float* rotatedFilter;
  // rotate f

  //rotatedFilter = rotateFilter(filter);

  // zero-pad dL
  int m = rows - 2;
  int n = cols - 2;
  int i, j, k, a, b, c;
  int y = 0;
  int z = 0;
  int index = 0;
  int size;
  size = (rows+2)*(cols+2)*depth;
  float* input_image_padded;
  input_image_padded = (float*)malloc(sizeof(float)*size);
  for(k = 0; k < (depth); k++){
    for(i = 0; i < (rows + 2); i++){
      for(j = 0; j < (cols + 2); j++){
        if(i > 0 && i < rows + 1 && j > 0 && j < cols + 1){
          input_image_padded[j + i*(cols+2) + k*(rows+2)*(cols+2)] = dL[(j-1) + (i-1)*cols + k*(rows)*(cols)];
        }
        else{
          input_image_padded[j + i*(cols+2) + k*(rows+2)*(cols+2)] = 0;
        }
      }
    }
  }

  // perform full convolution
  float* filtered_image = (float*)malloc(sizeof(float)*(rows*cols));
  float* patch;
  patch = (float*)malloc(sizeof(float)*(3*3*depth));
  for(k = 0; k < 1; k++){
    for(i = 0; i < rows; i+=stride){
      for(j = 0; j < cols; j+=stride){
        y = 0;
        for(c = 0; c < depth; c++){
            // printf("new patch dim\n" );
          for(a = 0; a < 3; a++){
            for(b = 0; b < 3; b++){

              patch[y] = input_image_padded[j+b + (i+a)*(cols+2) + (k+c)*(rows+2)*(cols+2)];
              y++;
            }
          }
        }
        float* matrixmult = rotatedMatrixMultiplier(patch, filter, 3, 3, 3, 3, depth);
        filtered_image[index] = sum(matrixmult, 3*3*depth);
        index++;
        free(matrixmult);
        z++;
      }
    }
  }
  //printf("%f\n", filtered_image[0]);
  free(patch);
  free(input_image_padded);
  return filtered_image;
}


layerBlock* createVGG16(float* inputImage){

  layers layerList;
  layerBlock* layerArray;
  float* prevImage;
  float* beforeReluCopy;
  float** weights;
  float* layerWeight;
  float* layerBias;
  int i, j, k, m, n, filterIndex;
  float* featureMap;
  float* currFeatureMap;
  float* convKernel;
  float* newFeatureMap;



  layerArray = (layerBlock*)malloc(sizeof(layerBlock)*13);
  // preprocessing done in python script
  // preprocess();

  // float* convFilter(float* input_image, float* weight, float bias, int rows, int cols, int depth);

  // CONVOLUTION LAYER 1
  // Block 1
  convKernel = (float*)malloc(sizeof(float)*27);
  featureMap = (float*)malloc(sizeof(float)*224*224*64);
  beforeReluCopy = (float*)malloc(sizeof(float)*224*224*64);
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

  for(i = 0; i < (224*224*64); i++){
    beforeReluCopy[i] = featureMap[i];
  }
  featureMap = relu(featureMap, 224*224*64);


  layerArray[0].blockOutput = featureMap;
  layerArray[0].beforeRelu = beforeReluCopy;
  layerArray[0].beforePool = NULL;
  layerArray[0].beforeConv = inputImage;
  layerArray[0].height = 224;
  layerArray[0].width = 224;
  layerArray[0].depth = 64;

  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);


  // Block 2
  filterIndex = 0;
  n = 0;
  newFeatureMap = (float*)malloc(sizeof(float)*224*224*64);
  beforeReluCopy = (float*)malloc(sizeof(float)*224*224*64);
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

  //free(featureMap);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  for(i = 0; i < (224*224*64); i++){
    beforeReluCopy[i] = newFeatureMap[i];
  }
  newFeatureMap = relu(newFeatureMap, 224*224*64);

  //printf("%f\n", newFeatureMap[1]);
  // perform Max Pooling
  float* pooledOutput;
  pooledOutput = maxpool(newFeatureMap, 2, 224, 224, 64);
  free(newFeatureMap);

  layerArray[1].blockOutput = pooledOutput;
  layerArray[1].beforeRelu = beforeReluCopy;
  layerArray[1].beforePool = newFeatureMap;
  layerArray[1].beforeConv = featureMap;
  layerArray[1].height = 112;
  layerArray[1].width = 112;
  layerArray[1].depth = 64;

  float* pooledOutput2;
  // CONVOLUTION LAYER 2
  // Block 1
  filterIndex = 0;
  n = 0;
  featureMap = (float*)malloc(sizeof(float)*112*112*128);
  convKernel = (float*)malloc(sizeof(float)*576);
  weights = getWeights("convlayer2_1.txt");
  beforeReluCopy = (float*)malloc(sizeof(float)*112*112*128);
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
      // int idx;
      // for(idx = 0; idx < 3; idx++){
      //   printf("%f\n", currFeatureMap[idx]);
      // }
      free(currFeatureMap);
      filterIndex = filterIndex + 576;
    }

    free(weights[0]);
    free(weights[1]);
    free(weights);
    free(convKernel);
    for(i = 0; i < (112*112*128); i++){
      beforeReluCopy[i] = featureMap[i];
    }
    featureMap = relu(featureMap, 112*112*128);

    //layerList.getConv2_1 = featureMap;

    layerArray[2].blockOutput = featureMap;
    layerArray[2].beforeRelu = beforeReluCopy;
    layerArray[2].beforePool = NULL;
    layerArray[2].beforeConv = pooledOutput;
    layerArray[2].height = 112;
    layerArray[2].width = 112;
    layerArray[2].depth = 128;


   // Block 2
   filterIndex = 0;
   n = 0;
   newFeatureMap = (float*)malloc(sizeof(float)*112*112*128);
   convKernel = (float*)malloc(sizeof(float)*1152);
   weights = getWeights("convlayer2_2.txt");
   beforeReluCopy = (float*)malloc(sizeof(float)*112*112*128);
     for(i = 0; i < 128; i++){
       layerWeight = weights[0];
       layerBias = weights[1];
       k = 0;
       for(j = filterIndex; j < filterIndex + 1152; j++){
         convKernel[k] = layerWeight[j];
         k++;
       }
       currFeatureMap = convFilter(featureMap, convKernel, layerBias[i], 112, 112, 128, 1);
       for(m = 0; m < 112*112; m++){
         newFeatureMap[n] = currFeatureMap[m];
         n++;
       }
       free(currFeatureMap);
       filterIndex = filterIndex + 1152;
     }
     //free(featureMap);
     free(weights[0]);
     free(weights[1]);
     free(weights);
     free(convKernel);
     for(i = 0; i < (112*112*128); i++){
       beforeReluCopy[i] = newFeatureMap[i];
     }
  newFeatureMap = relu(newFeatureMap, 112*112*128);





  pooledOutput2 = maxpool(newFeatureMap, 2, 112, 112, 128);
  free(newFeatureMap);





      layerArray[3].blockOutput = pooledOutput2;
      layerArray[3].beforeRelu = beforeReluCopy;
      layerArray[3].beforePool = newFeatureMap;
      layerArray[3].beforeConv = featureMap;
      layerArray[3].height = 56;
      layerArray[3].width = 56;
      layerArray[3].depth = 128;



  // CONVOLUTION LAYER 3
  // Block 1
  float* pooledOutput3;
  filterIndex = 0;
  n = 0;
  featureMap = (float*)malloc(sizeof(float)*56*56*256);
  convKernel = (float*)malloc(sizeof(float)*1152);
  weights = getWeights("convlayer3_1.txt");
  beforeReluCopy = (float*)malloc(sizeof(float)*56*56*256);
    for(i = 0; i < 256; i++){
      layerWeight = weights[0];
      layerBias = weights[1];
      k = 0;
      for(j = filterIndex; j < filterIndex + 1152; j++){
        convKernel[k] = layerWeight[j];
        k++;
      }
      currFeatureMap = convFilter(pooledOutput2, convKernel, layerBias[i], 56, 56, 128, 1);
      for(m = 0; m < 56*56; m++){
        featureMap[n] = currFeatureMap[m];
        n++;
      }
      free(currFeatureMap);
      filterIndex = filterIndex + 1152;
    }
    //free(pooledOutput);
    free(weights[0]);
    free(weights[1]);
    free(weights);
    free(convKernel);
    for(i = 0; i < (56*56*256); i++){
      beforeReluCopy[i] = featureMap[i];
    }
    featureMap = relu(featureMap, 56*56*256);

    //layerList.getConv3_1 = featureMap;

        layerArray[4].blockOutput = featureMap;
        layerArray[4].beforeRelu = beforeReluCopy;
        layerArray[4].beforePool = NULL;
        layerArray[4].beforeConv = pooledOutput2;
        layerArray[4].height = 56;
        layerArray[4].width = 56;
        layerArray[4].depth = 256;

    // Block 2
    filterIndex = 0;
    n = 0;
    newFeatureMap = (float*)malloc(sizeof(float)*56*56*256);
    convKernel = (float*)malloc(sizeof(float)*2304);
    weights = getWeights("convlayer3_2.txt");
    beforeReluCopy = (float*)malloc(sizeof(float)*56*56*256);
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


      //free(featureMap);
      free(weights[0]);
      free(weights[1]);
      free(weights);
      free(convKernel);
      for(i = 0; i < (56*56*256); i++){
        beforeReluCopy[i] = newFeatureMap[i];
      }
    newFeatureMap = relu(newFeatureMap, 56*56*256);
    //layerList.getConv3_2 = newFeatureMap;
    layerArray[5].blockOutput = newFeatureMap;
    layerArray[5].beforeRelu = beforeReluCopy;
    layerArray[5].beforePool = NULL;
    layerArray[5].beforeConv = featureMap;
    layerArray[5].height = 56;
    layerArray[5].width = 56;
    layerArray[5].depth = 256;


    // Block 3
    filterIndex = 0;
    n = 0;
    featureMap = (float*)malloc(sizeof(float)*56*56*256);
    convKernel = (float*)malloc(sizeof(float)*2304);
    weights = getWeights("convlayer3_3.txt");
    beforeReluCopy = (float*)malloc(sizeof(float)*56*56*256);
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
      //free(newFeatureMap);
      free(weights[0]);
      free(weights[1]);
      free(weights);
      free(convKernel);

      for(i = 0; i < (56*56*256); i++){
        beforeReluCopy[i] = featureMap[i];
      }
      featureMap = relu(featureMap, 56*56*256);
      pooledOutput3 = maxpool(featureMap, 2, 56, 56, 256);
      free(featureMap);


      //layerList.getConv3_3 = pooledOutput3;
      layerArray[6].blockOutput = pooledOutput3;
      layerArray[6].beforeRelu = beforeReluCopy;
      layerArray[6].beforePool = featureMap;
      layerArray[6].beforeConv = newFeatureMap;
      layerArray[6].height = 28;
      layerArray[6].width = 28;
      layerArray[6].depth = 256;


    // CONVOLUTION LAYER 4
    // Block 1
    float* pooledOutput4;
    filterIndex = 0;
    n = 0;
    featureMap = (float*)malloc(sizeof(float)*28*28*512);
    convKernel = (float*)malloc(sizeof(float)*2304);
    weights = getWeights("convlayer4_1.txt");
    beforeReluCopy = (float*)malloc(sizeof(float)*28*28*512);
      for(i = 0; i < 512; i++){
        layerWeight = weights[0];
        layerBias = weights[1];
        k = 0;
        for(j = filterIndex; j < filterIndex + 2304; j++){
          convKernel[k] = layerWeight[j];
          k++;
        }
        currFeatureMap = convFilter(pooledOutput3, convKernel, layerBias[i], 28, 28, 256, 1);
        for(m = 0; m < 28*28; m++){
          featureMap[n] = currFeatureMap[m];
          n++;
        }
        free(currFeatureMap);
        filterIndex = filterIndex + 2304;
      }
      //free(pooledOutput);
      free(weights[0]);
      free(weights[1]);
      free(weights);
      free(convKernel);

      for(i = 0; i < (28*28*512); i++){
        beforeReluCopy[i] = featureMap[i];
      }
      featureMap = relu(featureMap, 28*28*512);
      //layerList.getConv4_1 = featureMap;

      layerArray[7].blockOutput = featureMap;
      layerArray[7].beforeRelu = beforeReluCopy;
      layerArray[7].beforePool = NULL;
      layerArray[7].beforeConv = pooledOutput3;
      layerArray[7].height = 28;
      layerArray[7].width = 28;
      layerArray[7].depth = 512;




      // Block 2
      filterIndex = 0;
      n = 0;
      newFeatureMap = (float*)malloc(sizeof(float)*28*28*512);
      convKernel = (float*)malloc(sizeof(float)*4608);
      weights = getWeights("convlayer4_2.txt");
      beforeReluCopy = (float*)malloc(sizeof(float)*28*28*512);
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
        //free(featureMap);
        free(weights[0]);
        free(weights[1]);
        free(weights);
        free(convKernel);
        for(i = 0; i < (28*28*512); i++){
          beforeReluCopy[i] = newFeatureMap[i];
        }
        newFeatureMap = relu(newFeatureMap, 28*28*512);
        //layerList.getConv4_2 = newFeatureMap;

        layerArray[8].blockOutput = newFeatureMap;
        layerArray[8].beforeRelu = beforeReluCopy;
        layerArray[8].beforePool = NULL;
        layerArray[8].beforeConv = featureMap;
        layerArray[8].height = 28;
        layerArray[8].width = 28;
        layerArray[8].depth = 512;

        // Block 3
        filterIndex = 0;
        n = 0;
        featureMap = (float*)malloc(sizeof(float)*28*28*512);
        convKernel = (float*)malloc(sizeof(float)*4608);
        weights = getWeights("convlayer4_3.txt");
        beforeReluCopy = (float*)malloc(sizeof(float)*28*28*512);
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
        //  free(newFeatureMap);
          free(weights[0]);
          free(weights[1]);
          free(weights);
          free(convKernel);
          for(i = 0; i < (28*28*512); i++){
            beforeReluCopy[i] = featureMap[i];
          }
          featureMap = relu(featureMap, 28*28*512);
          pooledOutput4 = maxpool(featureMap, 2, 28, 28, 512);
          free(featureMap);
          //layerList.getConv4_3 = pooledOutput4;
          layerArray[9].blockOutput = pooledOutput4;
          layerArray[9].beforeRelu = beforeReluCopy;
          layerArray[9].beforePool = featureMap;
          layerArray[9].beforeConv = newFeatureMap;
          layerArray[9].height = 14;
          layerArray[9].width = 14;
          layerArray[9].depth = 512;

        // CONVOLUTION LAYER 5
        // Block 1
        float* pooledOutput5;
        filterIndex = 0;
        n = 0;
        featureMap = (float*)malloc(sizeof(float)*14*14*512);
        convKernel = (float*)malloc(sizeof(float)*4608);
        weights = getWeights("convlayer5_1.txt");
        beforeReluCopy = (float*)malloc(sizeof(float)*14*14*512);
          for(i = 0; i < 512; i++){
            layerWeight = weights[0];
            layerBias = weights[1];
            k = 0;
            for(j = filterIndex; j < filterIndex + 4608; j++){
              convKernel[k] = layerWeight[j];
              k++;
            }
            currFeatureMap = convFilter(pooledOutput4, convKernel, layerBias[i], 14, 14, 512, 1);
            for(m = 0; m < 14*14; m++){
              featureMap[n] = currFeatureMap[m];
              n++;
            }
            free(currFeatureMap);
            filterIndex = filterIndex + 4608;
          }
          //free(pooledOutput);
          free(weights[0]);
          free(weights[1]);
          free(weights);
          free(convKernel);
          for(i = 0; i < (14*14*512); i++){
            beforeReluCopy[i] = featureMap[i];
          }
          featureMap = relu(featureMap, 14*14*512);
          //layerList.getConv5_1 = featureMap;
          layerArray[10].blockOutput = featureMap;
          layerArray[10].beforeRelu = beforeReluCopy;
          layerArray[10].beforePool = NULL;
          layerArray[10].beforeConv = pooledOutput4;
          layerArray[10].height = 14;
          layerArray[10].width = 14;
          layerArray[10].depth = 512;

          // Block 2
          filterIndex = 0;
          n = 0;
          newFeatureMap = (float*)malloc(sizeof(float)*14*14*512);
          convKernel = (float*)malloc(sizeof(float)*4608);
          weights = getWeights("convlayer5_2.txt");
          beforeReluCopy = (float*)malloc(sizeof(float)*14*14*512);
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
            //free(featureMap);
            free(weights[0]);
            free(weights[1]);
            free(weights);
            free(convKernel);
            for(i = 0; i < (14*14*512); i++){
              beforeReluCopy[i] = newFeatureMap[i];
            }
          newFeatureMap = relu(newFeatureMap, 14*14*512);
          //layerList.getConv5_2 = newFeatureMap;

          layerArray[11].blockOutput = newFeatureMap;
          layerArray[11].beforeRelu = beforeReluCopy;
          layerArray[11].beforePool = NULL;
          layerArray[11].beforeConv = featureMap;
          layerArray[11].height = 14;
          layerArray[11].width = 14;
          layerArray[11].depth = 512;

          // Block 3
          filterIndex = 0;
          n = 0;
          featureMap = (float*)malloc(sizeof(float)*14*14*512);
          convKernel = (float*)malloc(sizeof(float)*4608);
          weights = getWeights("convlayer5_3.txt");
          beforeReluCopy = (float*)malloc(sizeof(float)*14*14*512);
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
            //free(newFeatureMap);
            free(weights[0]);
            free(weights[1]);
            free(weights);
            free(convKernel);






            float* reluSave;
            reluSave = (float*)malloc(sizeof(float)*14*14*512);
            for(i = 0; i < (14*14*512); i++){
              reluSave[i] = featureMap[i];
            }
             featureMap = relu(featureMap, 14*14*512);

            pooledOutput5 = maxpool(featureMap, 2, 14, 14, 512);
            //layerList.getConv5_3 = pooledOutput5;
            layerArray[12].blockOutput = pooledOutput5;
            layerArray[12].beforeRelu = beforeReluCopy;
            layerArray[12].beforePool = featureMap;
            layerArray[12].beforeConv = newFeatureMap;
            layerArray[12].height = 7;
            layerArray[12].width = 7;
            layerArray[12].depth = 512;

            return layerArray;



}

float* backPropLayer(float* dL, layerBlock* layers, int layerID){
  float* blockOut;
  float* befPool;
  float* backMaxOut;
  float* backReluOut;
  float* backConvOut;
  float** weights;
  int i, j, k, n, filterIndex, m;
  float* convKernel;
  float* layerWeight;
  float* layerBias;
  float* currFeatureMap;
  int layerIDflag;
  layerIDflag = 0;
  //  backConvOut = backMax(dL, blockOut, befPool, 2, 14, 14, 512);
  if(layerID == 12){
  // Conv 5_3
  backMaxOut = backMax(dL, layers[12].blockOutput, layers[12].beforePool, 2, 14, 14, 512);
  backReluOut = backRelu(backMaxOut, layers[12].beforeRelu, 14, 14, 512);
  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*14*14*512);
  weights = getWeights("convlayerback5_3.txt");
  convKernel = (float*)malloc(sizeof(float)*4608);
  for(i = 0; i < 512; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 4608; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 14, 14, 512, 1);
    for(m = 0; m < 14*14; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 4608;
  }
  free(backMaxOut);
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;

  printf("layerID is now %d\n", layerID);
  }

  // Conv 5_2
  if(layerID == 11){
    if(layerIDflag == 1){
      backReluOut = backRelu(backConvOut, layers[11].beforeRelu, 14, 14, 512);
      free(backConvOut);
    }
    else{
      backReluOut = backRelu(dL, layers[11].beforeRelu, 14, 14, 512);
    }

  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*14*14*512);
  weights = getWeights("convlayerback5_2.txt");
  convKernel = (float*)malloc(sizeof(float)*4608);
  for(i = 0; i < 512; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 4608; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 14, 14, 512, 1);
    for(m = 0; m < 14*14; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 4608;
  }
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }

  // Conv5_1
  if(layerID == 10){
    if(layerIDflag == 1){
      backReluOut = backRelu(backConvOut, layers[10].beforeRelu, 14, 14, 512);
      free(backConvOut);
    }
    else{
      backReluOut = backRelu(dL, layers[10].beforeRelu, 14, 14, 512);
    }
  filterIndex = 0;
  n = 0;
  weights = getWeights("convlayerback5_1.txt");
  backConvOut = (float*)malloc(sizeof(float)*14*14*512);
  convKernel = (float*)malloc(sizeof(float)*4608);
  for(i = 0; i < 512; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 4608; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 14, 14, 512, 1);
    for(m = 0; m < 14*14; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 4608;
  }
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }
//backMax(float* dL, float* result, float* x, int stride, int rows, int cols, int depth)


  // Conv 4_3
  if(layerID == 9){
    if(layerIDflag == 1){
      backMaxOut = backMax(backConvOut, layers[9].blockOutput, layers[9].beforePool, 2, 28, 28, 512);
      backReluOut = backRelu(backMaxOut, layers[9].beforeRelu, 28, 28, 512);
      free(backConvOut);
    }
    else{
      backMaxOut = backMax(dL, layers[9].blockOutput, layers[9].beforePool, 2, 28, 28, 512);
      backReluOut = backRelu(backMaxOut, layers[9].beforeRelu, 28, 28, 512);
    }

  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*28*28*512);
  weights = getWeights("convlayerback4_3.txt");
  convKernel = (float*)malloc(sizeof(float)*4608);
  for(i = 0; i < 512; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 4608; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 28, 28, 512, 1);
    for(m = 0; m < 28*28; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 4608;
  }
  free(backMaxOut);
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }


  // Conv 4_2
  if(layerID == 8){
    if(layerIDflag == 1){
      backReluOut = backRelu(backConvOut, layers[8].beforeRelu, 28, 28, 512);
      free(backConvOut);
    }
    else{
      backReluOut = backRelu(dL, layers[8].beforeRelu, 28, 28, 512);
    }

  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*28*28*512);
  weights = getWeights("convlayerback4_2.txt");
  convKernel = (float*)malloc(sizeof(float)*4608);
  for(i = 0; i < 512; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 4608; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 28, 28, 512, 1);
    for(m = 0; m < 28*28; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 4608;
  }
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }
//backConv(float* dL, float* filter, int stride, int rows, int cols, int depth)

  // Conv 4_1
  if(layerID == 7){
    if(layerIDflag == 1){
      backReluOut = backRelu(backConvOut, layers[7].beforeRelu, 28, 28, 512);
      free(backConvOut);
    }
    else{
      backReluOut = backRelu(dL, layers[7].beforeRelu, 28, 28, 512);
    }

  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*28*28*256);
  weights = getWeights("convlayerback4_1.txt");
  convKernel = (float*)malloc(sizeof(float)*4608);
  for(i = 0; i < 256; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 4608; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 28, 28, 512, 1);
    for(m = 0; m < 28*28; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 4608;
  }

  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }
  //
  // Conv 3_3
  if(layerID == 6){
    if(layerIDflag == 1){
      backMaxOut = backMax(backConvOut, layers[6].blockOutput, layers[6].beforePool, 2, 56, 56, 256);
      backReluOut = backRelu(backMaxOut, layers[6].beforeRelu, 56, 56, 256);
      free(backConvOut);
    }
    else{
      backMaxOut = backMax(dL, layers[6].blockOutput, layers[6].beforePool, 2, 56, 56, 256);
      backReluOut = backRelu(backMaxOut, layers[6].beforeRelu, 56, 56, 256);
    }

  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*56*56*256);
  weights = getWeights("convlayerback3_3.txt");
  convKernel = (float*)malloc(sizeof(float)*2304);
  for(i = 0; i < 256; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 2304; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 56, 56, 256, 1);
    for(m = 0; m < 56*56; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 2304;
  }
  free(backMaxOut);
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }
  // Conv 3_2
  if(layerID == 5){
    if(layerIDflag == 1){
      backReluOut = backRelu(backConvOut, layers[5].beforeRelu, 56, 56, 256);
      free(backConvOut);
    }
    else{
      backReluOut = backRelu(dL, layers[5].beforeRelu, 56, 56, 256);
    }

  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*56*56*256);
  weights = getWeights("convlayerback3_2.txt");
  convKernel = (float*)malloc(sizeof(float)*2304);
  for(i = 0; i < 256; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 2304; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 56, 56, 256, 1);
    for(m = 0; m < 56*56; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 2304;
  }
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }

  // Conv 3_1
  if(layerID == 4){
    if(layerIDflag == 1){
      backReluOut = backRelu(backConvOut, layers[4].beforeRelu, 56, 56, 256);
      free(backConvOut);
    }
    else{
      backReluOut = backRelu(dL, layers[4].beforeRelu, 56, 56, 256);
    }

  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*56*56*128);
  weights = getWeights("convlayerback3_1.txt");
  convKernel = (float*)malloc(sizeof(float)*2304);
  for(i = 0; i < 128; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 2304; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 56, 56, 256, 1);
    for(m = 0; m < 56*56; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 2304;
  }

  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }

  // Conv 2_2
  if(layerID == 3){
    if(layerIDflag == 1){
      backMaxOut = backMax(backConvOut, layers[3].blockOutput, layers[3].beforePool, 2, 112, 112, 128);
      backReluOut = backRelu(backMaxOut, layers[3].beforeRelu, 112, 112, 128);
      free(backConvOut);
    }
    else{
      backMaxOut = backMax(dL, layers[3].blockOutput, layers[3].beforePool, 2, 112, 112, 128);
      backReluOut = backRelu(backMaxOut, layers[3].beforeRelu, 112, 112, 128);
    }
  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*112*112*128);
  weights = getWeights("convlayerback2_2.txt");
  convKernel = (float*)malloc(sizeof(float)*1152);
  for(i = 0; i < 128; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 1152; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 112, 112, 128, 1);
    for(m = 0; m < 112*112; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 1152;
  }
  free(backMaxOut);
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }

  //
  // Conv 2_1
  if(layerID == 2){
    if(layerIDflag == 1){
      backReluOut = backRelu(backConvOut, layers[2].beforeRelu, 112, 112, 128);
      free(backConvOut);
    }
    else{
      backReluOut = backRelu(dL, layers[2].beforeRelu, 112, 112, 128);
    }
  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*112*112*64);
  weights = getWeights("convlayerback2_1.txt");
  convKernel = (float*)malloc(sizeof(float)*1152);
  for(i = 0; i < 64; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 1152; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 112, 112, 128, 1);
    for(m = 0; m < 112*112; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 1152;
  }
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }
  // Conv 1_2
  if(layerID == 1){
    if(layerIDflag == 1){
      backMaxOut = backMax(backConvOut, layers[1].blockOutput, layers[1].beforePool, 2, 224, 224, 64);
      backReluOut = backRelu(backMaxOut, layers[1].beforeRelu, 224, 224, 64);
      free(backConvOut);
    }
    else{
      backMaxOut = backMax(dL, layers[1].blockOutput, layers[1].beforePool, 2, 224, 224, 64);
      backReluOut = backRelu(backMaxOut, layers[1].beforeRelu, 224, 224, 64);
    }

  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*224*224*64);
  weights = getWeights("convlayerback1_2.txt");
  convKernel = (float*)malloc(sizeof(float)*576);
  for(i = 0; i < 64; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 576; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 224, 224, 64, 1);
    for(m = 0; m < 224*224; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 576;
  }
  free(backMaxOut);
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerID--;
  layerIDflag = 1;
  printf("layerID is now %d\n", layerID);
  }
  // Conv 1_1
  if(layerID == 0){
    if(layerIDflag == 1){
      backReluOut = backRelu(backConvOut, layers[0].beforeRelu, 224, 224, 64);
    }
    else{
      backReluOut = backRelu(dL, layers[0].beforeRelu, 224, 224, 64);
    }
  filterIndex = 0;
  n = 0;
  backConvOut = (float*)malloc(sizeof(float)*224*224*3);
  weights = getWeights("convlayerback1_1.txt");
  convKernel = (float*)malloc(sizeof(float)*576);
  for(i = 0; i < 3; i++){
    layerWeight = weights[0];
    layerBias = weights[1];
    k = 0;
    for(j = filterIndex; j < filterIndex + 576; j++){
      convKernel[k] = layerWeight[j];
      k++;
    }
    currFeatureMap = backConv(backReluOut, convKernel, 224, 224, 64, 1);
    for(m = 0; m < 224*224; m++){
      backConvOut[n] = currFeatureMap[m];
      n++;
    }
    free(currFeatureMap);
    filterIndex = filterIndex + 576;
  }
  free(backReluOut);
  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);
  layerIDflag = 1;
  printf("backProp done");
  }

  return backConvOut;
}

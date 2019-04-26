#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// IMPLEMENTATION OF VGG16 IN C:
typedef struct layerBlock{
  float* blockOutput;
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
  LayerBias = (float**)malloc(sizeof(float*)*2);
  if((weightsFile = fopen(layerFile, "r")) == NULL){
    printf("Content file not found!");
    return NULL;
  }
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
  fscanf(weightsFile, "%d", &length);
  getBias = (float*)malloc(sizeof(float)*length);

  for(i = 0; i < length; i++){
    fscanf(weightsFile, "%f", &val);
    getBias[i] = val;
  }
  LayerBias[1] = getBias;
  fclose(weightsFile);
  return LayerBias;
}

float* matrixIndexMultiplier(float* matrix1, float* matrix2, int h1, int w1, int h2, int w2, int depth){
  int i, j, k;
  if(w1 != h2){
    return NULL;
  }
  float* result = (float*)malloc(sizeof(float)*(h1*w2*depth));

  for(k = 0; k < depth; k++){
    for(i = 0; i < h1; i++){
      for(j = 0; j < w2; j++){
          result[i + j * h1 + k*h1*w2] = matrix1[i + j * h1 + k*h1*w2] * matrix2[i + j * h1 + k*h1*w2];
        }
      }
  }

  return result;
}

float sum(float* input, int size){
  float sum = 0.0;
  unsigned i;
  for(i = 0; i < size; i++){
    sum += input[i];
  }
  return sum;
}

float* preprocess(char* inputImageFile){
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

float* convFilter(float* input_image, float* weight, float bias, int rows, int cols, int depth, int stride){
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
          input_image_padded[j + i*(cols+2) + k*(rows+2)*(cols+2)] = input_image[(j-1) + (i-1)*cols + k*(rows)*(cols)];
        }
        else{
          input_image_padded[j + i*(cols+2) + k*(rows+2)*(cols+2)] = 0;
        }
      }
    }
  }

  float* filtered_image = (float*)malloc(sizeof(float)*(rows*cols));
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

  free(patch);
  free(input_image_padded);
  return filtered_image;
}

float* relu(float* x, int size){
  int i;
  for(i = 0; i < size; i++){
    if(x[i] < 0)
      x[i] = 0.0;
  }
  return x;
}

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
            }
          }
      }
    }
  }
  return dX;
}


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

float* rotateFilter(float* filter){
  float temp;
  int i, j;
  float* newFilter;
  newFilter = (float*)malloc(sizeof(float)*9);
  for (i = 0; i < 3 / 2; i++) {
      for (j = i; j < 3 - i - 1; j++) {
          temp = filter[i*3+j];
          filter[i*3+j] = filter[(2 - j)*3+i];
          filter[(2 - j)*3+i] = filter[(2 - i)*3+(2 - j)];
          filter[(2 - i)*3+(2 - j)] = filter[j*3+(2 - i)];
          filter[j*3+(2 - i)] = temp;
      }
  }


  for (i = 0; i < 3 / 2; i++) {
      for (j = i; j < 3 - i - 1; j++) {
          temp = filter[i*3+j];
          filter[i*3+j] = filter[(2 - j)*3+i];
          filter[(2 - j)*3+i] = filter[(2 - i)*3+(2 - j)];
          filter[(2 - i)*3+(2 - j)] = filter[j*3+(2 - i)];
          filter[j*3+(2 - i)] = temp;
      }
  }

  for(i = 0; i < 9; i++){
    newFilter[i] = filter[i];
  }
  return newFilter;
}

float* backConv(float* dL, float* filter, int stride, int rows, int cols, int depth){
  float* rotatedFilter;
  rotatedFilter = rotateFilter(filter);
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

  float* filtered_image = (float*)malloc(sizeof(float)*(rows*cols));
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
        float* matrixmult = matrixIndexMultiplier(patch, filter, 3, 3, 3, 3, depth);
        filtered_image[index] = sum(matrixmult, 3*3*depth);
        index++;
        free(matrixmult);
        z++;
      }
    }
  }
  free(patch);
  free(input_image_padded);
  return filtered_image;
}

layerBlock* createVGG16(float* inputImage){
  layers layerList;
  layerBlock* layerArray;
  float* prevImage;
  float** weights;
  float* layerWeight;
  float* layerBias;
  int i, j, k, m, n, filterIndex;
  float* featureMap;
  float* currFeatureMap;
  float* convKernel;
  float* newFeatureMap;
  layerArray = (layerBlock*)malloc(sizeof(layerBlock)*13);

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

  featureMap = relu(featureMap, 224*224*64);
  layerArray[0].blockOutput = featureMap;
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

  free(weights[0]);
  free(weights[1]);
  free(weights);
  free(convKernel);

  newFeatureMap = relu(newFeatureMap, 224*224*64);
  float* pooledOutput;
  pooledOutput = maxpool(newFeatureMap, 2, 224, 224, 64);
  free(newFeatureMap);
  layerArray[1].blockOutput = pooledOutput;
  layerArray[1].height = 112;
  layerArray[1].width = 112;
  layerArray[1].depth = 64;


  // CONVOLUTION LAYER 2
  // Block 1
  float* pooledOutput2;
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
    free(weights[0]);
    free(weights[1]);
    free(weights);
    free(convKernel);

    featureMap = relu(featureMap, 112*112*128);
    layerArray[2].blockOutput = pooledOutput;
    layerArray[2].height = 112;
    layerArray[2].width = 112;
    layerArray[2].depth = 128;

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
       currFeatureMap = convFilter(featureMap, convKernel, layerBias[i], 112, 112, 128, 1);
       for(m = 0; m < 112*112; m++){
         newFeatureMap[n] = currFeatureMap[m];
         n++;
       }
       free(currFeatureMap);
       filterIndex = filterIndex + 1152;
     }
     free(weights[0]);
     free(weights[1]);
     free(weights);
     free(convKernel);

  newFeatureMap = relu(newFeatureMap, 112*112*128);
  pooledOutput2 = maxpool(newFeatureMap, 2, 112, 112, 128);
  free(newFeatureMap);
  layerArray[3].blockOutput = pooledOutput2;
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
    free(weights[0]);
    free(weights[1]);
    free(weights);
    free(convKernel);
    featureMap = relu(featureMap, 56*56*256);
    layerArray[4].blockOutput = featureMap;
    layerArray[4].height = 56;
    layerArray[4].width = 56;
    layerArray[4].depth = 256;

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
      free(weights[0]);
      free(weights[1]);
      free(weights);
      free(convKernel);

    newFeatureMap = relu(newFeatureMap, 56*56*256);
    layerArray[5].blockOutput = newFeatureMap;
    layerArray[5].height = 56;
    layerArray[5].width = 56;
    layerArray[5].depth = 256;

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
      free(weights[0]);
      free(weights[1]);
      free(weights);
      free(convKernel);


      featureMap = relu(featureMap, 56*56*256);
      pooledOutput3 = maxpool(featureMap, 2, 56, 56, 256);
      free(featureMap);
      layerArray[6].blockOutput = pooledOutput3;
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
      free(weights[0]);
      free(weights[1]);
      free(weights);
      free(convKernel);


      featureMap = relu(featureMap, 28*28*512);
      layerArray[7].blockOutput = featureMap;
      layerArray[7].height = 28;
      layerArray[7].width = 28;
      layerArray[7].depth = 512;

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
        free(weights[0]);
        free(weights[1]);
        free(weights);
        free(convKernel);

        newFeatureMap = relu(newFeatureMap, 28*28*512);
        layerArray[8].blockOutput = newFeatureMap;
        layerArray[8].height = 28;
        layerArray[8].width = 28;
        layerArray[8].depth = 512;

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
          free(weights[0]);
          free(weights[1]);
          free(weights);
          free(convKernel);

          featureMap = relu(featureMap, 28*28*512);
          pooledOutput4 = maxpool(featureMap, 2, 28, 28, 512);
          free(featureMap);
          layerArray[9].blockOutput = pooledOutput4;
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
          free(weights[0]);
          free(weights[1]);
          free(weights);
          free(convKernel);

          featureMap = relu(featureMap, 14*14*512);
          layerArray[10].blockOutput = featureMap;
          layerArray[10].height = 14;
          layerArray[10].width = 14;
          layerArray[10].depth = 512;

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
            free(weights[0]);
            free(weights[1]);
            free(weights);
            free(convKernel);

          newFeatureMap = relu(newFeatureMap, 14*14*512);
          layerArray[11].blockOutput = newFeatureMap;
          layerArray[11].height = 14;
          layerArray[11].width = 14;
          layerArray[11].depth = 512;

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
            layerArray[12].blockOutput = pooledOutput5;
            layerArray[12].height = 7;
            layerArray[12].width = 7;
            layerArray[12].depth = 512;
            // float* backMaxOut;
            // float* backReluOut;
            // float* backConvOut;
            // backMaxOut = backMax(pooledOutput5, pooledOutput5, featureMap, 2, 14, 14, 512);
            // backReluOut = backRelu(backMaxOut, reluSave, 14, 14, 512);
            // weights = getWeights("convlayer5_3.txt");
            // layerWeight = weights[0];
            // backConvOut = backConv(backReluOut, layerWeight, 2, 14, 14, 512);
            // free(backMaxOut);
             return layerArray;

}

// STYLE TRANSFER
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include "VGG16_net.h"
// COST FUNCTION
// used for both style and content loss
float meanSquaredError(float* tensor1, float* tensor2, int h, int w, int d){
	float reducedSum = 0.0;
	float matSub = 0.0;
	int i = 0;
	int numElements = h*w*d;

  	for(i = 0; i < numElements; i++){
  		matSub = (tensor1[i]-tensor2[i])*(tensor1[i]-tensor2[i]);
  		reducedSum += matSub;
  	}

  	return reducedSum / numElements;
}

// GRAM MATRIX FUNCTIONS FOR STYLE LOSS
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

float* gramMatrix(float* tensor, int h, int w, int d){
  float* transpose = (float*)malloc(sizeof(float)*(h*w*d));
  for(int i = 0; i < (h*w); i++){
    for(int j = 0; j < d; j++){
      transpose[j*h*w + i] = tensor[i*d +j];
    }
  }
  float* gram = matrixMultiplier(tensor, transpose, h*w, d, d, h*w);
  // no need for transpose
  free(transpose);
  // return gramMatrix!
  return gram;
}

// // CONTENT LOSS FUNCTION
// float contentLoss(layers* contentLayers, layers* mixedLayers, int* layerIds, int idSize){
  // float totalLoss = 0.0;
  // for(int i = 0; i < idSize; i++){
  //   totalLoss += meanSquaredError(mixedLayers[layerIds[i]], contentLayers[layerIds[i]],
  //      mixedLayers[layerIds[i]].h, mixedLayers[layerIds[i]].w, mixedLayers[layerIds[i]].d);
  // }
  // totalLoss /= idSize;
  // return totalLoss;
// }
//
// // BACKPROP FOR CONTENT LOSS FUNCTION
// // returns a malloced dE with size of the input image tensor (calls other backProp functions)
// float* backContent(layers* contentLayers, layers*mixedLayers, int* layerIds, int idSize){
//   float* dL = (float*)malloc(sizeof(float)*(224*224*64));
//   float* dLTemp;
//   float* dLTempBacked;
//   float mixVal = 0.0;
//   float contentVal = 0.0;
//   int size = 0;
//   for(int i = 0; i < idSize; i++){
//     // size of tensor output from specified layer
//     size = mixedLayers[layerIds[i]].h *
//        mixedLayers[layerIds[i]].w * mixedLayers[layerIds[i]].d);
//
//     // gradient of MSE function
//     dLTemp = (float*)malloc(sizeof(float)*(size));
//     for(int x = 0; x < size; x++){
//       mixVal = mixedLayers[layerIds[i]].tensor[x];
//       contentVal = contentLayers[layerIds[i]].tensor[x];
//       dLTemp[x] *= (2.0*(mixVal - contentVal)/size);
//     }
//
//     // sum rule for partial derivatives
//     dLTempBacked = backPropLayer(layerIds, dLTemp);
//     // don't need dLTemp anymore
//     free(dLTemp);
//
//     // sum rule for partial derivatives
//     for(int k = 0; k < (224*224*64); k++){
//       dL[k] += dLTempBacked[k];
//     }
//     // don't need dLTempBacked
//     free(dLTempBacked);
//   }
//   return dL;
// }

// STYLE LOSS FUNCTION
float styleLoss(layers* styleLayers, layers* mixedLayers, int* layerIds, int idSize){
  float totalLoss = 0.0;
  for(int i = 0; i < idSize; i++){
    totalLoss += meanSquaredError(mixedLayers[layerIds[i]], contentLayers[layerIds[i]],
       mixedLayers[layerIds[i]].h, mixedLayers[layerIds[i]].w, mixedLayers[layerIds[i]].d);
  }
  totalLoss /= idSize;
  return totalLoss;
}
// main function
int main(){
  
  return 0;
}

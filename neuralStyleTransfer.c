// STYLE TRANSFER
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "vgg16_header.h"
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
// float* matrixMultiplier(float* matrix1, float* matrix2, int h1, int w1, int h2, int w2){
//
//   // declarations
//   int i, j, k;
//   float sum;
//   sum = 0;
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
//       for(k = 0; k < h2; k++){
//         sum = sum + matrix1[(i*w1)+k]*matrix2[(k*w2)+j];
//       }
//       //printf("%f ", sum);
//       result[i*w2+j] = sum;
//       sum = 0;
//     }
//   }
//
//   return result;
// }

float* gramMatrix(float* tensor, int h, int w, int d){
  float* transpose = (float*)malloc(sizeof(float)*(h*w*d));
  for(int i = 0; i < (h*w); i++){
    for(int j = 0; j < d; j++){
      transpose[j*h*w + i] = tensor[i*d +j];
    }
  }
  float* gram = matrixMultiplier(transpose, tensor, d, h*w, h*w, d);
  // no need for transpose
  free(transpose);
  // return gramMatrix!
  return gram;
}

// CONTENT LOSS FUNCTION
float contentLoss(layerBlock* contentLayers, layerBlock* mixedLayers, int* layerIds, int idSize){
  float totalLoss = 0.0;
  for(int i = 0; i < idSize; i++){
    totalLoss += meanSquaredError(contentLayers[layerIds[i]].blockOutput, mixedLayers[layerIds[i]].blockOutput,
       mixedLayers[layerIds[i]].height, mixedLayers[layerIds[i]].width, mixedLayers[layerIds[i]].depth);
  }
  totalLoss /= idSize;
  return totalLoss;
}

// BACKPROP FOR CONTENT LOSS FUNCTION
// returns a malloced dE with size of the input image tensor (calls other backProp functions)
float* backContent(layerBlock* contentLayers, layerBlock*mixedLayers, int* layerIds, int idSize, float local){
	// changed this to 224*224*3
  float* dL = (float*)malloc(sizeof(float)*(224*224*3));
  for(int i = 0; i < 224*224*3; i++){
    dL[i] = 0;
  }
  float* dLTemp;
  float* dLTempBacked;
  float mixVal = 0.0;
  float contentVal = 0.0;
  int size = 0;
  for(int i = 0; i < idSize; i++){
    // size of tensor output from specified layer
    size = mixedLayers[layerIds[i]].height *
       mixedLayers[layerIds[i]].width * mixedLayers[layerIds[i]].depth;

    // gradient of MSE function
    dLTemp = (float*)malloc(sizeof(float)*(size));
    for(int x = 0; x < size; x++){
      mixVal = mixedLayers[layerIds[i]].blockOutput[x];
      contentVal = contentLayers[layerIds[i]].blockOutput[x];
			// changed this to mixVal - contentVal because thats how it is in styletrans.py
      dLTemp[x] = local*(2.0*(mixVal - contentVal)/size);
    }

    // sum rule for partial derivatives
    dLTempBacked = backPropLayer(dLTemp, mixedLayers, layerIds[i]);
    // don't need dLTemp anymore
    free(dLTemp);

    // sum rule for partial derivatives
		// changed the bound to 224*224*3
    for(int k = 0; k < (224*224*3); k++){
      dL[k] += dLTempBacked[k];
    }
    // don't need dLTempBacked
    free(dLTempBacked);
  }
  return dL;

}

// STYLE LOSS FUNCTION
float styleLoss(layerBlock* styleLayers, layerBlock* mixedLayers, int* layerIds, int idSize){
	float totalLoss = 0.0;
  int h, w, d;
  for(int i = 0; i < idSize; i++){
    h = mixedLayers[layerIds[i]].height;
    w = mixedLayers[layerIds[i]].width;
    d = mixedLayers[layerIds[i]].depth;
    float* gramMixed = gramMatrix(mixedLayers[layerIds[i]].blockOutput, h, w, d);
    float* gramStyle = gramMatrix(styleLayers[layerIds[i]].blockOutput, h, w, d);
    // the reason there's a one for height is bc meanSquaredError expects a 3D tensor
    // but it is receiving a 2D gram matrix
    totalLoss += meanSquaredError(gramStyle, gramMixed,1, d, d);
		// need to free both gramMatricies
    free(gramMixed);
    free(gramStyle);
  }
  totalLoss /= idSize;
  return totalLoss;
}
// BACKPROP (TF GRADIENTS VERSION) FOR GRAM MATRIX
float* backMult(float* matrix1, int h1, int w1, int h2, int w2){

  // declarations
  int i, j, k;
  // num columns of first matrix must equal num rows of second matrix
  if(w1 != h2){
    return NULL;
  }
  // result will have same num rows as first matrix, same num columns as second
  float* result = (float*)malloc(sizeof(float)*(h2*w2));
  for(int x = 0; x < (h2*w2); x++)
    result[x] = 0;
  // multiply matrices
  for(i = 0; i < h1; i++){
    for(j = 0; j < w2; j++){
      for(k = 0; k < h2; k++){
        result[(k*w2)+j] += matrix1[(i*w1)+k];
      }
    }
  }

  return result;
}

// BACKPROP FOR STYLE
float* backStyle(layerBlock* styleLayers, layerBlock*mixedLayers, int* layerIds, int idSize, float local){
	float* dL = (float*)malloc(sizeof(float)*(224*224*3));
  for(int i = 0; i < 224*224*3; i++){
    dL[i] = 0;
  }
  float* dLTemp;
  float* dLTempBacked;
  float* backGramMix;
  float* backGramStyle;
  float mixVal = 0.0;
  float styleVal = 0.0;
  int size = 0;
  int h,w,d;
  for(int i = 0; i < idSize; i++){
    // size of tensor output from specified layer
    h = mixedLayers[layerIds[i]].height;
    w = mixedLayers[layerIds[i]].width;
    d = mixedLayers[layerIds[i]].depth;
    size = h * w * d;
    backGramMix = backMult(mixedLayers[layerIds[i]].blockOutput,d, h*w, h*w, d);
    backGramStyle = backMult(styleLayers[layerIds[i]].blockOutput,d, h*w, h*w, d);
    // gradient of MSE function
    dLTemp = (float*)malloc(sizeof(float)*(size));
    for(int x = 0; x < size; x++){
      mixVal = backGramMix[x];
      styleVal = backGramStyle[x];
      dLTemp[x] *= local*(2.0*(mixVal - styleVal)/size);
    }

    // sum rule for partial derivatives
    dLTempBacked = backPropLayer(dLTemp, mixedLayers, layerIds[i]);
    // don't need dLTemp anymore
    free(dLTemp);

    // sum rule for partial derivatives
    for(int k = 0; k < (224*224*3); k++){
      dL[k] += dLTempBacked[k];
    }
    // don't need dLTempBacked or any of the backGram matricies
    free(dLTempBacked);
    free(backGramStyle);
    free(backGramMix);
  }
  return dL;
}
// standard deviation function used in styletransfer to calculate scaled step size
float stdev(float* tensor, int size){
	float mean = 0;
	float mean_diff = 0;
	for(int i = 0; i < size; i++){
		mean += tensor[i];
	}
	mean /= size;
	for(int i = 0; i < size; i++){
		mean_diff += (mean - tensor[i])*(mean - tensor[i]);
	}
	return (float)sqrt(mean_diff);
}

// STYLE TRANSFER ALGORITHM FUNCTION
// I'm assuming we don't need np.squeeze(grad) conversions
float* styleTransfer(float* contentImage, float* styleImage, int* contentIds,
    int contentIdSize, int* styleIds, int styleIdSize, float contentWeight,
    float styleWeight, float denoiseWeight, int iterations, float stepSize){
			float content_loss, style_loss, denoiseLoss, totalLoss, localStyle, localContent, scaledStep;
      float* gradContent;
      float* gradStyle;
      float gradient[224*224*3];
      // temporary until we write denoise
      denoiseLoss = 0.0;
      layerBlock* contentLayers = createVGG16(contentImage);
      layerBlock* styleLayers = createVGG16(styleImage);
      layerBlock* mixedLayers;
			printf("content and style layers created");
      // generate random image
      float* mixedImage = (float*)malloc(sizeof(float)*(224*224*3));
      srand(time(0));
      for(int x = 0; x < (224*224*3); x++)
        mixedImage[x] = rand() % 255;
      // gradient descent algorithm
      // totalLoss = contentWeight*(content_loss/(content_loss)) + styleWeight*(style_loss/style_loss)

 			printf("random image generated");
			for(int i = 0; i < iterations; i++){
        // reinput into neural network
        mixedLayers = createVGG16(mixedImage);
				printf("mixedLayers generated ");
        // recalculate loss based on this
        content_loss = contentLoss(contentLayers, mixedLayers, contentIds, contentIdSize);
        style_loss = styleLoss(styleLayers, mixedLayers, styleIds, styleIdSize);
        // weights of each loss function on gradient
        localStyle = (styleWeight/style_loss);
        localContent = (contentWeight/content_loss);
        // indpendent gradient  calculation for each loss function w/weights
        gradContent = backContent(contentLayers, mixedLayers, contentIds, contentIdSize, localContent);
        gradStyle = backStyle(styleLayers, mixedLayers, styleIds, styleIdSize, localStyle);

        // compute total gradient
        for(int x = 0; x < (224*224*3); x++){
          gradient[x] = gradStyle[x] + gradContent[x];
        }
        // create scaled step size/learning rate based on the gradient
        scaledStep = stepSize/stdev(gradient, (224*224*3));

        // update mixed image based on this (clip range of pixel values to 0-255)
        for(int x = 0; x < (224*224*3); x++){
          mixedImage[x] = fmod(fabsf((mixedImage[x] - gradient[x]*scaledStep)), 255);
        }
        // save image
        // free generated gradient vectors
        free(gradContent);
        free(gradStyle);

				// FREE THE STRUCT
				// not sure if you need to free stuff within the struct depends on how you allocated it
				for(int i = 0; i < 13; i++){
					free(mixedLayers[i].blockOutput);
					free(mixedLayers[i].beforeRelu);
					free(mixedLayers[i].beforePool);
					free(mixedLayers[i].beforeConv);
				}
				free(mixedLayers);
				printf("Iteration %d\n done", i);
      }
			// return image
      return mixedImage;
    }
// main function
int main(){
	float* afterTenIterations;
	float* contentImage;
	float* styleImage;
	int contentIds[4] = {3};
	int styleIds[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	contentImage = preprocess("content.txt");
	styleImage = preprocess("style.txt");
	afterTenIterations = styleTransfer(contentImage, styleImage, contentIds,
	    1, styleIds, 13, 1.5, 10.0, 0.3, 10, 10.0);

	FILE* imageAfter;
	if((imageAfter = fopen("result.txt", "w")) == NULL){
	  printf("Content file not found!");
	}

	int idxr, idxc;
	for(idxr = 0; idxr < (244*3); idxr++){
		for(idxc = 0; idxc < 224; idxc++){
			fprintf(imageAfter, "%f", afterTenIterations[idxr*224+idxc]);
		}
		fprintf(imageAfter, "\n");
	}

	fclose(imageAfter);
  return 0;
}

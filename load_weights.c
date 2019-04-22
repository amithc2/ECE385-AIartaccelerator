#include <stdio.h>
#include <stdlib.h>

float** getWeights(char* layerFile){
  // variable declarations

  FILE* weightsFile;
  float* getLayer;
  float* getBias;
  float** LayerBias;
  float val;
  int dimVal1, dimVal2, dimVal3, dimVal4;
  int i, length;

  // return both the getLayer and getBias
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
  return LayerBias;
}

int main(){
  float** layerAndBias;
  float* getLayer;
  float* getBias;
  layerAndBias = getWeights("convlayer1_1.txt");
  int i, j, length;
  length = 64;

  getLayer = layerAndBias[0];
  getBias = layerAndBias[1];

  for(i = 0; i < 27*length; i++){
      printf("%f ", getLayer[i]);
  }
  for(i = 0; i < length; i++){
      printf("%f ", getBias[i]);
  }
  free(layerAndBias[0]);
  free(layerAndBias[1]);
  free(layerAndBias);

  
    // int dimVal1, dimVal2, dimVal3, dimVal4, bleh;
    // FILE* weightsFile;
    // if((weightsFile = fopen("convlayer1_1.txt", "r")) == NULL){
    //   printf("Content file not found!");
    //   return 0;
    // }
    //
    // fscanf(weightsFile, "%d", &dimVal1);
    // fscanf(weightsFile, "%d", &dimVal2);
    // fscanf(weightsFile, "%d", &dimVal3);
    // fscanf(weightsFile, "%d", &dimVal4);
    // fscanf(weightsFile, "%d", &bleh);
    // printf("dimVal1 is %d", dimVal1);
    // printf("dimVal2 is %d", dimVal2);
    // printf("dimVal3 is %d", dimVal3);
    // printf("dimVal4 is %d", dimVal4);
    // printf("bleh is %d", bleh);
}

#include <stdio.h>
#include <stdlib.h>

float* getWeights(){
  FILE* weightsFile;
  float* firstConvLayer;
  float val;
  int i,length;
  length = 27*64;
  firstConvLayer = (float*)malloc(sizeof(float)*length);
  if((weightsFile = fopen("convlayer1_1.txt", "r")) == NULL){
    printf("Content file not found!");
    return NULL;
  }

  for(i = 0; i < length; i++){
    fscanf(weightsFile, "%f", &val);
    firstConvLayer[i] = val;
  }

  return firstConvLayer;
}

int main(){
  float* firstLayer;
  firstLayer = getWeights();
  int i, j, length;
  length = 27*64;

  for(i = 0; i < length; i++){
    printf("\n");
    for(j = 0; j < 3; j++){
      printf("%f ", firstLayer[i]);
    }
  }
  free(firstLayer);
}

#include <stdio.h>
#include <stdlib.h>



// Structure to hold the content tensor and value tensor 
typedef struct vgg16{
	float* contentTensor;
	float* valueTensor;
	int height;
	int width;
	int depth;
} tensorHolder;


// Flatten tensors from 3d to 1d 
float* flatten(float*** tensor, int height, int width, int depth){

	// dynamically allocate memory for flattenedArray
	float* flattenedArray;
	flattenedArray = malloc(sizeof(tensor));

	// index 3d array to 1d array with following method
	// Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
	int i = 0;
	int j = 0;
	int k = 0;
	int flatIdx;
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			for(k = 0; k < depth; k++){
				flatIdx = i + width * (j + depth * k);
				flattenedArray[flatIdx] = tensor[i][j][k];
			}
		}
	}

	return flattenedArray;
}



// Calculate the mean squared error given two 1d tensors
float meanSquaredError(float* tensor1, float* tensor2){
	float reducedSum = 0.0;
  	float matSub = 0.0;
  	int i = 0;
  	int numElements = 0;

  	for(i = 0; i < sizeof(tensor1); i++){
  		matSub = (tensor1[i]-tensor2[i])*(tensor1[i]-tensor2[i]);
  		reducedSum += matSub;
  		numElements++;
  	}

  	return reducedSum / numElements;
}


// Function to calculate total content loss
float createContentLoss(tensorHolder* layers){

	float val;
	float lossVal;
	float totalLoss;

	// instantiate layer losses array
	float layerLosses[sizeof(layers)];

	// calculate content loss
	int i;
	for(i = 0; i < sizeof(layers); i++){
		lossVal = meanSquaredError((layers[i]).contentTensor,(layers[i]).valueTensor);
		layerLosses[i] = lossVal;
	}

	totalLoss = layerLosses[0];
	return totalLoss;
}

// Multiply tensor with its transpose to produce gram matrix
//float* createGramMatrix(float* tensor){

//}

// Function to calculate total style loss
//float createStyleLoss(tensorHolder model){

//}

// Function to calculate gradient loss
//float calculateGradientLoss(){

//}


int main(){
	float arr[2][2][2];
	arr[0][0][0] = 2.3;
	arr[0][0][1] = 4.6;
	arr[0][1][0] = 9.2;
	arr[1][0][0] = 4.1;
	arr[0][1][1] = 8.7;
	arr[1][1][0] = 6.5;
	arr[1][0][1] = 3.4;
	arr[1][1][1] = 1.2;

	float* flatBoi;
	//flatBoi = flatten(arr, 2, 2, 2);
	free(flatBoi);
}

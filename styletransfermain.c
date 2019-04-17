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



void getContentTensor(){

	int temp = 0;

	// get dimensions of tensor
	int notUsed = 0;
	int h = 0;
	int w = 0;
	int d = 0;

	char discard[256];

	FILE* contentFile;
	if((contentFile = fopen("content.txt","r")) == NULL){
		printf("Content file not found!");
		exit(1);
	}

	fscanf(contentFile, "%d", &notUsed);
	fscanf(contentFile, "%d", &h);
	fscanf(contentFile, "%d", &w);
	fscanf(contentFile, "%d", &d);
	
	// create tensor
	float* tensor;
	tensor = malloc(sizeof(h*w*d));

	// Discard the next line using a buffer
	fgets(discard, 50, content_file);

	// Iterate through file and get tensor values
	for(int i = 0; i < h*w*d; i++){
		fscanf(contentFile, "%f", &temp);
		tensor[i] = temp;
	}
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

	float lossVal = 0;
	float totalLoss = 0;

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

}

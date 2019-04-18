#include <stdio.h>
#include <stdlib.h>

struct vgg16{
	float* layer_tensor;
	float* value_tensor;
	int h, w,d;
};

float* getTensor(int h, int w, int d, FILE* contentFile){
	int i;
	float temp = 0;
	char discard[256];

	// create tensor
	float* tensor;
	int length = h*w*d;
	tensor = malloc(sizeof(float)*length);
	// Discard the next line using a buffer
	fscanf(contentFile, "%s", discard);
	// Iterate through file and get tensor values
	for(i = 0; i < length; i++){
		fscanf(contentFile, "%f", &temp);
		tensor[i] = temp;
	}

	return tensor;
}
struct vgg16* makeLayers(char* layerfilename, char* valuefilename, int size){
	struct vgg16* layers = (struct vgg16*)malloc(sizeof(struct vgg16)*size);
	FILE* layerfile;
	FILE* valuefile;
	long int LayerOffset;
	long int ValueOffset;
	int notUsed, h, w, d;
	if((layerfile = fopen(layerfilename,"r")) == NULL){
		printf("Content file not found!");
		return NULL;
	}
	if((valuefile = fopen(valuefilename,"r")) == NULL){
		printf("Content file not found!");
		return NULL;
	}
	for(int i = 0; i < size; i++){

		fscanf(layerfile, "%d", &notUsed);
		fscanf(layerfile, "%d", &h);
		fscanf(layerfile, "%d", &w);
		fscanf(layerfile, "%d", &d);

		layers[i].h = h;
		layers[i].w = w;
		layers[i].d = d;

		// printf("Dimensions are %d %d %d %d\n", notUsed, layers[i].h, layers[i].w, layers[i].d);

		layers[i].layer_tensor = getTensor(h, w, d, layerfile);
		// printf("%f\n", layers[i].layer_tensor[0]);

		fscanf(valuefile, "%d", &notUsed);
		fscanf(valuefile, "%d", &h);
		fscanf(valuefile, "%d", &w);
		fscanf(valuefile, "%d", &d);

		// printf("Dimensions are %d %d %d %d\n", notUsed, h, w, d);

		layers[i].value_tensor = getTensor(h, w, d, valuefile);
		ValueOffset = ftell(valuefile);
		if(fseek(valuefile, ValueOffset, 0)){
			exit(1);
		}
		LayerOffset = ftell(layerfile);
		if(fseek(layerfile, LayerOffset, 0)){
			exit(1);
		}
	}
	return layers;
}
void deleteLayers(struct vgg16* layers, int size){
	for(int i = 0; i < size; i++){
		free(layers[i].layer_tensor);
		free(layers[i].value_tensor);
	}
	free(layers);
}

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

// Function to calculate total content loss
float createContentLoss(struct vgg16* layers){

	float lossVal = 0;
	float totalLoss = 0;

	// instantiate layer losses array
	float layerLosses[sizeof(layers)/sizeof(struct vgg16)];

	// calculate content loss
	int i;
	for(i = 0; i < sizeof(layers)/sizeof(struct vgg16); i++){
		lossVal = meanSquaredError((layers[i]).layer_tensor,(layers[i]).value_tensor, layers[i].h, layers[i].w, layers[i].d);
		layerLosses[i] = lossVal;
	}

	totalLoss = layerLosses[0];
	return totalLoss;
}

float* createGramMatrix(float* tensor, int h, int w, int d){
	// create transpose of tensor
	float transpose[h*w*d];
	// have to malloc result, make sure to delete in style layer function
	int rows = h*w;
	float* result = (float*)malloc(sizeof(float)*(rows*rows));


	int i, j, k;
	// cols == d
	for(i = 0; i < rows; i++){
		for(j = 0; j < d; j++){
			transpose[j*rows + i] = tensor[i*d +j];
		}
	}

	// multiply the 2D tensors and store result (THIS IS CURRENTLY CAUSING SEGFAULT)
	for(i = 0; i < rows; i++){
		for(j = 0; j < rows; j++){
			result[rows*i + j] = 0;
			for(k = 0; k < d; k++){
				result[rows*i + j] += tensor[d*i + k] * transpose[rows*k + j];
			}
		}
	}

	//return result
	return result;
}

float createStyleLoss(struct vgg16* layers){

	float lossVal = 0;
	float totalLoss = 0;

	// instantiate gramLayers
	float* gramLayers[sizeof(layers)/sizeof(struct vgg16)];

	// instantiate layerLosses array
	float layerLosses[sizeof(layers)/sizeof(struct vgg16)];

	// find gramLayers
	int i;
	for(i = 0; i < sizeof(layers)/sizeof(struct vgg16); i++){
		gramLayers[i] = createGramMatrix(layers[i].layer_tensor, layers[i].h, layers[i].w, layers[i].d);
	}


	// calculate style loss

	for(i = 0; i < sizeof(layers)/sizeof(struct vgg16); i++){
		lossVal = meanSquaredError(gramLayers[i],(layers[i]).value_tensor, layers[i].h, layers[i].w, layers[i].d);
		layerLosses[i] = lossVal;
	}


	// reduceMean of layerLosses array
	float sum = 0;
	int numElements = 0;
	for(i = 0; i < sizeof(layerLosses); i++){
		sum += layerLosses[i];
		numElements++;
	}

	totalLoss = sum / numElements;
	return totalLoss;

}

int main(){
	struct vgg16* styles = makeLayers("stylelayer.txt", "stylevalue.txt", 13);
	struct vgg16* content = makeLayers("contentlayer.txt", "contentlayer.txt", 1);
	printf("The float value returned by createContentLoss: %f\n", createContentLoss(content));
	float* gram = createGramMatrix(styles[0].layer_tensor, styles[0].h, styles[0].w, styles[0].d);
	for(size_t i = 0; i < styles[0].h*styles[0].d*styles[0].w; i++)
		printf("The gram matrix: %f\n", gram[i]);
	free(gram);
	deleteLayers(styles, 13);
	deleteLayers(content, 1);

	// free(contentLayerTensor1);
	// free(contentValueTensor1);
	// free(styleLayerTensor1);
	// free(styleValueTensor1);

	return 0;
}

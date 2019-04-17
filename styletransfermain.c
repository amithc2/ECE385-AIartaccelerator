
// Structure to hold the content tensor and value tensor 
typedef struct vgg16{
	float* contentTensor;
	float* valueTensor;
	int height;
	int width;
	int depth;
} tensorHolder;


// Flatten tensors from 3d to 1d 
float* flatten(float*** tensor){

}



// Calculate the mean squared error given two 1d tensors
float meanSquaredError(float* tensor1, float* tensor2){

}


// Function to calculate total content loss
float createContentLoss(tensorHolder model){

}

// Multiply tensor with its transpose to produce gram matrix
float* createGramMatrix(float* tensor){

}

// Function to calculate total style loss
float createStyleLoss(tensorHolder model){

}

// Function to calculate gradient loss
float calculateGradientLoss(){

}


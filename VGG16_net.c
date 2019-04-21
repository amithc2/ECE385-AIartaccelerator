#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// IMPLEMENTATION OF VGG16 IN C:



// GARBAGE MULTIPLE PLS DELETE
float* multiply(float* a, float* b, int h1, int w1, int h2, int w2){
  float* result = (float*)malloc(sizeof(float)*(w1*h2));
  for(int j = 0; j < w1*h2; j++){
    result[0] = 0.0;
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
float* preprocess(float* im){
  // RGB2BGR macro
  // Resize image
  // subtract vgg mean
  return NULL;
}

// here is the general description of the stack of conv layers
// all conv layers are  3x3 in  the vgg16 framework
// here is some psuedo code:
/*
* float* conv_layer(array type image, array type weight){
    int m = length of first dimension of image (ex.224) SUBTRACTED BY 2
    int n = length of second dimension of image (ex.224) SUBTRACTED BY 2

    float* new_image[m, n] = {0}->this isn't actually how you do it but just initializer list of zeroes
    float* patch = temporary image patch that is 3x3
    for(int i = 0; i < m; i++){
      for(int j = 0; j < n; j++){
        patch = im[i to i+3, j to  j+3] literally just hand write this its not that hard
        new_image[i, j] = sum(multiply(patch, weight)) -> once we  finish the multiply func we just call  that and sum up indicies
      }
    }
    return new_image;
  }
*/
// assuming the weights are going to be 3x3 filters
float* conv_layer(float* input_image, float* weight, int h, int w){
  int m = w - 2;
  int n = h - 2;
  int i, j, a, b;
  float* filtered_image = (float*)malloc(sizeof(float)*(m*n));
  float patch[m*n];
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      for(a = 0; a < 3; a++)
        for(b = 0; b < 3; b++)
          patch[i*(m+a) + j + b] = input_image[i*(m+a) + j*b];
      filtered_image[i*m + j] = sum(multiply(patch, weight, h, w, 3, 3), 9);
    }
  }
  return filtered_image;
}

// ReLU to introduce non linearity after convolutional layer
// this should  be  a simple single for loop:
/*
  if  we   pass by pointer void:
  void ReLU(float* x){
    for(int i = 0; i < x.length or smt; i++){
      if(x[i] < 0)
        x[i] = 0;
    }
  }
*/
void relu(float* x, int size){
  for(int i = 0; i < size; i++)
    if(x[i] < 0)
      x[i] = 0.0;
}

// you might think we should use average pooling instead of max pooling for this use
// and you are absolutely  right, but we just didn't do it here because of time constraints on
// this project. We will use pre-trained weights for this so we don't want avg pooling to backfire
// Here is some psuedo  code:
/*
  float* maxpool(float* x, int stride){
    for(int i = 0; i < x.length rows or whatever; i++){
      for(int j = 0; j < x cols; i++){
        x[]
      }
    }
  }
*/

// softmax : this is used in the last layer for vgg16
// Here is some psuedo code:
/*
  float* softmax(float* x){
    float softmax_sum = 0.0;
    float* softmax_result = malloc(sizeof(x));
    for(size_t i = 0; i < sizeof(x)/sizeof(float); i++){
      softmax_result[i] = expf(x[i]);
      softmax_sum+=softmax_result[i];
    }
    then divide
  }
*/
float* softmax(float* x, int size){
  float softmax_sum = 0.0;
  float* softmax_result = malloc(sizeof(x)*size);
  for(int i = 0; i < size; i++){
    softmax_result[i] = expf(x[i]);
    softmax_sum += softmax_result[i];
  }
  for(int i = 0; i < size; i++){
    softmax_result[i] /= softmax_sum;
  }
  return softmax_result;
}

// MAIN FUNCTION IS BEING USED SIMPLY FOR TESTING PLEASE REMOVE LATER:
int main(){
  //  test for relu helper function
  float test[4] = {-.54, 54.6, 67.3, -.34};
  relu(test, 4);
  for(int i=0; i < 4; i++)
    printf("%f\n", test[i]);
  // test for sum helper function
  float test_sum = sum(test, 4);
  printf("(%f)\n", test_sum);
  // test for softmax
  float softmax_test[6] = { 8, 14, 16,  8, 14,  1};
  float* softmax_result_test = softmax(softmax_test, 6);
  for(int i=0; i < 6; i++)
    printf("%f\n", softmax_result_test[i]);

  return 0;
}

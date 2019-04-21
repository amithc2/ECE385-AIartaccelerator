// IMPLEMENTATION OF VGG16 IN C:


// this will be the pre processing function for vgg
// resize the image to 224x224
// vgg_mean computed from the training set to preprocess images
// vgg_mean= array value of : [123.68, 116.779, 103.939] in numpy this is float data type
// subtract each pixel in the image by the vgg_mean
// next  convert  RGB  to  BGR by right shifting


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
  }
*/

// MAIN FUNCTION IS BEING USED SIMPLY FOR TESTING PLEASE REMOVE LATER:

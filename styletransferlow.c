#include <stdio.h>
#include <stdlib.h>



static float content_tensor1[75][75][256];
static float content_tensor2[75][75][256];
static float style_tensor1[19][19][512];
static float style_tensor2[298][300][64];

// currently everything is a fixed width, I'm hoping the vgg16 function pads for us
static int h, w, d;

void getContentTensor1(){
  // Shape of content tensor is 1x75x75x256


  // HOW THE TENSOR IS FORMATTED:
  // Every dimension index is accounted for in the second index so if you want
  // to access dimension index 2, e.g. content_tensor[0][2][0][0]
  // The third dimension is each of the 75 lines, and the fourth dimension is the 256 values
  // in each line
  
  char discard[256];
  float read_num;
  FILE* content_file;
  if((content_file = fopen("content.txt","r")) == NULL){
    printf("Content file not found!");
    exit(1);
  }

  // Discard the first three lines of the content file
  fgets(discard, 50, content_file);
  printf("%s\n",discard);
  fgets(discard, 50, content_file);
  printf("%s\n",discard);

  // Iterate through file and get tensor values
  for(int i = 0; i < 75; i++){
    for(int j = 0; j < 75; j++){
      for(int k = 0; k < 256; k++){
        fscanf(content_file,"%f",&read_num);
        content_tensor1[i][j][k] = read_num;
      }
    }

  }


}
void getContentTensor2(){
  // Shape of content tensor is 1x75x75x256


  // HOW THE TENSOR IS FORMATTED:
  // Every dimension index is accounted for in the second index so if you want
  // to access dimension index 2, e.g. content_tensor[0][2][0][0]
  // The third dimension is each of the 75 lines, and the fourth dimension is the 256 values
  // in each line
  
  char discard[256];
  float read_num;
  FILE* content_file;
  if((content_file = fopen("content2.txt","r")) == NULL){
    printf("Content file not found!");
    exit(1);
  }

  // Discard the first three lines of the content file
  fgets(discard, 50, content_file);
  printf("%s\n",discard);
  fgets(discard, 50, content_file);
  printf("%s\n",discard);

  // Iterate through file and get tensor values
  for(int i = 0; i < 75; i++){
    for(int j = 0; j < 75; j++){
      for(int k = 0; k < 256; k++){
        fscanf(content_file,"%f",&read_num);
        content_tensor2[i][j][k] = read_num;
      }
    }

  }


}

void getStyleTensor1(){
    // Shape of content tensor is 1x19x19x512

  // HOW THE TENSOR IS FORMATTED:
  // Every dimension index is accounted for in the second index so if you want
  // to access dimension index 2, e.g. content_tensor[0][2][0][0]
  // The third dimension is each of the 75 lines, and the fourth dimension is the 256 values
  // in each line
  
  char discard[256];
  float read_num;
  FILE* content_file;
  if((content_file = fopen("style.txt","r")) == NULL){
    printf("Content file not found!");
    exit(1);
  }

  // Discard the first three lines of the content file
  fgets(discard, 50, content_file);
  printf("%s\n",discard);
  fgets(discard, 50, content_file);
  printf("%s\n",discard);

  // Iterate through file and get tensor values
  for(int i = 0; i < 19; i++){
    for(int j = 0; j < 19; j++){
      for(int k = 0; k < 512; k++){
        fscanf(content_file,"%f",&read_num);
        style_tensor1[i][j][k] = read_num;
      }
    }

  }


}

void getStyleTensor2(){
    // Shape of content tensor is 1x298x300x64

  // HOW THE TENSOR IS FORMATTED:
  // Every dimension index is accounted for in the second index so if you want
  // to access dimension index 2, e.g. content_tensor[0][2][0][0]
  // The third dimension is each of the 75 lines, and the fourth dimension is the 256 values
  // in each line
  
  char discard[256];
  float read_num;
  FILE* content_file;
  if((content_file = fopen("style2.txt","r")) == NULL){
    printf("Content file not found!");
    exit(1);
  }

  // Discard the first three lines of the content file
  fgets(discard, 50, content_file);
  printf("%s\n",discard);
  fgets(discard, 50, content_file);
  printf("%s\n",discard);

  // Iterate through file and get tensor values
  for(int i = 0; i < 298; i++){
    for(int j = 0; j < 300; j++){
      for(int k = 0; k < 64; k++){
        fscanf(content_file,"%f",&read_num);
        style_tensor2[i][j][k] = read_num;
      }
    }

  }

}


// mean squared error funciton, which is our loss function
//
double mean_squared_error(double a_tensor [h][w][d], double b_tensor [h][w][d], int height, int width, int depth){
  //index = (i * height + j) * depth + k; if we wanna do 1D
  double reduced_sum = 0.0;
  double subtract_mul = 0.0;
  int elements = 0;
  int i = 0;
  int j = 0;
  int k = 0;
  for(i = 0; i < height; i++){
    for(j = 0; j < width; j++){
      for(k = 0; k < depth; k++){
        //a_tensor[i][j][k]
        elements++;
        subtract_mul = (a_tensor[i][j][k] - b_tensor[i][j][k])*(a_tensor[i][j][k] - b_tensor[i][j][k]);
        reduced_sum += subtract_mul;
      }
    }
  }
  return reduced_sum / elements;
}

//code to flatten 3D tensor to 2D
// double[][d] flatten_to2D(double tensor [h][w][d]){
//
// }
//
// double[][d] gram_matrix(double tensor [h][w][d]){
//   double tensor_reduced[][d] = flatten_to2D(tensor);
//   double gram[][d] = {0};
//   int i, j;
//   for(i = 0; i < w; i++){
//     for(j = 0; j < d; j++){
//       gram[i][j] = tensor_reduced[i][j]*tensor_reduced[j][i];
//     }
//   }
//   return gram;
// }

//the VGG16 model HAS to correspond to the look up table info for the content image
double create_content_loss(VGG16 model, double content_image [h][w][d], int* layer_indexes){
  // gets tensors from vgg16 net this is essentially the filters that will be applied to the content image
  //layers = model.get_layer_tensors();
  // gets value outputs from filter tensor * image tensor 

  double layer_losses[
}
// general thoughts on text file conversion: theres only two images so all we need to do is
// return layer values for those two images and we have done all the work we need to do
// essentially just run python in command line to update text files
// then run C code on NIOS II
int main(){
  h = 2;
  w = 2;
  d = 2;
  int i = 0;
  int j = 0;
  int k = 0;
  double bigmeme[h][w][d];
  double why[h][w][d];
  for(i = 0; i < h; i++){
    for(j = 0; j < w; j++){
      for(k = 0; k < d; k++){
        bigmeme[i][j][k] = 1.0;
        printf("Hopefully this is correct index : %f \n", bigmeme[i][j][k]);
        why[i][j][k] = 3.0;
        printf("Hopefully this is correct index : %f \n", why[i][j][k]);
      }
    }
  }
  double meme = mean_squared_error(bigmeme, why, h, w, d);
  printf("Hopefully this is correct : %f", meme);
  return 0;
}

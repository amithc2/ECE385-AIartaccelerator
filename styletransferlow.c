#include <stdio.h>
static int h, w, d;
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

// double[h][w][d] gram_matrix(double tensor [h][w][d]){
//
// }
// double create_content_loss(VGG16 model, double content_image [][][4],   int* layer_indexes ){
//
// }
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

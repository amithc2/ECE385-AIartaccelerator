#include <stdio.h>
#include <stdlib.h>


float* matrixMultiplier(float* matrix1, float* matrix2, int h1, int w1, int h2, int w2){

  // declarations
  int i, j, k;
  float sum;
  sum = 0;
  // num columns of first matrix must equal num rows of second matrix
  if(w1 != h2){
    return NULL;
  }
  // result will have same num rows as first matrix, same num columns as second
  float* result = (float*)malloc(sizeof(float)*(h1*w2));

  // multiply matrices
  for(i = 0; i < h1; i++){
    for(j = 0; j < w2; j++){
      for(k = 0; k < h2; k++){
        sum = sum + matrix1[(i*w1)+k]*matrix2[(k*w2)+j];
      }
      //printf("%f ", sum);
      result[i*w2+j] = sum;
      sum = 0;
    }
  }

  return result;
}


int main(){
  //width * row + col
   int i, j;

  // operands and result
  float first2d[2][3];
  float second2d[3][2];
  float first[6];
  float second[6];
  float* mult;
  // test values in matrices
  first2d[0][0] = 1;
  first2d[0][1] = 2;
  first2d[0][2] = 3;
  first2d[1][0] = 4;
  first2d[1][1] = 5;
  first2d[1][2] = 6;

  second2d[0][0] = 7;
  second2d[0][1] = 8;
  second2d[1][0] = 9;
  second2d[1][1] = 10;
  second2d[2][0] = 11;
  second2d[2][1] = 12;

  for(i = 0; i < 2; i++){
    for(j = 0; j < 3; j++){
      first[(3*i)+j] = first2d[i][j];
    }
  }

  for(i = 0; i < 3; i++){
    for(j = 0; j < 2; j++){
      second[(2*i)+j] = second2d[i][j];
    }
  }

  mult = matrixMultiplier(first, second, 2, 3, 3, 2);

  for(i = 0; i < 4; i++){
    printf("%f ", mult[i]);
  }


  free(mult);


}

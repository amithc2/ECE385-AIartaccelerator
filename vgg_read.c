#include <stdio.h>
#include <stdlib.h>



struct vgg16{
	float**** content_tensor1;
	float content_tensor2[1][75][75][256];
	float style_tensor1[1][75][75][256];
	float style_tensor2[1][75][75][256];
};
static int h, w, d;
static float* content_tensor1;
static float* content_tensor2;
static float style_tensor1[1][19][19][512];
static float style_tensor2[1][298][300][64];

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
	for(int i = 0; i < h*w*d; i++){
		fscanf(content_file, "%f", &read_num);
		content_tensor1[i] = read_num;
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
				content_tensor2[0][i][j][k] = read_num;
			}
		}

	}

	//return content_tensor;

	// These were just tests I wrote to make sure the values were stored to every
	// layer correctly

	// Test first layer

	//printf("%f\n", content_tensor[0][0][0][0]);
	//printf("%f\n", content_tensor[0][0][0][1]);
	//printf("%f\n", content_tensor[0][0][0][2]);
	//printf("%f\n", content_tensor[0][0][0][3]);


	//printf("\n");

	// Test second layer

	//printf("%f\n", content_tensor[0][0][1][3]);
	//printf("%f\n", content_tensor[0][0][2][0]);
	//printf("%f\n", content_tensor[0][0][3][0]);
	//printf("%f\n", content_tensor[0][0][9][0]);

	//printf("\n");

	// Test third layer

	//printf("%f\n", content_tensor[0][0][74][255]);
	//printf("%f\n", content_tensor[0][1][0][0]);
	//printf("%f\n", content_tensor[0][2][0][0]);
	//printf("%f\n", content_tensor[0][3][0][0]);
	//printf("%f\n", content_tensor[0][25][0][0]);


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
				style_tensor1[0][i][j][k] = read_num;
			}
		}

	}

	//return content_tensor;

	// These were just tests I wrote to make sure the values were stored to every
	// layer correctly

	// Test first layer

	//printf("%f\n", content_tensor[0][0][0][0]);
	//printf("%f\n", content_tensor[0][0][0][1]);
	//printf("%f\n", content_tensor[0][0][0][2]);
	//printf("%f\n", content_tensor[0][0][0][3]);


	//printf("\n");

	// Test second layer

	//printf("%f\n", content_tensor[0][0][1][3]);
	//printf("%f\n", content_tensor[0][0][2][0]);
	//printf("%f\n", content_tensor[0][0][3][0]);
	//printf("%f\n", content_tensor[0][0][9][0]);

	//printf("\n");

	// Test third layer

	//printf("%f\n", content_tensor[0][0][74][255]);
	//printf("%f\n", content_tensor[0][1][0][0]);
	//printf("%f\n", content_tensor[0][2][0][0]);
	//printf("%f\n", content_tensor[0][3][0][0]);
	//printf("%f\n", content_tensor[0][25][0][0]);
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
				style_tensor2[0][i][j][k] = read_num;
			}
		}

	}

	//return content_tensor;

	// These were just tests I wrote to make sure the values were stored to every
	// layer correctly

	// Test first layer

	//printf("%f\n", content_tensor[0][0][0][0]);
	//printf("%f\n", content_tensor[0][0][0][1]);
	//printf("%f\n", content_tensor[0][0][0][2]);
	//printf("%f\n", content_tensor[0][0][0][3]);


	//printf("\n");

	// Test second layer

	//printf("%f\n", content_tensor[0][0][1][3]);
	//printf("%f\n", content_tensor[0][0][2][0]);
	//printf("%f\n", content_tensor[0][0][3][0]);
	//printf("%f\n", content_tensor[0][0][9][0]);

	//printf("\n");

	// Test third layer

	//printf("%f\n", content_tensor[0][0][74][255]);
	//printf("%f\n", content_tensor[0][1][0][0]);
	//printf("%f\n", content_tensor[0][2][0][0]);
	//printf("%f\n", content_tensor[0][3][0][0]);
	//printf("%f\n", content_tensor[0][25][0][0]);
}

int main(){
	getContentTensor1();
	getContentTensor2();
	getStyleTensor1();
	getStyleTensor2();

	printf("%f\n", content_tensor1[0][0][0][2]);
	printf("%f\n", content_tensor2[0][0][0][2]);
	printf("%f\n", style_tensor1[0][0][0][6]);
	printf("%f\n", style_tensor2[0][0][0][8]);
	return 0;
}

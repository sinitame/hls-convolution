#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define IMG_WIDTH 220
#define IMG_HEIGHT 220
#define KERNEL_DIM 3
#define DIM_IN 3

void saveImage(const std::string path, cv::InputArray inArr);
void printSmallMatrixCVShort(char *title, cv::InputArray inArr);
void printSmallMatrixCVChar(char *title, cv::InputArray inArr);
void conv2dByHand(cv::InputArray imageIn, char imageOut[IMG_HEIGHT][IMG_WIDTH],  char kernel[KERNEL_DIM][KERNEL_DIM][DIM_IN], int kSize);



short processWindowAndKernel(short kernel[KERNEL_DIM][KERNEL_DIM][DIM_IN], char window[KERNEL_DIM][KERNEL_DIM][DIM_IN], int dim);


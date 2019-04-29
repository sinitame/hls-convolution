#include "testUtils.h"


void saveImage(const std::string path, cv::InputArray inArr)
{
	double min;
	double max;
	cv::minMaxIdx(inArr, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(inArr, adjMap, 255 / max);
	cv::imwrite(path,adjMap);
}

void printSmallMatrixCVShort(char *title, cv::InputArray inArr)
{
	printf("\%s\n",title);
	printf("----------------------------------\n");
	for (int rows = 0; rows < inArr.getMat().rows; rows++)
	{
		printf("ROW: %02d --->  ",rows);
		for (int cols = 0; cols < IMG_WIDTH; cols++)
		{
			printf(" %04d ",inArr.getMat().at<short>(rows,cols));
		}
		printf("\n");
	}
	printf("\n");
}

void printSmallMatrixCVChar(char *title, cv::InputArray inArr)
{
	printf("\%s\n",title);
	printf("----------------------------------\n");
	for (int rows = 0; rows < inArr.getMat().rows; rows++)
	{
		printf("ROW: %02d --->  ",rows);
		for (int cols = 0; cols < IMG_WIDTH; cols++)
		{
			printf(" %04d ",inArr.getMat().at<unsigned char>(rows,cols));
		}
		printf("\n");
	}
	printf("\n");
}

// The convolution is all about summing the product of the window and the kernel
short processWindowAndKernel(char kernel[KERNEL_DIM][KERNEL_DIM][DIM_IN], unsigned char window[KERNEL_DIM][KERNEL_DIM], int dim)
{
	short accumulator = 0;

	for (int rows = 0; rows < KERNEL_DIM; rows++)
	{
		for (int cols = 0; cols < KERNEL_DIM; cols++)
		{
			accumulator = accumulator + (kernel[rows][cols][dim] * window[rows][cols]);
		}
	}
	return accumulator;
}

void conv2dByHand(cv::InputArray imageIn, char imageOut[IMG_HEIGHT][IMG_WIDTH],  char kernel[KERNEL_DIM][KERNEL_DIM][DIM_IN], int kSize)
{
	// Get the kernel center (For example on the 3x3 is 1)
	int kCenterX = kSize / 2;
	int kCenterY = kSize / 2;
	short accConvolution = 0;
	short numKernelElements = kSize * kSize;

	// Sampling window (Should be the same size of the kernel)
	unsigned char samplingWindow[KERNEL_DIM][KERNEL_DIM];

	// Iterate on the image pixels

	for (int rows = 0; rows < IMG_HEIGHT; rows++)
	{
		for (int cols = 0; cols < IMG_WIDTH; cols++)
		{
			// Accumulator for the convolution should be reset
			accConvolution = 0;

			// Avoid calculate out of the image boundaries
			if ((rows >= kSize-1) && (cols >= kSize-1))
			{
				for (int dim = 0; dim <DIM_IN;dim++)
				{
					// Populate our window by sampling on the image
					for (int rowsWindow = 0; rowsWindow < kSize; rowsWindow++)
					{
						for (int colsWindow = 0; colsWindow < kSize; colsWindow++)
						{
							// The window must be centered on the current pixel, so we need to
							// subtract some pixels back from the current positon
							int repos_row = rows-(kSize-1)+rowsWindow;
							int repos_col = cols-(kSize-1)+colsWindow;
							short currPixVal = imageIn.getMat().at<cv::Vec3b>(repos_row,repos_col)[dim];
							samplingWindow[rowsWindow][colsWindow] = currPixVal;
						}
					}


					// Now multiply the window by the kernel and sum it's result
					accConvolution += processWindowAndKernel(kernel,samplingWindow,dim);
				}
				imageOut[rows][cols] = accConvolution;
			}

		}
	}

}

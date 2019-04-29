#include <stdio.h>
#include "conv_layer.h"
#include "testUtils.h"


// Edge
char kernel[KERNEL_DIM][KERNEL_DIM][DIM_IN] = {
		{{-1, -1, -1},
		{-1, -1, -1},
		{-1, -1, -1}},

		{{-1, -1, -1},
		{8, 8, 8},
		{-1, -1, -1}},

		{{-1, -1, -1},
		{-1, -1, -1},
		{-1, -1, -1}}
};

// Use with morphological (Erode, Dilate)
/*char kernel[KERNEL_DIM*KERNEL_DIM] = {
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
};*/

// Image File path
char outImage[IMG_HEIGHT][IMG_WIDTH];
char outImageRef[IMG_HEIGHT][IMG_WIDTH];

int main()
{
	// Read input image
	printf("Load image %s\n",INPUT_IMAGE_CORE);
	cv::Mat imageSrc;
	imageSrc = cv::imread(INPUT_IMAGE_CORE, CV_LOAD_IMAGE_COLOR);

	// Convert to grayscale
	//cv::cvtColor(imageSrc, imageSrc, CV_BGR2GRAY);
	//printf("Image Rows:%d Cols:%d\n",imageSrc.rows, imageSrc.cols);

	// Define streams for input and output
	hls::stream<uint_8> inputStream;
	hls::stream<uint_8> outputStream;

	// OpenCV mat that point to a array (cv::Size(Width, Height))
	cv::Mat imgCvOut(cv::Size(imageSrc.cols, imageSrc.rows), CV_8UC1, outImage, cv::Mat::AUTO_STEP);
	cv::Mat imgCvOutRef(cv::Size(imageSrc.cols, imageSrc.rows), CV_8UC1, outImageRef, cv::Mat::AUTO_STEP);

	// Populate the input stream with the image bytes
	for (int idxRows=0; idxRows < imageSrc.rows; idxRows++)
	{
		for (int idxCols=0; idxCols < imageSrc.cols; idxCols++)
		{
			for (int dim = 0; dim < DIM_IN; dim++)
			{
				uint_8 valIn;
				valIn = imageSrc.at<cv::Vec3b>(idxRows,idxCols)[dim];
				inputStream << valIn;
			}

		}
	}

	// Do the convolution (Reference)
	printf("Calling Reference function\n");
 	conv2dByHand(imageSrc,outImageRef,kernel,KERNEL_DIM);
	printf("Reference function ended\n");

	// Save image out file or display

		printSmallMatrixCVChar("Ref Core", imgCvOutRef);

		printf("Saving image Ref\n");
		saveImage(std::string(OUTPUT_IMAGE_REF) ,imgCvOutRef);



	// Do the convolution on the core (Third parameter choose operation 0(conv),1(erode),2(dilate)
	printf("Calling Core function\n");
	conv_layer(inputStream, outputStream);
	printf("Core function ended\n");

	// Take data from the output stream to our array outImage (Pointed in opencv)
		for (int idxRows=0; idxRows < imageSrc.rows; idxRows++)
			{
				for (int idxCols=0; idxCols < imageSrc.cols; idxCols++)
				{
					uint_8 valOut;
					outputStream.read(valOut);
					outImage[idxRows][idxCols] = valOut;
				}
			}




		printSmallMatrixCVChar("Res Core", imgCvOut);
		printf("Saving image\n");
		saveImage(std::string(OUTPUT_IMAGE_CORE) ,imgCvOut);

	return 0;

}

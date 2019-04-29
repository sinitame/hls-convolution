#include "conv_layer.h"

void conv_layer(hls::stream<uint_8> &inStream, hls::stream<uint_8> &outStream, char weights[KERNEL_DIM][KERNEL_DIM][DIM_IN])
{
#pragma HLS INTERFACE axis port=inStream
#pragma HLS INTERFACE axis port=outStream
#pragma HLS INTERFACE bram port=weights
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS


	hls::LineBuffer<KERNEL_DIM,IMG_WIDTH,unsigned char> lineBuff1;
	hls::LineBuffer<KERNEL_DIM,IMG_WIDTH,unsigned char> lineBuff2;
	hls::LineBuffer<KERNEL_DIM,IMG_WIDTH,unsigned char> lineBuff3;

	hls::Window<KERNEL_DIM,KERNEL_DIM,short> window1;
	hls::Window<KERNEL_DIM,KERNEL_DIM,short> window2;
	hls::Window<KERNEL_DIM,KERNEL_DIM,short> window3;

	hls::Window<KERNEL_DIM,KERNEL_DIM,short> *windows[DIM_IN] = {&window1, &window2, &window3};
	hls::LineBuffer<KERNEL_DIM,IMG_WIDTH,unsigned char> *lineBuffArray[DIM_IN] = {&lineBuff1, &lineBuff2, &lineBuff3};


	bool pixValid = false;
	int res1 = 0;
	int res2 = 0;
	int res3 = 0;
	short result = 0;

	int idCol =0;
	int dim = 0;
	int winPos;
	int idRow =0;
	int idPix = 0;
	int idPixOut = 0;
	int waitTicks = ((IMG_WIDTH*(KERNEL_DIM-1)+KERNEL_DIM)*3)/2;
	int countWait = 0;

	uint_8 pixelChannel;

	while (!inStream.empty())
	{
#pragma HLS PIPELINE
		pixelChannel = inStream.read();
		unsigned char pixelIn = pixelChannel;

		lineBuffArray[dim]->shift_up(idCol);
		lineBuffArray[dim]->insert_top(pixelIn,idCol);

		for (int idWinX = 0; idWinX < KERNEL_DIM; idWinX++)
		{
			for (int idWinY = 0; idWinY < KERNEL_DIM; idWinY++)
			{
				winPos = idPixOut%(IMG_WIDTH-2);
				short val = (short)lineBuffArray[dim]->getval(idWinX, idWinY + winPos);
				windows[dim]->insert(val,idWinX,idWinY);
			}
		}

		result = 0;
		//Calculate when we can convolve (buffer filled enough)
		if ((idRow >=KERNEL_DIM-1) && (idCol >= KERNEL_DIM-1)){
			if (dim == DIM_IN-1){
				idPixOut++;

				multWindow(&window1, weights, 0);
				res1 = sumWindow(&window1);

				multWindow(&window2, weights, 1);
				res2 = sumWindow(&window2);

				multWindow(&window3, weights, 2);
				res3 = sumWindow(&window3);

				result = (short)(res1 + res2 + res3);
			}
		} else {
			result = 0;
		}


		countWait++;
		if ((countWait > waitTicks) && (dim == DIM_IN-1))
		{
			outStream.write(result);
			// Calculate pixel number of an image
			if (idPix<IMG_WIDTH*IMG_HEIGHT*DIM_IN)
			{
				idPix++;
			} else {
				idPix = 0;
			}
		}

		//Calculate row and column index
		if (dim == DIM_IN - 1)
		{
			if (idCol < IMG_WIDTH-1)
			{
				idCol++;
			} else {
				idCol = 0;
				idRow++;
			}
			dim = 0;
		} else {
			dim++;
		}

	}

	// Now send the remaining zeros (Just the (Number of delayed ticks)
	for (countWait = 0; countWait < waitTicks; countWait++)
	{
		result = 0;
		outStream.write(result);
	}
};








// Sum all values inside window (Already multiplied by the kernel)
short sumWindow(hls::Window<KERNEL_DIM,KERNEL_DIM,short> *window)
{
	short accumulator = 0;

	// Iterate on the window multiplying and accumulating the kernel and sampling window
	for (int idxRow = 0; idxRow < KERNEL_DIM; idxRow++)
	{
		for (int idxCol = 0; idxCol < KERNEL_DIM; idxCol++)
		{
			accumulator = accumulator + (short)window->getval(idxRow,idxCol);
		}
	}
	return accumulator;
}

// Multiply each coefficient of a window to the corresponding coefficient of the kernel
void multWindow(hls::Window<KERNEL_DIM,KERNEL_DIM,short> *win, char kernel[KERNEL_DIM][KERNEL_DIM][DIM_IN], int dim){


	for (int idxWinX = 0; idxWinX < KERNEL_DIM; idxWinX++)
		{
			for (int idxWinY = 0; idxWinY < KERNEL_DIM; idxWinY++)
			{
				short val = (short)win->getval(idxWinX, idxWinY);

				// Multiply kernel by the sampling window
				val = (short)(kernel[idxWinX][idxWinY][dim] * val);
				win->insert(val,idxWinX,idxWinY);
			}
		}

}




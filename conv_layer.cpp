#include "conv_layer.h"

void conv_layer(hls::stream<uint_8> &inStream, hls::stream<uint_8> &outStream)
{
#pragma HLS INTERFACE axis port=inStream
#pragma HLS INTERFACE axis port=outStream
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

	char weights[KERNEL_DIM][KERNEL_DIM][DIM_IN] = {
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

	hls::Window<KERNEL_DIM,KERNEL_DIM,short> windows[DIM_IN];
	hls::LineBuffer<KERNEL_DIM,IMG_WIDTH,unsigned char> lineBuffArray[DIM_IN];

	for (int feature_in = 0; feature_in < DIM_IN; feature_in++)
	{
		hls::LineBuffer<KERNEL_DIM,IMG_WIDTH,unsigned char> lineBuff;
		hls::Window<KERNEL_DIM,KERNEL_DIM,short> window;

		windows[feature_in] = window;
		lineBuffArray[feature_in] = lineBuff;
	}



	bool pixValid = false;
	short res = 0;
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

	//while (!inStream.empty())
	for (int i=0; i< IMG_WIDTH*IMG_HEIGHT*DIM_IN; i++)
	{
#pragma HLS PIPELINE
		pixelChannel = inStream.read();
		unsigned char pixelIn = pixelChannel;

		lineBuffArray[dim].shift_up(idCol);
		lineBuffArray[dim].insert_top(pixelIn,idCol);

		for (int idWinX = 0; idWinX < KERNEL_DIM; idWinX++)
		{
			for (int idWinY = 0; idWinY < KERNEL_DIM; idWinY++)
			{
				winPos = idPixOut%(IMG_WIDTH-2);
				short val = (short)lineBuffArray[dim].getval(idWinX, idWinY + winPos);
				windows[dim].insert(val,idWinX,idWinY);
			}
		}

		result = 0;
		//Calculate when we can convolve (buffer filled enough)
		if ((idRow >=KERNEL_DIM-1) && (idCol >= KERNEL_DIM-1)){
			if (dim == DIM_IN-1){
				idPixOut++;

				for (int feature_in = 0; feature_in < DIM_IN; feature_in++)
				{
#pragma HLS UNROLL factor=3
					multWindow(&windows[feature_in], weights, feature_in);
					res = sumWindow(&windows[feature_in]);
					result += res;
				}
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




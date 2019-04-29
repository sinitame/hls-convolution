#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "hls_video.h"
#include <ap_int.h>

#define IMG_WIDTH 220
#define IMG_HEIGHT 220
#define KERNEL_DIM 3
#define DIM_IN 3

#define INPUT_IMAGE_CORE  "/home/sinitambirivoutin/Documents/neural_net/lena.png"
#define OUTPUT_IMAGE_CORE "/home/sinitambirivoutin/Documents/neural_net/lena_out.png"
#define OUTPUT_IMAGE_REF "/home/sinitambirivoutin/Documents/neural_net/lena_ref.png"


typedef ap_uint<8> uint_8;


short sumWindow(hls::Window<KERNEL_DIM,KERNEL_DIM,short> *window);
void multWindow(hls::Window<KERNEL_DIM,KERNEL_DIM,short> *win, char kernel[KERNEL_DIM][KERNEL_DIM][DIM_IN], int dim);
void win_layer(hls::stream<uint_8> &inStream, hls::Window<KERNEL_DIM,KERNEL_DIM,short> ** windows);

void conv_layer(hls::stream<uint_8> &inStream, hls::stream<uint_8> &outStream);
//char kernel[KERNEL_DIM][KERNEL_DIM][DIM_IN]

#endif

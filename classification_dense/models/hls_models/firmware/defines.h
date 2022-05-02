#ifndef DEFINES_H_
#define DEFINES_H_

#ifndef __INTELFPGA_COMPILER__
#include "ac_int.h"
#include "ac_fixed.h"
#define hls_register
#else
#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 1
#define N_LAYER_2 10
#define N_LAYER_4 2


//hls-fpga-machine-learning insert layer-precision
typedef ac_fixed<16,6,true> model_default_t;
typedef ac_fixed<16,6,true> input_t;
typedef ac_fixed<16,6,true> layer2_t;
typedef ac_fixed<8,1,true> dense_4_weight_t;
typedef ac_fixed<8,1,true> dense_4_bias_t;
typedef ac_int<1, false> layer2_index;
typedef ac_fixed<16,6,true> dense_4_relu_default_t;
typedef ac_fixed<18,8,true> dense_4_relutable_t;
typedef ac_fixed<16,6,true> layer3_t;
typedef ac_fixed<16,6,true> layer4_t;
typedef ac_fixed<8,1,true> dense_5_weight_t;
typedef ac_fixed<8,1,true> dense_5_bias_t;
typedef ac_int<1, false> layer4_index;
typedef ac_fixed<16,6,true> dense_5_sigmoid_default_t;
typedef ac_fixed<18,8,true> dense_5_sigmoidtable_t;
typedef ac_fixed<16,6,true> result_t;


#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif

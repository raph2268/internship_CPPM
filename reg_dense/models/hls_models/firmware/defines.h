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
#define N_LAYER_2 26
#define N_LAYER_4 26
#define N_LAYER_6 1


//hls-fpga-machine-learning insert layer-precision
typedef ac_fixed<16,6,true> model_default_t;
typedef ac_fixed<16,6,true> input_t;
typedef ac_fixed<16,6,true> layer2_t;
typedef ac_fixed<15,4,true> weight2_t;
typedef ac_fixed<15,4,true> bias2_t;
typedef ac_int<1, false> layer2_index;
typedef ac_fixed<16,6,true> layer3_t;
typedef ac_fixed<18,8,true> q_dense_429_relu_table_t;
typedef ac_fixed<16,6,true> layer4_t;
typedef ac_fixed<15,4,true> weight4_t;
typedef ac_fixed<15,4,true> bias4_t;
typedef ac_int<1, false> layer4_index;
typedef ac_fixed<16,6,true> layer5_t;
typedef ac_fixed<18,8,true> q_dense_430_relu_table_t;
typedef ac_fixed<16,6,true> result_t;
typedef ac_fixed<15,4,true> weight6_t;
typedef ac_fixed<15,4,true> bias6_t;
typedef ac_int<1, false> layer6_index;


#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif

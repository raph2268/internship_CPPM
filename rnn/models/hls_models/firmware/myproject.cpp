//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"

//hls-fpga-machine-learning insert weights
#include "weights/kernel_2.h"
#include "weights/bias_2.h"
#include "weights/recurrent_kernel_2.h"
#include "weights/w3.h"
#include "weights/b3.h"

#ifndef __INTELFPGA_COMPILER__
output_data myproject(
   input_data inputs
) {
#else
//hls-fpga-machine-learning insert cpragmas
hls_max_concurrency(0)
hls_component_ii(1)
hls_scheduler_target_fmax_mhz(200)
component output_data myproject(
   input_data inputs
) {
#endif
    hls_register output_data outputs;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[OUT_HEIGHT_2];
    simple_rnn_network<input_t, layer2_t, config2 ,input_t>(inputs.simple_rnn_input, layer2_out, kernel_2,recurrent_kernel_2,bias_2);

    layer3_t layer3_out[N_LAYER_3];
    nnet::dense_resource<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3);

    hls_register output_data layer4_out;
    nnet::relu<layer3_t, result_t, relu_config4>(layer3_out, outputs.layer4_out);

    return outputs;
}

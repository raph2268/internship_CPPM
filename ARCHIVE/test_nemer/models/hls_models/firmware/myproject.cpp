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
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w4.h"
#include "weights/b4.h"

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

    layer2_t layer2_out[N_LAYER_2] hls_register;
    nnet::dense_resource<input_t, layer2_t, config2>(inputs.dense_input, layer2_out, w2, b2);

    layer3_t layer3_out[N_LAYER_2] hls_register;
    nnet::sigmoid<layer2_t, layer3_t, sigmoid_config3>(layer2_out, layer3_out);

    nnet::dense_resource<layer3_t, result_t, config4>(layer3_out, outputs.layer4_out, w4, b4);

    return outputs;
}

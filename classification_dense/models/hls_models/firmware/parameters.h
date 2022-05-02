#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"

//hls-fpga-machine-learning insert layer-config
struct config2 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 10;
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef dense_4_bias_t bias_t;
    typedef dense_4_weight_t weight_t;
    typedef layer2_index index_t;
};

struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = 10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense_4_relutable_t table_t;
};

struct config4 : nnet::dense_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 20;
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef dense_5_bias_t bias_t;
    typedef dense_5_weight_t weight_t;
    typedef layer4_index index_t;
};

struct sigmoid_config5 : nnet::activ_config {
    static const unsigned n_in = 2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense_5_sigmoidtable_t table_t;
};



#endif

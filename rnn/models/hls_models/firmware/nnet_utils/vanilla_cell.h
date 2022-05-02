
#include "HLS/hls.h"
#include <stdio.h>
#include "HLS/ac_int.h"
#ifdef __INTELFPGA_COMPILER__
#include "HLS/ac_fixed.h"
#else
#include "ref/ac_fixed.h"
#endif
#include "HLS/ac_fixed_math.h"

#include "nnet_activation.h"

#ifndef HLS_SYNTHESIS
  #include <iostream>
  #include <fstream>
#endif


#ifndef SIMULATION_TIMES
  #define SIMULATION_TIMES 1
#endif
#ifndef TIMESTAMP_UNROLLING
  #define TIMESTAMP_UNROLLING
#endif


using namespace ihc;


struct vanilla_config: public nnet::activ_config{
 static const unsigned n_in=8;
 static const unsigned n_timestamp=5;
 typedef ac_fixed<16,6,true> weight_t;
 typedef ac_fixed<23,3,true> fixed_p_internal_t;
};

//--------------------------------------------------------- COMUM CODE -----------------------------------------------

template<class data_T, typename res_T, typename CONFIG_T, class WEIGHT_T>
void multiply_W(data_T input,
                res_T *out,
                const WEIGHT_T *kernel) {
    MULTIPLY_W_LOOP:
    #pragma unroll
    for (int j = 0; j < CONFIG_T::n_in; j++) {
      out[j] = input * kernel[j];
    }
}

template<class data_T, typename res_T, typename CONFIG_T, class WEIGHT_T>
void multiply_U(data_T *inputs, 
                res_T out[],
                const WEIGHT_T *recurrent_kernel) {
  MULTIPLY_U_LOOP_I:
  for (int i = 0 ; i <  CONFIG_T::n_in ; i++){
    out[i] = 0;
    MULTIPLY_U_LOOP_J: 
    #pragma unroll
    for (int j=0;j< CONFIG_T::n_in ; j++){
      out[i] += inputs[j] * recurrent_kernel[j*CONFIG_T::n_in +i];
    }
  }

}


template<typename res_T, typename CONFIG_T, class WEIGHT_T>
void add_bias(res_T *inputs,
              res_T *out,
              const WEIGHT_T *bias) {

    ADD_BIAS_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        out[i] = inputs[i] + bias[i];

    }

}
template<class data_T, class res_T,typename CONFIG_T>
void multiply_vectors(data_T *in1, data_T *in2, res_T out[] ) {
    MULTIPLY_VECT_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        out[i] = (res_T)(in1[i] * in2[i]);
    }
}
template<typename res_T,typename CONFIG_T>
void add_vectors(res_T *in1, res_T *in2, res_T out[] ) {

    ADD_VECTOR_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        out[i] = (res_T) in1[i] + in2[i];

    }
}


// ---------------------------------------- VANILLA-RNN CODE -------------------------------------------------
template<class data_T, typename CONFIG_T, class WEIGHT_T>
 void vanilla_cell(
          data_T *hidden_state,
          data_T *hidden_state_o,
          data_T inputs,
          WEIGHT_T *kernel,
          WEIGHT_T *rec_kernel,
          WEIGHT_T *bias);

template<class data_T, class res_T, typename CONFIG_T, class WEIGHT_T>
  void vanilla_network(data_T input0[CONFIG_T::n_timestamp], res_T res[CONFIG_T::n_out],
const WEIGHT_T *kernel, const WEIGHT_T *rec_kernel, const WEIGHT_T *bias){


  data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
  data_T hidden_state_temp[CONFIG_T::n_in]  ;
  data_T h[CONFIG_T::n_in] hls_register    ;

  static data_T inputs[CONFIG_T::n_timestamp] hls_register;

  INIT_LOOP:
  #pragma unroll
  for (int x = 0; x < CONFIG_T::n_in; x++) {
    hidden_state[x][0]=0;
  }


  /*#ifndef HLS_SYNTHESIS
  for(int z=0; z<6; z++){
    std::cout<<"CONFIG_T n_in: " << CONFIG_T::n_in <<std::endl;
    for(int p=0; p<8; p++){
      std::cout<<"saida layer 2 z:"<< z <<"p:"<< p << " :" << hidden_state[p][z] <<std::endl;
    }
  }
  std::cout<<"-------------------Negisty ---------------------- " <<std::endl;
  #endif*/


  #pragma unroll
  #pragma ivdep
  for (int j=0; j<CONFIG_T::n_timestamp; j++){
    inputs[j] = input0[j];
  }
  
  /*
  #ifndef HLS_SYNTHESIS
  for(int z=0; z<5; z++){
    std::cout<<"inputs: " << inputs[z] <<std::endl;
  }
  std::cout<<"-------------------Negisty ---------------------- " <<std::endl;
  #endif
  */
  #pragma unroll TIMESTAMP_UNROLLING
  for (int i=0; i < CONFIG_T::n_timestamp; i++){
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state_temp[x] = hidden_state[x][i];
    }
    
    vanilla_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,inputs[i], kernel, rec_kernel, bias);
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][i+1]=h[x];
    }
  }
  #pragma unroll
  for (int x = 0; x < CONFIG_T::n_in; x++) {
    res[x] = hidden_state[x][CONFIG_T::n_timestamp];
  }
}

template<class data_T, typename CONFIG_T, typename WEIGHT_T>
void vanilla_cell(
          data_T *hidden_state,
          data_T *hidden_state_o,
          data_T inputs,
          const WEIGHT_T *kernel, 
          const WEIGHT_T *rec_kernel, 
          const WEIGHT_T *bias){


        //----------------------
        //Internals definitions
        //----------------------


         // Gate outputs




        //Weight multiplication
        typename vanilla_config::fixed_p_internal_t afterW[CONFIG_T::n_in] hls_register;
        multiply_W<data_T,vanilla_config::fixed_p_internal_t,CONFIG_T,WEIGHT_T>(inputs, afterW, kernel);
        //Bias addition
        typename vanilla_config::fixed_p_internal_t afterBias[CONFIG_T::n_in] hls_register;
        add_bias<vanilla_config::fixed_p_internal_t,CONFIG_T,WEIGHT_T>(afterW,afterBias, bias);

        //hidden        
        typename vanilla_config::fixed_p_internal_t hiddenCand[CONFIG_T::n_in] hls_register;
        multiply_U<data_T,vanilla_config::fixed_p_internal_t,CONFIG_T,WEIGHT_T>(hidden_state, hiddenCand, rec_kernel);
        typename vanilla_config::fixed_p_internal_t afterAdd[CONFIG_T::n_in];
        add_vectors<vanilla_config::fixed_p_internal_t, CONFIG_T>(afterBias, hiddenCand, afterAdd);

        data_T h[CONFIG_T::n_in]/* hls_register*/;
        nnet::relu<vanilla_config::fixed_p_internal_t,data_T,CONFIG_T>(afterAdd, h);  //recurrent_activation
       OUTPUT_WRITE_LOOP:
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_in; x++) {
          hidden_state_o[x]=h[x];
        }

        return;
}



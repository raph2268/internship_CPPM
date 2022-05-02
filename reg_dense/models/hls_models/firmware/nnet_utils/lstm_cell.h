#ifndef LSTMCELLH
#define LSTMCELLH

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

using namespace ihc;


struct lstm_config : public nnet::activ_config{
 static const unsigned n_in=10;
 static const unsigned n_timestamp=10;

 typedef ac_fixed<16,6,true> weight_t;
};


#ifndef HLS_SYNTHESIS
  #include <iostream>
  #include <fstream>
#endif
#ifndef TIMESTAMP_UNROLLING
  #define TIMESTAMP_UNROLLING
#endif


template<class data_T, class res_T,typename CONFIG_T,class WEIGHT_T>
void multiply_W(data_T input, res_T *out, const WEIGHT_T *weight) {
    MULTIPLY_W_LOOP:
    if(input != 1){
      #ifndef HLS_SYNTHESIS
      //   std::cout<<"Opçao caca"<<std::endl;
      #endif
      #pragma unroll
      for (int j = 0; j < CONFIG_T::n_in; j++) {
        //out[j] = input * WEIGHT_T::kernel[j];
        out[j] = input * weight[j];
      }
    }

    else{
      #ifndef HLS_SYNTHESIS
         //std::cout<<"Opçao baba"<<std::endl;
      #endif
      #pragma unroll
      for (int i = 0; i < CONFIG_T::n_in ; i++) {
          out[i] = 0;
          #pragma unroll
           for (int j = 0; j < CONFIG_T::n_in; j++) {
              //out[i] += /*out[i] +*/ inputs[j] * WEIGHT_T::recurrent_kernel[j][i];
              out[i] += /*out[i] +*/ input[j] * weight[j*CONFIG_T::n_in +i];
          }
      }
    }
}
template<class data_T, class res_T,typename CONFIG_T,class WEIGHT_T>
void multiply_U(data_T *inputs, res_T out[], const WEIGHT_T *weight) {
    MULTIPLY_U_LOOP_I:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in ; i++) {
        out[i] = 0;
        MULTIPLY_U_LOOP_J:
        #pragma unroll
         for (int j = 0; j < CONFIG_T::n_in; j++) {
            //out[i] += /*out[i] +*/ inputs[j] * WEIGHT_T::recurrent_kernel[j][i];
            //out[i] += /*out[i] +*/ inputs[j] * weight[j*CONFIG_T::n_in +i];
            out[i] += (data_T) inputs[j] * weight[j*CONFIG_T::n_in +i];

        }
    }
}
//template<class data_T, typename CONFIG_T, typename WEIGHT_T>
template<class data_T,class res_T, typename CONFIG_T, class WEIGHT_T>
void add_bias(data_T *inputs,res_T *out,const WEIGHT_T *bias) {

    ADD_BIAS_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        //inputs[i] = inputs[i] + WEIGHT_T::bias[i];
        out[i] = inputs[i] + bias[i];

    }

}
template<class data_T, class res_T, typename CONFIG_T>
void multiply_vectors(data_T *in1, data_T *in2, res_T *out) {
    MULTIPLY_VECT_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        out[i] = in1[i] * in2[i];

    }
}
template<class data_T, class res_T,typename CONFIG_T>
void add_vectors(data_T *in1,data_T *in2,res_T *out) {

    ADD_VECTOR_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        out[i] = in1[i] + in2[i];

    }
}
template<class data_T, typename CONFIG_T,class  WEIGHT_T>
void lstm_cell(
          data_T *hidden_state,
          data_T *hidden_state_o,
          data_T *cell_state,
          data_T *cell_state_o,
          data_T inputs ,
          WEIGHT_T *WI   , WEIGHT_T *WF   , WEIGHT_T *WC   , WEIGHT_T *WO  ,
          WEIGHT_T *RWI  , WEIGHT_T *RWF  , WEIGHT_T *RWC  , WEIGHT_T *RWO ,
          WEIGHT_T *BI   , WEIGHT_T *BF   , WEIGHT_T *BC   , WEIGHT_T *BO);


template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network(data_T input0[CONFIG_T::n_timestamp],res_T res[CONFIG_T::n_out],
            const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
            const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
            const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

    data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T cell_state  [CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_in]    ;
    data_T cell_state_temp  [CONFIG_T::n_in]    ;
    data_T h[CONFIG_T::n_in]    ;
    data_T c[CONFIG_T::n_in]    ;

    static data_T inputs[CONFIG_T::n_timestamp] ;
    //#ifndef HLS_SYNTHESIS
       //std::cout<<"Opçao 1"<<std::endl;
    //#endif

    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep
    //input0 - verification
    for (int j=0; j<CONFIG_T::n_timestamp; j++){
    inputs[j] = input0[j];
  }

    #pragma unroll TIMESTAMP_UNROLLING
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
        cell_state_temp[x]   = cell_state[x][i];
      }
      lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }
    #pragma unroll
    //output - verification
    for (int x = 0; x < CONFIG_T::n_in; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
    }
  }


template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network(data_T input0[CONFIG_T::n_timestamp],res_T res[CONFIG_T::n_timestamp][CONFIG_T::n_out],
            const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
            const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
            const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

    data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T cell_state  [CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_in]    ;
    data_T cell_state_temp  [CONFIG_T::n_in]    ;
    data_T h[CONFIG_T::n_in]    ;
    data_T c[CONFIG_T::n_in]    ;

    static data_T inputs[CONFIG_T::n_timestamp] ;
    #ifndef HLS_SYNTHESIS
       std::cout<<"Opçao 2"<<std::endl;
    #endif
    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep

    //input0 - verification
    for (int j=0; j<CONFIG_T::n_timestamp; j++){
    inputs[j] = input0[j];
  }


    #pragma unroll TIMESTAMP_UNROLLING
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
        cell_state_temp[x]   = cell_state[x][i];
      }
      lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }
    #pragma unroll
    //output - verification

    for(int x = 0; x < CONFIG_T::n_timestamp; x++){
      for(int h = 0; h < CONFIG_T::n_in; h++){
          res[x][h]= hidden_state[h][x+1];
      }
    }
  }

template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network(data_T input0[CONFIG_T::n_timestamp][CONFIG_T::n_in],res_T res[CONFIG_T::n_out],
            const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
            const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
            const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

    data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T cell_state  [CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_in]     ;
    data_T cell_state_temp  [CONFIG_T::n_in]     ;
    data_T h[CONFIG_T::n_in]     ;
    data_T c[CONFIG_T::n_in]     ;

    static data_T inputs[CONFIG_T::n_timestamp][CONFIG_T::n_in] ;
    #ifndef HLS_SYNTHESIS
       std::cout<<"Opçao 3"<<std::endl;
    #endif
    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep
    //input0 - verification
    for (int j=0; j<CONFIG_T::n_timestamp; j++){
      for (int z=0; z<CONFIG_T::n_in; z++){
      inputs[j][z] = input0[j][z];
    }
  }
    #pragma unroll TIMESTAMP_UNROLLING
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
        cell_state_temp[x]   = cell_state[x][i];
      }
      lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }
    #pragma unroll
    //output - verification
    for (int x = 0; x < CONFIG_T::n_in; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
    }
  }


template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network(data_T input0[CONFIG_T::n_timestamp][CONFIG_T::n_in],res_T res[CONFIG_T::n_timestamp][CONFIG_T::n_out],
            const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
            const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
            const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

    data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T cell_state  [CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_in]     ;
    data_T cell_state_temp  [CONFIG_T::n_in]     ;
    data_T h[CONFIG_T::n_in]     ;
    data_T c[CONFIG_T::n_in]     ;

    static data_T inputs[CONFIG_T::n_timestamp][CONFIG_T::n_in] ;
    #ifndef HLS_SYNTHESIS
       std::cout<<"Opçao 4"<<std::endl;
    #endif
    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep

    //input0 - verification
      for (int j=0; j<CONFIG_T::n_timestamp; j++){
        for (int z=0; z<CONFIG_T::n_in; z++){
        inputs[j][z] = input0[j][z];
      }
    }

    #pragma unroll TIMESTAMP_UNROLLING
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
        cell_state_temp[x]   = cell_state[x][i];
      }
      lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }
    #pragma unroll
    //output - verification
    for(int x = 0; x < CONFIG_T::n_timestamp; x++){
      for(int h = 0; h < CONFIG_T::n_in; h++){
          res[x][h]= hidden_state[h][x+1];
      }
    }
  }

  template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network(data_T input0,res_T res[CONFIG_T::n_timestamp][CONFIG_T::n_out],
            const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
            const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
            const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

    data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T cell_state  [CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_in]     ;
    data_T cell_state_temp  [CONFIG_T::n_in]     ;
    data_T h[CONFIG_T::n_in]     ;
    data_T c[CONFIG_T::n_in]     ;

    static data_T inputs[CONFIG_T::n_timestamp] ;
    #ifndef HLS_SYNTHESIS
       std::cout<<"Opçao 5"<<std::endl;
    #endif
    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep
    //input0 - verification
    for (int j=1;j<CONFIG_T::n_timestamp; j++){
      inputs[j-1] = inputs[j];
    }
    inputs[CONFIG_T::n_timestamp-1]=input0;

    #pragma unroll TIMESTAMP_UNROLLING
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
        cell_state_temp[x]   = cell_state[x][i];
      }
      lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }
    #pragma unroll
    //output - verification
    for(int x = 0; x < CONFIG_T::n_timestamp; x++){
      for(int h = 0; h < CONFIG_T::n_in; h++){
          res[x][h]= hidden_state[h][x+1];
      }
    }
  }

  template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network(data_T input0,res_T res[CONFIG_T::n_out],
            const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
            const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
            const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

    data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T cell_state  [CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_in]     ;
    data_T cell_state_temp  [CONFIG_T::n_in]     ;
    data_T h[CONFIG_T::n_in]     ;
    data_T c[CONFIG_T::n_in]     ;

    static data_T inputs[CONFIG_T::n_timestamp] ;
    #ifndef HLS_SYNTHESIS
       std::cout<<"Opçao 6"<<std::endl;
    #endif
    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep
    //input0 - verification
    for (int j=1;j<CONFIG_T::n_timestamp; j++){
      inputs[j-1] = inputs[j];
    }
    inputs[CONFIG_T::n_timestamp-1]=input0;

    #pragma unroll TIMESTAMP_UNROLLING
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
        cell_state_temp[x]   = cell_state[x][i];
      }
      lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_in; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }
    #pragma unroll
    //output - verification
    for (int x = 0; x < CONFIG_T::n_in; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
    }
  }

/*
res_T _return_sequences(){
  if(CONFIG_T::return_sequences == false ){
    return res[CONFIG_T::n_out];
  }
  else{
    return res[CONFIG_T::n_timestamp][CONFIG_T::n_out];
  }
}
data_T _sliding_window(){
  if(CONFIG_T::return_sequences == false ){
    return input0[CONFIG_T::n_timestamp][CONFIG_T::n_in];
  }
  else{
    return res[CONFIG_T::n_timestamp][CONFIG_T::n_out];
  }
}

template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
void lstm_network(data_T input0,res_T _return_sequences(),
          const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
          const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
          const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

  data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
  data_T cell_state  [CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
  data_T hidden_state_temp[CONFIG_T::n_in] hls_register    ;
  data_T cell_state_temp  [CONFIG_T::n_in] hls_register    ;
  data_T h[CONFIG_T::n_in] hls_register    ;
  data_T c[CONFIG_T::n_in] hls_register    ;

  #ifndef HLS_SYNTHESIS
    std::cout<<"configuration_rs :"<< CONFIG_T::return_sequences  <<std::endl;
  #endif

  static data_T inputs[CONFIG_T::n_timestamp] hls_register;

  INIT_LOOP:
  #pragma unroll
  for (int x = 0; x < CONFIG_T::n_in; x++) {
    hidden_state[x][0]=0;
    cell_state[x][0]=0;
  }

  #pragma unroll
  #pragma ivdep
  //input0 - verification

  #pragma unroll TIMESTAMP_UNROLLING
  for (int i=0; i < CONFIG_T::n_timestamp; i++){
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state_temp[x] = hidden_state[x][i];
      cell_state_temp[x]   = cell_state[x][i];
    }
    lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][i+1]=h[x];
      cell_state[x][i+1]=c[x];
    }
  }
  #pragma unroll
  if(CONFIG_T::return_sequences == false){
    for (int x = 0; x < CONFIG_T::n_in; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
    }
  }
  }
  else{
    for (int x = 0; x < CONFIG_T::n_in; x++) {
        for(int h = 0; h < CONFIG_T::n_timestamp; h++){
          res[h][x]= hidden_state[x][h];
        }
    }
  }
  }
}
*/

/*
void lstm_network(data_T input0,res_T res[CONFIG_T::n_out],
          const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
          const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
          const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

  data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
  data_T cell_state  [CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
  data_T hidden_state_temp[CONFIG_T::n_in] hls_register    ;
  data_T cell_state_temp  [CONFIG_T::n_in] hls_register    ;
  data_T h[CONFIG_T::n_in] hls_register    ;
  data_T c[CONFIG_T::n_in] hls_register    ;

  #ifndef HLS_SYNTHESIS
    std::cout<<"configuration_rs :"<< CONFIG_T::return_sequences  <<std::endl;
  #endif

  static data_T inputs[CONFIG_T::n_timestamp] hls_register;

  INIT_LOOP:
  #pragma unroll
  for (int x = 0; x < CONFIG_T::n_in; x++) {
    hidden_state[x][0]=0;
    cell_state[x][0]=0;
  }

  #pragma unroll
  #pragma ivdep
  //input0 - verification

  #pragma unroll TIMESTAMP_UNROLLING
  for (int i=0; i < CONFIG_T::n_timestamp; i++){
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state_temp[x] = hidden_state[x][i];
      cell_state_temp[x]   = cell_state[x][i];
    }
    lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][i+1]=h[x];
      cell_state[x][i+1]=c[x];
    }
  }
  #pragma unroll
  //output - verification
}
*/
template<class data_T, typename CONFIG_T, typename WEIGHT_T>
void lstm_cell(
          data_T *hidden_state,
          data_T *hidden_state_o,
          data_T *cell_state,
          data_T *cell_state_o,
          data_T inputs,
          const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
          const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
          const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

        //----------------------
        //Internals definitions
        //----------------------

        data_T i_afterW   [lstm_config::n_in] ;
        data_T i_afterBias[lstm_config::n_in] ;
        data_T c_afterW   [lstm_config::n_in] ;
        data_T c_afterBias[lstm_config::n_in] ;
        data_T o_afterW   [lstm_config::n_in] ;
        data_T o_afterBias[lstm_config::n_in] ;
        data_T f_afterW   [lstm_config::n_in] ;
        data_T f_afterBias[lstm_config::n_in] ;

        // Hidden state Gate candidates, intermediate variables
         data_T i_hiddenCand[lstm_config::n_in] ;
         data_T f_hiddenCand[lstm_config::n_in] ;
         data_T c_hiddenCand[lstm_config::n_in] ;
         data_T o_hiddenCand[lstm_config::n_in] ;
        // AfterAddition, intermediate variables
         data_T i_afterAdd[lstm_config::n_in] ;
         data_T f_afterAdd[lstm_config::n_in] ;
         data_T c_afterAdd[lstm_config::n_in] ;
         data_T o_afterAdd[lstm_config::n_in] ;

         // Gate outputs
        data_T gate_i[lstm_config::n_in] ;
        data_T gate_f[lstm_config::n_in] ;
        data_T gate_c[lstm_config::n_in] ;
        data_T gate_o[lstm_config::n_in] ;
        data_T gate_ic[lstm_config::n_in] ;
        data_T gate_forget[lstm_config::n_in] ;

         data_T h[lstm_config::n_in] /*hls_register*/;


        //intermediate variable cell calculation
        data_T cell_act_multp[lstm_config::n_in] ;
        data_T cell_act_add[lstm_config::n_in] ;


        //-----------Gate I Calculations
        //Weight multiplication
        multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, i_afterW , WI);
        //Bias addition
        add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(i_afterW,i_afterBias,BI);
        //Hidden Candidate
        multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, i_hiddenCand,RWI);
        add_vectors<data_T,data_T,CONFIG_T>(i_afterBias, i_hiddenCand,i_afterAdd);
        //Activation
        //hls_fpga insert recurrent_activation --- Gate I


        //-----------Gate F Calculations
        //Weight multiplication
        multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, f_afterW , WF);
        //Bias addition
        add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(f_afterW,f_afterBias,BF);
        //Hidden Candidate
        multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, f_hiddenCand,RWF);
        add_vectors<data_T,data_T,CONFIG_T>(f_afterBias, f_hiddenCand,f_afterAdd);
        //Activation
        //hls_fpga insert recurrent_activation --- Gate F


        //-----------Gate C Calculations
         //Weight multiplication
        multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, c_afterW , WC);
        //Bias addition
        add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(c_afterW,c_afterBias,BC);
        //Hidden Candidate
        multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, c_hiddenCand,RWC);
        add_vectors<data_T,data_T,CONFIG_T>(c_afterBias, c_hiddenCand,c_afterAdd);
        //Activation
        //hls_fpga insert activation  --- Gate C


        //-----------gate I and C multiply
        multiply_vectors<data_T,data_T,CONFIG_T>(gate_i, gate_c, gate_ic);

        //-----------Gate O Calculations
        multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, o_afterW,WO);
        add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(o_afterW,o_afterBias,BO);
        multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, o_hiddenCand,RWO);
        add_vectors<data_T,data_T,CONFIG_T>(o_afterBias, o_hiddenCand ,o_afterAdd);
        //hls_fpga insert recurrent_activation  --- Gate O


        //-----------Cell State Calculation
        multiply_vectors<data_T,data_T,CONFIG_T>(gate_f, cell_state, cell_act_multp);
        add_vectors<data_T,data_T,CONFIG_T>(gate_ic, cell_act_multp,cell_act_add);

        //-----------Forget gate Calculation
        //hls_fpga insert activation  --- Forget Gate

        multiply_vectors<data_T,data_T,CONFIG_T>(gate_o, gate_forget ,  h);


       OUTPUT_WRITE_LOOP:
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_in; x++) {
          hidden_state_o[x]=h[x];
          cell_state_o[x]=cell_act_add[x];
        }

        return;


}
#endif

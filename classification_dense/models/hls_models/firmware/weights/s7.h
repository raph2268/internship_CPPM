//Numpy array shape [1]
//Min 0.250000000000
//Max 0.250000000000
//Number of zeros 0

#ifndef S7_H_
#define S7_H_

#ifdef __INTELFPGA_COMPILER__
hls_init_on_powerup
#endif
static const exponent_scale7_t s7[2] = {{1, -2}, {1, -2}};

#endif

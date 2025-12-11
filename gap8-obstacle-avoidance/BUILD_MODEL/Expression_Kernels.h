#ifndef NANOFLOWNET_UNQUANTIZED_BASIC_KERNELS_H
#define NANOFLOWNET_UNQUANTIZED_BASIC_KERNELS_H
#include "Gap.h"
#include "math_funcs.h"
#include "Gap.h"

typedef struct {
    unsigned int I0;
    unsigned int I1;
    signed char *__restrict__  expr_0_in_0;
    signed char *__restrict__  expr_0_in_1;
    signed char *__restrict__  expr_0_in_2;
    signed char *__restrict__  expr_0_out_0;
} s213_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    signed char *__restrict__  expr_1_in_0;
    signed char *__restrict__  expr_1_in_1;
    signed char *__restrict__  expr_1_out_0;
} s223_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    signed char *__restrict__  expr_2_in_0;
    signed char *__restrict__  expr_2_in_1;
    signed char *__restrict__  expr_2_in_2;
    signed char *__restrict__  expr_2_out_0;
} s203_kernel_args_t;


void s213_kernel(s213_kernel_args_t *Args);

void s223_kernel(s223_kernel_args_t *Args);

void s203_kernel(s203_kernel_args_t *Args);


#endif // NANOFLOWNET_UNQUANTIZED_BASIC_KERNELS_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpointer-sign"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "Expression_Kernels.h"

static int CoreCountDynamic = 1;
static int ActiveCore = gap_ncore();

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap_ncore();
	Log2Core = gap_fl1(NCore);
	Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
	return Chunk;
}

#ifndef AT_NORM
#define AT_NORM(x, n)   gap_roundnorm_reg((x), (n))
#endif
#define ATLShift(x, n)  ((x) << (n))

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s213_kernel(s213_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_0_in_0 = Args->expr_0_in_0; // (32, 14, 20) int8 1.738 Q7
    signed char *__restrict__  expr_0_in_1 = Args->expr_0_in_1; // (32, 1, 1)   int8 1.008 Q7
    signed char *__restrict__  expr_0_in_2 = Args->expr_0_in_2; // (32, 14, 20) int8 2.402 Q7
    signed char *__restrict__  expr_0_out_0 = Args->expr_0_out_0; // (32, 14, 20) int8 4.823 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 14, 20) var shapes:
    // expr_0_out_0: (32, 14, 20) expr_0_in_0: (32, 14, 20) expr_0_in_1: (32,
    // 1, 1) expr_0_in_2: (32, 14, 20)
    // Iteration reduced to spaces ((0,), (1, 2))
    // Fixed spaces ()
    // Parameteric spaces ((0,), (1, 2))
    // Paralelized space (0,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        for (int i1=0; i1<I1; i1++) {
            // inputs expr_0_in_0: int8 1.738 Q7 expr_0_in_1: int8 1.008 Q7
            // expr_0_in_2: int8 2.402 Q7
            // expr_0_out_0 = Cast(Clip(Norm(Mul(Add(Norm(Mul(Mul(Cast(expr_0_in_0, int32), Cast(expr_0_in_1, int32)), [187]), [15]), Sub(Cast(expr_0_in_2, int32), [-128])), [255]), [9]), -128, 127), int8)
            expr_0_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((gap_roundnorm_reg(((((int)expr_0_in_0[(i0*I1)+i1])*((int)expr_0_in_1[i0]))*(187)), (15))+(((int)expr_0_in_2[(i0*I1)+i1])-(-128)))*(255)), (9))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s223_kernel(s223_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_1_in_0 = Args->expr_1_in_0; // (64, 28, 40) int8 1.927 Q7
    signed char *__restrict__  expr_1_in_1 = Args->expr_1_in_1; // (64, 1, 1)   int8 1.008 Q7
    signed char *__restrict__  expr_1_out_0 = Args->expr_1_out_0; // (64, 28, 40) int8 2.878 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 28, 40) var shapes:
    // expr_1_out_0: (64, 28, 40) expr_1_in_0: (64, 28, 40) expr_1_in_1: (64,
    // 1, 1)
    // Iteration reduced to spaces ((0,), (1, 2))
    // Fixed spaces ()
    // Parameteric spaces ((0,), (1, 2))
    // Paralelized space (0,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        for (int i1=0; i1<I1; i1++) {
            // inputs expr_1_in_0: int8 1.927 Q7 expr_1_in_1: int8 1.008 Q7
            // expr_1_out_0 = Cast(Clip(Norm(Mul(Norm(Add(Norm(Mul(Mul(Cast(expr_1_in_0, int32), Cast(expr_1_in_1, int32)), [129]), [6]), LShift(Cast(expr_1_in_0, int32), [8])), [1]), [171]), [15]), -128, 127), int8)
            expr_1_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_1_in_0[(i0*I1)+i1])*((int)expr_1_in_1[i0]))*(129)), (6))+(((int)expr_1_in_0[(i0*I1)+i1])<<(8))), (1))*(171)), (15))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s203_kernel(s203_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_2_in_0 = Args->expr_2_in_0; // (32, 7, 10) int8 5.676 Q7
    signed char *__restrict__  expr_2_in_1 = Args->expr_2_in_1; // (32, 1, 1)  int8 1.008 Q7
    signed char *__restrict__  expr_2_in_2 = Args->expr_2_in_2; // (32, 1, 1)  int8 5.281 Q7
    signed char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (32, 7, 10) int8 16.567 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 7, 10) var shapes:
    // expr_2_out_0: (32, 7, 10) expr_2_in_0: (32, 7, 10) expr_2_in_1: (32, 1,
    // 1) expr_2_in_2: (32, 1, 1)
    // Iteration reduced to spaces ((0,), (1, 2))
    // Fixed spaces ()
    // Parameteric spaces ((0,), (1, 2))
    // Paralelized space (0,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        for (int i1=0; i1<I1; i1++) {
            // inputs expr_2_in_0: int8 5.676 Q7 expr_2_in_1: int8 1.008 Q7
            // expr_2_in_2: int8 5.281 Q7
            // expr_2_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Mul(Cast(expr_2_in_0, int32), Cast(expr_2_in_1, int32)), Sub(Cast(expr_2_in_2, int32), [-128])), [6]), [233]), [15]), -128, 127), int8)
            expr_2_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_2_in_0[(i0*I1)+i1])*((int)expr_2_in_1[i0]))*(((int)expr_2_in_2[i0])-(-128))), (6))*(233)), (15))), ((7))));
        }
    }
    gap_waitbarrier(0);
}


#pragma GCC diagnostic pop
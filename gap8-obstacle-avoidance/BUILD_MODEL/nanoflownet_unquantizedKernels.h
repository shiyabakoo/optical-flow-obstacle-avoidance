#ifndef __NANOFLOWNET_UNQUANTIZEDKERNEL_H__
#define __NANOFLOWNET_UNQUANTIZEDKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "Gap.h"
#include "nanoflownet_unquantized.h"
#include "CNN_BasicKernels_SQ8.h"
#include "ResizeBasicKernels.h"
#include "Expression_Kernels.h"
#include "CNN_Copy.h"
#define _nanoflownet_unquantized_L1_Memory_SIZE 24528
#define _nanoflownet_unquantized_L2_Memory_SIZE 1088
#define _nanoflownet_unquantized_L2_Memory_Dyn_SIZE 28912
extern char *nanoflownet_unquantized_L1_Memory; /* Size given for generation: 24544 bytes, used: 24528 bytes */
extern char *nanoflownet_unquantized_L2_Memory; /* Size used for generation (static): 1088 bytes */
extern char *nanoflownet_unquantized_L2_Memory_Dyn; /* Size used for generation (dynamic): 28912 bytes */
extern void S1_Op_input_1_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S3_Op_input_2_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S7_Conv2d_2x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S10_Conv2d_16x2x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S11_AveragePool_3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S14_Conv2d_32x16x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S17_Conv2d_16x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S20_Conv2d_32x16x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S23_Conv2d_32x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S26_Conv2d_16x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S119_Conv2d_16x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S120_Conv2d_16x16x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S122_Op_CONV_2D_0_12_fusion_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S123_Conv2d_32x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S124_Op_CONCATENATION_0_19_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S125_Op_DEPTHWISE_CONV_2D_0_13_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S129_Conv2d_32x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S130_Conv2d_16x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S131_Op_CONCATENATION_0_19_qin1(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S132_Op_DEPTHWISE_CONV_2D_0_15_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S133_Conv2d_16x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S134_Conv2d_8x16x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S135_Conv2d_8x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S136_Conv2d_8x8x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S138_AveragePool_3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S139_Conv2d_64x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S140_Conv2d_64x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S141_Conv2d_64x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S142_Op_CONCATENATION_0_88_qin1(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S149_Conv2d_64x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S150_Conv2d_32x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S151_Conv2d_32x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S152_Conv2d_32x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S154_Conv2d_64x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S156_Conv2d_64x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S157_Conv2d_32x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S158_Op_CONCATENATION_0_36_qin1(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S159_Op_DEPTHWISE_CONV_2D_0_32_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S160_Conv2d_32x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S161_Conv2d_16x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S162_Op_CONCATENATION_0_36_qin2(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S163_Op_DEPTHWISE_CONV_2D_0_34_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S164_Conv2d_16x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S165_Conv2d_16x16x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S167_AveragePool_3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S168_Conv2d_128x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S169_Conv2d_128x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S170_Conv2d_128x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S175_Conv2d_128x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S176_Conv2d_64x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S177_Op_CONCATENATION_0_45_qin1(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S178_Op_DEPTHWISE_CONV_2D_0_43_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S179_Conv2d_64x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S180_Conv2d_64x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S182_Conv2d_128x256x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S184_Conv2d_128x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S185_Conv2d_64x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S186_Op_CONCATENATION_0_53_qin1(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S187_Op_DEPTHWISE_CONV_2D_0_49_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S188_Conv2d_64x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S189_Conv2d_32x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S190_Op_CONCATENATION_0_53_qin2(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S191_Op_DEPTHWISE_CONV_2D_0_51_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S192_Conv2d_32x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S193_Conv2d_32x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S195_Op_MEAN_0_54(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S197_Conv2d_32x256x1x1_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S198_Conv2d_256x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S199_Conv2d_32x256x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S200_Op_MEAN_0_62(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S202_Conv2d_32x32x1x1_Sigmoid(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S203_Op_expr_2(
		signed char * __restrict__ expr_2_in_0,
		signed char * __restrict__ expr_2_in_1,
		signed char * __restrict__ expr_2_in_2,
		signed char * __restrict__ expr_2_out_0);
extern void S204_Op_RESIZE_BILINEAR_0_71(
		signed char * In,
		signed char * Out);
extern void S205_Conv2d_32x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S206_Conv2d_32x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S208_Conv2d_128x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S209_Conv2d_32x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S210_Op_MEAN_0_76(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S212_Conv2d_32x32x1x1_Sigmoid(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S213_Op_expr_0(
		signed char * __restrict__ expr_0_in_0,
		signed char * __restrict__ expr_0_in_1,
		signed char * __restrict__ expr_0_in_2,
		signed char * __restrict__ expr_0_out_0);
extern void S214_Op_RESIZE_BILINEAR_0_85(
		signed char * In,
		signed char * Out);
extern void S215_Conv2d_32x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S216_Conv2d_32x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S218_Conv2d_64x96x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S219_Op_MEAN_0_90(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S221_Conv2d_16x64x1x1_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S222_Conv2d_64x16x1x1_Sigmoid(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S223_Op_expr_1(
		signed char * __restrict__ expr_1_in_0,
		signed char * __restrict__ expr_1_in_1,
		signed char * __restrict__ expr_1_out_0);
extern void S224_Conv2d_64x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S225_Conv2d_64x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S226_Conv2d_2x64x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern int nanoflownet_unquantizedCNN_Construct();
extern int nanoflownet_unquantizedCNN_Destruct();
extern int nanoflownet_unquantizedCNN_Memory(int Which);
extern int nanoflownet_unquantizedCNN(
		signed char * __restrict__ Input_1,
		signed char * __restrict__ Input_2,
		signed char * __restrict__ Output_1);
extern unsigned int AT_GraphPerf[92];
extern unsigned int AT_GraphPerf_CNN_Total;
extern char * AT_GraphNodeNames[92];
extern unsigned int AT_GraphOperInfosNames[92];
#endif

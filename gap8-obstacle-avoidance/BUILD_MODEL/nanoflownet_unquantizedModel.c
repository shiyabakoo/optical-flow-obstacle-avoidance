#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"
#include "ResizeGenerator.h"

#include "CNN_Copy_Generators.h"

void load_expressions_kernels() {
    LibKernelTemplate(
        "s213_kernel_args_t",
        CArgs(6,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_0_in_0"),
            TCArg("signed char *__restrict__ ", "expr_0_in_1"),
            TCArg("signed char *__restrict__ ", "expr_0_in_2"),
            TCArg("signed char *__restrict__ ", "expr_0_out_0")
        )
    );
    
    LibKernel(
        "s213_kernel",
        CALL_PARALLEL,
        0,
        "s213_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s223_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_1_in_0"),
            TCArg("signed char *__restrict__ ", "expr_1_in_1"),
            TCArg("signed char *__restrict__ ", "expr_1_out_0")
        )
    );
    
    LibKernel(
        "s223_kernel",
        CALL_PARALLEL,
        0,
        "s223_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s203_kernel_args_t",
        CArgs(6,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_2_in_0"),
            TCArg("signed char *__restrict__ ", "expr_2_in_1"),
            TCArg("signed char *__restrict__ ", "expr_2_in_2"),
            TCArg("signed char *__restrict__ ", "expr_2_out_0")
        )
    );
    
    LibKernel(
        "s203_kernel",
        CALL_PARALLEL,
        0,
        "s203_kernel_args_t",
        0
    );
}



int s213_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (32, 14, 20) spaces: ((0,), (1, 2)) 
        // parametric_spaces: ((0,), (1, 2)) 
        // exterior_shape: (32, 280.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 32, 8), IterParSpace(KER_ITER_D1, 280, 1), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(4,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_2"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_out_0")
        ),
        Calls(1,
            Call("s213_kernel", LOC_D1,
                Bindings(6,
                    K_ArgPar("expr_0_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_0_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_0_in_0", KER_ARG_TILE),
                    K_Arg("expr_0_in_1", KER_ARG_TILE),
                    K_Arg("expr_0_in_2", KER_ARG_TILE),
                    K_Arg("expr_0_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_0_out_0 axes: (0, 1)
        // var: expr_0_in_0 axes: (0, 1)
        // var: expr_0_in_1 axes: (0,)
        // var: expr_0_in_2 axes: (0, 1)
        KerArgs(4,
            KerArg("expr_0_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_0_out_0"),
            KerArg("expr_0_in_0",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_0"),
            KerArg("expr_0_in_1",  KerArgSpace(1, KER_ITER_D0),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_1"),
            KerArg("expr_0_in_2",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_2")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 8960, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 26912, 0);
        AddKernelArgDim(Name, "expr_0_in_0",  4, 32, 14, 20, 1);
        AddKernelArgDim(Name, "expr_0_in_1",  4, 32, 1, 1,   1);
        AddKernelArgDim(Name, "expr_0_in_2",  4, 32, 14, 20, 1);
        AddKernelArgDim(Name, "expr_0_out_0", 4, 32, 14, 20, 1);
    }
    return (Kernel!=0);
}
int s223_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (64, 28, 40) spaces: ((0,), (1, 2)) 
        // parametric_spaces: ((0,), (1, 2)) 
        // exterior_shape: (64, 1120.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 64, 8), IterParSpace(KER_ITER_D1, 1120, 1), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_out_0")
        ),
        Calls(1,
            Call("s223_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_1_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_1_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_1_in_0", KER_ARG_TILE),
                    K_Arg("expr_1_in_1", KER_ARG_TILE),
                    K_Arg("expr_1_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_1_out_0 axes: (0, 1)
        // var: expr_1_in_0 axes: (0, 1)
        // var: expr_1_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_1_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_1_out_0"),
            KerArg("expr_1_in_0",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_1_in_0"),
            KerArg("expr_1_in_1",  KerArgSpace(1, KER_ITER_D0),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_1_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 71680, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 143424, 0);
        AddKernelArgDim(Name, "expr_1_in_0",  4, 64, 28, 40, 1);
        AddKernelArgDim(Name, "expr_1_in_1",  4, 64, 1, 1,   1);
        AddKernelArgDim(Name, "expr_1_out_0", 4, 64, 28, 40, 1);
    }
    return (Kernel!=0);
}
int s203_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (32, 7, 10) spaces: ((0,), (1, 2)) 
        // parametric_spaces: ((0,), (1, 2)) 
        // exterior_shape: (32, 70.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 32, 8), IterParSpace(KER_ITER_D1, 70, 1), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(4,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_2_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_2_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_2_in_2"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_2_out_0")
        ),
        Calls(1,
            Call("s203_kernel", LOC_D1,
                Bindings(6,
                    K_ArgPar("expr_2_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_2_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_2_in_0", KER_ARG_TILE),
                    K_Arg("expr_2_in_1", KER_ARG_TILE),
                    K_Arg("expr_2_in_2", KER_ARG_TILE),
                    K_Arg("expr_2_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_2_out_0 axes: (0, 1)
        // var: expr_2_in_0 axes: (0, 1)
        // var: expr_2_in_1 axes: (0,)
        // var: expr_2_in_2 axes: (0,)
        KerArgs(4,
            KerArg("expr_2_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_2_out_0"),
            KerArg("expr_2_in_0",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_2_in_0"),
            KerArg("expr_2_in_1",  KerArgSpace(1, KER_ITER_D0),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_2_in_1"),
            KerArg("expr_2_in_2",  KerArgSpace(1, KER_ITER_D0),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_2_in_2")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 2240, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 4544, 0);
        AddKernelArgDim(Name, "expr_2_in_0",  4, 32, 7, 10, 1);
        AddKernelArgDim(Name, "expr_2_in_1",  4, 32, 1, 1,  1);
        AddKernelArgDim(Name, "expr_2_in_2",  4, 32, 1, 1,  1);
        AddKernelArgDim(Name, "expr_2_out_0", 4, 32, 7, 10, 1);
    }
    return (Kernel!=0);
}

void nanoflownet_unquantizedModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 7, "Gap.h", "nanoflownet_unquantized.h", "CNN_BasicKernels_SQ8.h", "ResizeBasicKernels.h", "CNN_BasicKernels_SQ8.h", "Expression_Kernels.h", "CNN_Copy.h");
    SetGeneratedFilesNames("nanoflownet_unquantizedKernels.c", "nanoflownet_unquantizedKernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "nanoflownet_unquantized_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "nanoflownet_unquantized_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "nanoflownet_unquantized_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "nanoflownet_unquantized_L3_Flash", "nanoflownet_unquantized_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();
    LoadResizeLibrary();
    LoadCNN_Copy_Library();
    load_expressions_kernels();

    
    // generator for input_1_copy
    CNN_Copy("S1_Op_input_1_copy", 0, 17920, 1);
    
    
    // generator for input_2_copy
    CNN_Copy("S3_Op_input_2_copy", 0, 17920, 1);
    
    CNN_GenControl_T gen_ctrl_S7_Conv2d_2x1x3x3;
    CNN_InitGenCtrl(&gen_ctrl_S7_Conv2d_2x1x3x3);
    CNN_SetGenCtrl(&gen_ctrl_S7_Conv2d_2x1x3x3, "PADTYPE", AT_OPT_VAL(1));
    // generator for DEPTHWISE_CONV_2D_0_1
    CNN_ConvolutionPoolAct_SQ8("S7_Conv2d_2x1x3x3", &gen_ctrl_S7_Conv2d_2x1x3x3,
                               4, 1,
                               2, 2, 160, 112,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S10_Conv2d_16x2x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S10_Conv2d_16x2x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S10_Conv2d_16x2x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_2_fusion
    CNN_ConvolutionPoolAct_SQ8("S10_Conv2d_16x2x1x1_Relu", &gen_ctrl_S10_Conv2d_16x2x1x1_Relu,
                               4, 1,
                               2, 16, 80, 56,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S11_AveragePool_3x3;
    CNN_InitGenCtrl(&gen_ctrl_S11_AveragePool_3x3);
    CNN_SetGenCtrl(&gen_ctrl_S11_AveragePool_3x3, "PADTYPE", AT_OPT_VAL(1));
    // generator for AVERAGE_POOL_2D_0_3
    CNN_PoolAct_SQ8("S11_AveragePool_3x3", &gen_ctrl_S11_AveragePool_3x3,
                    16, 80, 56,
                    KOP_AVGPOOL, 3, 3, 1, 1, 2, 2, 1,
                    KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S14_Conv2d_32x16x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S14_Conv2d_32x16x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S14_Conv2d_32x16x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_4_fusion
    CNN_ConvolutionPoolAct_SQ8("S14_Conv2d_32x16x1x1_Relu", &gen_ctrl_S14_Conv2d_32x16x1x1_Relu,
                               4, 1,
                               16, 32, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S17_Conv2d_16x1x3x3;
    CNN_InitGenCtrl(&gen_ctrl_S17_Conv2d_16x1x3x3);
    CNN_SetGenCtrl(&gen_ctrl_S17_Conv2d_16x1x3x3, "PADTYPE", AT_OPT_VAL(1));
    // generator for DEPTHWISE_CONV_2D_0_5
    CNN_ConvolutionPoolAct_SQ8("S17_Conv2d_16x1x3x3", &gen_ctrl_S17_Conv2d_16x1x3x3,
                               4, 1,
                               16, 16, 80, 56,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S20_Conv2d_32x16x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S20_Conv2d_32x16x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S20_Conv2d_32x16x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_6_fusion
    CNN_ConvolutionPoolAct_SQ8("S20_Conv2d_32x16x1x1_Relu", &gen_ctrl_S20_Conv2d_32x16x1x1_Relu,
                               4, 1,
                               16, 32, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_7
    CNN_ConvolutionPoolAct_SQ8("S23_Conv2d_32x1x3x3", 0,
                               4, 1,
                               32, 32, 40, 28,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S26_Conv2d_16x32x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S26_Conv2d_16x32x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S26_Conv2d_16x32x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_8_fusion
    CNN_ConvolutionPoolAct_SQ8("S26_Conv2d_16x32x1x1_Relu", &gen_ctrl_S26_Conv2d_16x32x1x1_Relu,
                               4, 1,
                               32, 16, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_9
    CNN_ConvolutionPoolAct_SQ8("S119_Conv2d_16x1x3x3", 0,
                               4, 1,
                               16, 16, 40, 28,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S120_Conv2d_16x16x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S120_Conv2d_16x16x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S120_Conv2d_16x16x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_10_fusion
    CNN_ConvolutionPoolAct_SQ8("S120_Conv2d_16x16x1x1_Relu", &gen_ctrl_S120_Conv2d_16x16x1x1_Relu,
                               4, 1,
                               16, 16, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    
    // generator for CONV_2D_0_12_fusion_qin0
    CNN_Convert("S122_Op_CONV_2D_0_12_fusion_qin0", 1, 1, 71680, KOP_CONVERT_FP_FP_SCALE);
    
    CNN_GenControl_T gen_ctrl_S123_Conv2d_32x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S123_Conv2d_32x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S123_Conv2d_32x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_12_fusion
    CNN_ConvolutionPoolAct_SQ8("S123_Conv2d_32x64x1x1_Relu", &gen_ctrl_S123_Conv2d_32x64x1x1_Relu,
                               4, 1,
                               64, 32, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    
    // generator for CONCATENATION_0_19_qin0
    CNN_Convert("S124_Op_CONCATENATION_0_19_qin0", 1, 1, 35840, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for DEPTHWISE_CONV_2D_0_13_qin0
    CNN_Convert("S125_Op_DEPTHWISE_CONV_2D_0_13_qin0", 1, 1, 35840, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for DEPTHWISE_CONV_2D_0_13
    CNN_ConvolutionPoolAct_SQ8("S129_Conv2d_32x1x3x3", 0,
                               4, 1,
                               32, 32, 40, 28,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S130_Conv2d_16x32x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S130_Conv2d_16x32x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S130_Conv2d_16x32x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_14_fusion
    CNN_ConvolutionPoolAct_SQ8("S130_Conv2d_16x32x1x1_Relu", &gen_ctrl_S130_Conv2d_16x32x1x1_Relu,
                               4, 1,
                               32, 16, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    
    // generator for CONCATENATION_0_19_qin1
    CNN_Convert("S131_Op_CONCATENATION_0_19_qin1", 1, 1, 17920, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for DEPTHWISE_CONV_2D_0_15_qin0
    CNN_Convert("S132_Op_DEPTHWISE_CONV_2D_0_15_qin0", 1, 1, 17920, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for DEPTHWISE_CONV_2D_0_15
    CNN_ConvolutionPoolAct_SQ8("S133_Conv2d_16x1x3x3", 0,
                               4, 1,
                               16, 16, 40, 28,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S134_Conv2d_8x16x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S134_Conv2d_8x16x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S134_Conv2d_8x16x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_16_fusion
    CNN_ConvolutionPoolAct_SQ8("S134_Conv2d_8x16x1x1_Relu", &gen_ctrl_S134_Conv2d_8x16x1x1_Relu,
                               4, 1,
                               16, 8, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_17
    CNN_ConvolutionPoolAct_SQ8("S135_Conv2d_8x1x3x3", 0,
                               4, 1,
                               8, 8, 40, 28,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S136_Conv2d_8x8x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S136_Conv2d_8x8x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S136_Conv2d_8x8x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_18_fusion
    CNN_ConvolutionPoolAct_SQ8("S136_Conv2d_8x8x1x1_Relu", &gen_ctrl_S136_Conv2d_8x8x1x1_Relu,
                               4, 1,
                               8, 8, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S138_AveragePool_3x3;
    CNN_InitGenCtrl(&gen_ctrl_S138_AveragePool_3x3);
    CNN_SetGenCtrl(&gen_ctrl_S138_AveragePool_3x3, "PADTYPE", AT_OPT_VAL(1));
    // generator for AVERAGE_POOL_2D_0_20
    CNN_PoolAct_SQ8("S138_AveragePool_3x3", &gen_ctrl_S138_AveragePool_3x3,
                    64, 40, 28,
                    KOP_AVGPOOL, 3, 3, 1, 1, 2, 2, 1,
                    KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S139_Conv2d_64x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S139_Conv2d_64x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S139_Conv2d_64x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_21_fusion
    CNN_ConvolutionPoolAct_SQ8("S139_Conv2d_64x64x1x1_Relu", &gen_ctrl_S139_Conv2d_64x64x1x1_Relu,
                               4, 1,
                               64, 64, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S140_Conv2d_64x1x3x3;
    CNN_InitGenCtrl(&gen_ctrl_S140_Conv2d_64x1x3x3);
    CNN_SetGenCtrl(&gen_ctrl_S140_Conv2d_64x1x3x3, "PADTYPE", AT_OPT_VAL(1));
    // generator for DEPTHWISE_CONV_2D_0_22
    CNN_ConvolutionPoolAct_SQ8("S140_Conv2d_64x1x3x3", &gen_ctrl_S140_Conv2d_64x1x3x3,
                               4, 1,
                               64, 64, 40, 28,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S141_Conv2d_64x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S141_Conv2d_64x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S141_Conv2d_64x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_23_fusion
    CNN_ConvolutionPoolAct_SQ8("S141_Conv2d_64x64x1x1_Relu", &gen_ctrl_S141_Conv2d_64x64x1x1_Relu,
                               4, 1,
                               64, 64, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    
    // generator for CONCATENATION_0_88_qin1
    CNN_Convert("S142_Op_CONCATENATION_0_88_qin1", 1, 1, 71680, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for DEPTHWISE_CONV_2D_0_24
    CNN_ConvolutionPoolAct_SQ8("S149_Conv2d_64x1x3x3", 0,
                               4, 1,
                               64, 64, 20, 14,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S150_Conv2d_32x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S150_Conv2d_32x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S150_Conv2d_32x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_25_fusion
    CNN_ConvolutionPoolAct_SQ8("S150_Conv2d_32x64x1x1_Relu", &gen_ctrl_S150_Conv2d_32x64x1x1_Relu,
                               4, 1,
                               64, 32, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_26
    CNN_ConvolutionPoolAct_SQ8("S151_Conv2d_32x1x3x3", 0,
                               4, 1,
                               32, 32, 20, 14,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S152_Conv2d_32x32x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S152_Conv2d_32x32x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S152_Conv2d_32x32x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_27_fusion
    CNN_ConvolutionPoolAct_SQ8("S152_Conv2d_32x32x1x1_Relu", &gen_ctrl_S152_Conv2d_32x32x1x1_Relu,
                               4, 1,
                               32, 32, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S154_Conv2d_64x128x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S154_Conv2d_64x128x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S154_Conv2d_64x128x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_29_fusion
    CNN_ConvolutionPoolAct_SQ8("S154_Conv2d_64x128x1x1_Relu", &gen_ctrl_S154_Conv2d_64x128x1x1_Relu,
                               4, 1,
                               128, 64, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_30
    CNN_ConvolutionPoolAct_SQ8("S156_Conv2d_64x1x3x3", 0,
                               4, 1,
                               64, 64, 20, 14,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S157_Conv2d_32x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S157_Conv2d_32x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S157_Conv2d_32x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_31_fusion
    CNN_ConvolutionPoolAct_SQ8("S157_Conv2d_32x64x1x1_Relu", &gen_ctrl_S157_Conv2d_32x64x1x1_Relu,
                               4, 1,
                               64, 32, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    
    // generator for CONCATENATION_0_36_qin1
    CNN_Convert("S158_Op_CONCATENATION_0_36_qin1", 1, 1, 8960, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for DEPTHWISE_CONV_2D_0_32_qin0
    CNN_Convert("S159_Op_DEPTHWISE_CONV_2D_0_32_qin0", 1, 1, 8960, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for DEPTHWISE_CONV_2D_0_32
    CNN_ConvolutionPoolAct_SQ8("S160_Conv2d_32x1x3x3", 0,
                               4, 1,
                               32, 32, 20, 14,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S161_Conv2d_16x32x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S161_Conv2d_16x32x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S161_Conv2d_16x32x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_33_fusion
    CNN_ConvolutionPoolAct_SQ8("S161_Conv2d_16x32x1x1_Relu", &gen_ctrl_S161_Conv2d_16x32x1x1_Relu,
                               4, 1,
                               32, 16, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    
    // generator for CONCATENATION_0_36_qin2
    CNN_Convert("S162_Op_CONCATENATION_0_36_qin2", 1, 1, 4480, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for DEPTHWISE_CONV_2D_0_34_qin0
    CNN_Convert("S163_Op_DEPTHWISE_CONV_2D_0_34_qin0", 1, 1, 4480, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for DEPTHWISE_CONV_2D_0_34
    CNN_ConvolutionPoolAct_SQ8("S164_Conv2d_16x1x3x3", 0,
                               4, 1,
                               16, 16, 20, 14,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S165_Conv2d_16x16x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S165_Conv2d_16x16x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S165_Conv2d_16x16x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_35_fusion
    CNN_ConvolutionPoolAct_SQ8("S165_Conv2d_16x16x1x1_Relu", &gen_ctrl_S165_Conv2d_16x16x1x1_Relu,
                               4, 1,
                               16, 16, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S167_AveragePool_3x3;
    CNN_InitGenCtrl(&gen_ctrl_S167_AveragePool_3x3);
    CNN_SetGenCtrl(&gen_ctrl_S167_AveragePool_3x3, "PADTYPE", AT_OPT_VAL(1));
    // generator for AVERAGE_POOL_2D_0_37
    CNN_PoolAct_SQ8("S167_AveragePool_3x3", &gen_ctrl_S167_AveragePool_3x3,
                    128, 20, 14,
                    KOP_AVGPOOL, 3, 3, 1, 1, 2, 2, 1,
                    KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S168_Conv2d_128x128x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S168_Conv2d_128x128x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S168_Conv2d_128x128x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_38_fusion
    CNN_ConvolutionPoolAct_SQ8("S168_Conv2d_128x128x1x1_Relu", &gen_ctrl_S168_Conv2d_128x128x1x1_Relu,
                               4, 1,
                               128, 128, 10, 7,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S169_Conv2d_128x1x3x3;
    CNN_InitGenCtrl(&gen_ctrl_S169_Conv2d_128x1x3x3);
    CNN_SetGenCtrl(&gen_ctrl_S169_Conv2d_128x1x3x3, "PADTYPE", AT_OPT_VAL(1));
    // generator for DEPTHWISE_CONV_2D_0_39
    CNN_ConvolutionPoolAct_SQ8("S169_Conv2d_128x1x3x3", &gen_ctrl_S169_Conv2d_128x1x3x3,
                               4, 1,
                               128, 128, 20, 14,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S170_Conv2d_128x128x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S170_Conv2d_128x128x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S170_Conv2d_128x128x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_40_fusion
    CNN_ConvolutionPoolAct_SQ8("S170_Conv2d_128x128x1x1_Relu", &gen_ctrl_S170_Conv2d_128x128x1x1_Relu,
                               4, 1,
                               128, 128, 10, 7,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_41
    CNN_ConvolutionPoolAct_SQ8("S175_Conv2d_128x1x3x3", 0,
                               4, 1,
                               128, 128, 10, 7,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S176_Conv2d_64x128x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S176_Conv2d_64x128x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S176_Conv2d_64x128x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_42_fusion
    CNN_ConvolutionPoolAct_SQ8("S176_Conv2d_64x128x1x1_Relu", &gen_ctrl_S176_Conv2d_64x128x1x1_Relu,
                               4, 1,
                               128, 64, 10, 7,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    
    // generator for CONCATENATION_0_45_qin1
    CNN_Convert("S177_Op_CONCATENATION_0_45_qin1", 1, 1, 4480, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for DEPTHWISE_CONV_2D_0_43_qin0
    CNN_Convert("S178_Op_DEPTHWISE_CONV_2D_0_43_qin0", 1, 1, 4480, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for DEPTHWISE_CONV_2D_0_43
    CNN_ConvolutionPoolAct_SQ8("S179_Conv2d_64x1x3x3", 0,
                               4, 1,
                               64, 64, 10, 7,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S180_Conv2d_64x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S180_Conv2d_64x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S180_Conv2d_64x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_44_fusion
    CNN_ConvolutionPoolAct_SQ8("S180_Conv2d_64x64x1x1_Relu", &gen_ctrl_S180_Conv2d_64x64x1x1_Relu,
                               4, 1,
                               64, 64, 10, 7,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S182_Conv2d_128x256x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S182_Conv2d_128x256x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S182_Conv2d_128x256x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_46_fusion
    CNN_ConvolutionPoolAct_SQ8("S182_Conv2d_128x256x1x1_Relu", &gen_ctrl_S182_Conv2d_128x256x1x1_Relu,
                               4, 1,
                               256, 128, 10, 7,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_47
    CNN_ConvolutionPoolAct_SQ8("S184_Conv2d_128x1x3x3", 0,
                               4, 1,
                               128, 128, 10, 7,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S185_Conv2d_64x128x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S185_Conv2d_64x128x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S185_Conv2d_64x128x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_48_fusion
    CNN_ConvolutionPoolAct_SQ8("S185_Conv2d_64x128x1x1_Relu", &gen_ctrl_S185_Conv2d_64x128x1x1_Relu,
                               4, 1,
                               128, 64, 10, 7,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    
    // generator for CONCATENATION_0_53_qin1
    CNN_Convert("S186_Op_CONCATENATION_0_53_qin1", 1, 1, 4480, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for DEPTHWISE_CONV_2D_0_49_qin0
    CNN_Convert("S187_Op_DEPTHWISE_CONV_2D_0_49_qin0", 1, 1, 4480, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for DEPTHWISE_CONV_2D_0_49
    CNN_ConvolutionPoolAct_SQ8("S188_Conv2d_64x1x3x3", 0,
                               4, 1,
                               64, 64, 10, 7,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S189_Conv2d_32x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S189_Conv2d_32x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S189_Conv2d_32x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_50_fusion
    CNN_ConvolutionPoolAct_SQ8("S189_Conv2d_32x64x1x1_Relu", &gen_ctrl_S189_Conv2d_32x64x1x1_Relu,
                               4, 1,
                               64, 32, 10, 7,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    
    // generator for CONCATENATION_0_53_qin2
    CNN_Convert("S190_Op_CONCATENATION_0_53_qin2", 1, 1, 2240, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for DEPTHWISE_CONV_2D_0_51_qin0
    CNN_Convert("S191_Op_DEPTHWISE_CONV_2D_0_51_qin0", 1, 1, 2240, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for DEPTHWISE_CONV_2D_0_51
    CNN_ConvolutionPoolAct_SQ8("S192_Conv2d_32x1x3x3", 0,
                               4, 1,
                               32, 32, 10, 7,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S193_Conv2d_32x32x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S193_Conv2d_32x32x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S193_Conv2d_32x32x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_52_fusion
    CNN_ConvolutionPoolAct_SQ8("S193_Conv2d_32x32x1x1_Relu", &gen_ctrl_S193_Conv2d_32x32x1x1_Relu,
                               4, 1,
                               32, 32, 10, 7,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for MEAN_0_54
    CNN_GlobalPoolAct_SQ8("S195_Op_MEAN_0_54", 0,
                          256, 5, 14,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S197_Conv2d_32x256x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S197_Conv2d_32x256x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S197_Conv2d_32x256x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_59_fusion
    CNN_ConvolutionPoolAct_SQ8("S197_Conv2d_32x256x1x1_Relu", &gen_ctrl_S197_Conv2d_32x256x1x1_Relu,
                               4, 1,
                               256, 32, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    // generator for DEPTHWISE_CONV_2D_0_60
    CNN_ConvolutionPoolAct_SQ8("S198_Conv2d_256x1x3x3", 0,
                               4, 1,
                               256, 256, 10, 7,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S199_Conv2d_32x256x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S199_Conv2d_32x256x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S199_Conv2d_32x256x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_61_fusion
    CNN_ConvolutionPoolAct_SQ8("S199_Conv2d_32x256x1x1_Relu", &gen_ctrl_S199_Conv2d_32x256x1x1_Relu,
                               4, 1,
                               256, 32, 10, 7,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for MEAN_0_62
    CNN_GlobalPoolAct_SQ8("S200_Op_MEAN_0_62", 0,
                          32, 5, 14,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S202_Conv2d_32x32x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S202_Conv2d_32x32x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S202_Conv2d_32x32x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_67_fusion
    CNN_ConvolutionPoolAct_SQ8("S202_Conv2d_32x32x1x1_Sigmoid", &gen_ctrl_S202_Conv2d_32x32x1x1_Sigmoid,
                               4, 1,
                               32, 32, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_SIGMOID);
    
    
    // generator for expr_2
    s203_kernel_gen("S203_Op_expr_2");
    
    
    // generator for RESIZE_BILINEAR_0_71
    GenerateResizeMultiChannel("S204_Op_RESIZE_BILINEAR_0_71", 10, 7, 20, 14, 32, SIGNED_INOUT, KOP_BILINEAR_RESIZE);
    
    // generator for DEPTHWISE_CONV_2D_0_72
    CNN_ConvolutionPoolAct_SQ8("S205_Conv2d_32x1x3x3", 0,
                               4, 1,
                               32, 32, 20, 14,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S206_Conv2d_32x32x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S206_Conv2d_32x32x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S206_Conv2d_32x32x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_73_fusion
    CNN_ConvolutionPoolAct_SQ8("S206_Conv2d_32x32x1x1_Relu", &gen_ctrl_S206_Conv2d_32x32x1x1_Relu,
                               4, 1,
                               32, 32, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    // generator for DEPTHWISE_CONV_2D_0_74
    CNN_ConvolutionPoolAct_SQ8("S208_Conv2d_128x1x3x3", 0,
                               4, 1,
                               128, 128, 20, 14,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S209_Conv2d_32x128x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S209_Conv2d_32x128x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S209_Conv2d_32x128x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_75_fusion
    CNN_ConvolutionPoolAct_SQ8("S209_Conv2d_32x128x1x1_Relu", &gen_ctrl_S209_Conv2d_32x128x1x1_Relu,
                               4, 1,
                               128, 32, 20, 14,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for MEAN_0_76
    CNN_GlobalPoolAct_SQ8("S210_Op_MEAN_0_76", 0,
                          32, 10, 28,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S212_Conv2d_32x32x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S212_Conv2d_32x32x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S212_Conv2d_32x32x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_81_fusion
    CNN_ConvolutionPoolAct_SQ8("S212_Conv2d_32x32x1x1_Sigmoid", &gen_ctrl_S212_Conv2d_32x32x1x1_Sigmoid,
                               4, 1,
                               32, 32, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_SIGMOID);
    
    
    // generator for expr_0
    s213_kernel_gen("S213_Op_expr_0");
    
    
    // generator for RESIZE_BILINEAR_0_85
    GenerateResizeMultiChannel("S214_Op_RESIZE_BILINEAR_0_85", 20, 14, 40, 28, 32, SIGNED_INOUT, KOP_BILINEAR_RESIZE);
    
    // generator for DEPTHWISE_CONV_2D_0_86
    CNN_ConvolutionPoolAct_SQ8("S215_Conv2d_32x1x3x3", 0,
                               4, 1,
                               32, 32, 40, 28,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S216_Conv2d_32x32x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S216_Conv2d_32x32x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S216_Conv2d_32x32x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_87_fusion
    CNN_ConvolutionPoolAct_SQ8("S216_Conv2d_32x32x1x1_Relu", &gen_ctrl_S216_Conv2d_32x32x1x1_Relu,
                               4, 1,
                               32, 32, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S218_Conv2d_64x96x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S218_Conv2d_64x96x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S218_Conv2d_64x96x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_89_fusion
    CNN_ConvolutionPoolAct_SQ8("S218_Conv2d_64x96x1x1_Relu", &gen_ctrl_S218_Conv2d_64x96x1x1_Relu,
                               4, 1,
                               96, 64, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for MEAN_0_90
    CNN_GlobalPoolAct_SQ8("S219_Op_MEAN_0_90", 0,
                          64, 20, 56,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S221_Conv2d_16x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S221_Conv2d_16x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S221_Conv2d_16x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_95_fusion
    CNN_ConvolutionPoolAct_SQ8("S221_Conv2d_16x64x1x1_Relu", &gen_ctrl_S221_Conv2d_16x64x1x1_Relu,
                               4, 1,
                               64, 16, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S222_Conv2d_64x16x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S222_Conv2d_64x16x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S222_Conv2d_64x16x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_96_fusion
    CNN_ConvolutionPoolAct_SQ8("S222_Conv2d_64x16x1x1_Sigmoid", &gen_ctrl_S222_Conv2d_64x16x1x1_Sigmoid,
                               4, 1,
                               16, 64, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_SIGMOID);
    
    
    // generator for expr_1
    s223_kernel_gen("S223_Op_expr_1");
    
    // generator for DEPTHWISE_CONV_2D_0_100
    CNN_ConvolutionPoolAct_SQ8("S224_Conv2d_64x1x3x3", 0,
                               4, 1,
                               64, 64, 40, 28,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S225_Conv2d_64x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S225_Conv2d_64x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S225_Conv2d_64x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_101_fusion
    CNN_ConvolutionPoolAct_SQ8("S225_Conv2d_64x64x1x1_Relu", &gen_ctrl_S225_Conv2d_64x64x1x1_Relu,
                               4, 1,
                               64, 64, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S226_Conv2d_2x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S226_Conv2d_2x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S226_Conv2d_2x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_102
    CNN_ConvolutionPoolAct_SQ8("S226_Conv2d_2x64x1x1", &gen_ctrl_S226_Conv2d_2x64x1x1,
                               4, 1,
                               64, 2, 40, 28,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("nanoflownet_unquantizedCNN",
        /* Arguments either passed or globals */
            CArgs(331,
                TCArgInfo("signed char * __restrict__", "Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Input_2", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2dse", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2dse.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modeloutputconv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modeloutputconv2d.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S7_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S7_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S7_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S7_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S7_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S7_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2dse_1b6142d5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2dse_1b6142d5.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2dbias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2dbias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S10_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S10_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S10_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S10_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00447 out: 0.00447  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S10_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S10_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S11_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S11_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2dconv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2dconv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2dbias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Conv2dbias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S14_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S14_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S14_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S14_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00737 out: 0.00737  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S14_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S14_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelconv2d_10conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_10conv2d.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S17_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S17_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S17_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S17_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S17_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S17_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_f875d3fa", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_f875d3fa.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_1bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_1bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S20_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S20_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S20_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S20_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00710 out: 0.00710  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S20_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S20_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_6a286dd7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_6a286dd7.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S23_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S23_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S23_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S23_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S23_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S23_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_1d2f6f73", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_1d2f6f73.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_2bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_2bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S26_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S26_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S26_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S26_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00737 out: 0.00737  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S26_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S26_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_3.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_3_34a7c8eb", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_3_34a7c8eb.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_3bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_3bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_1conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_1conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2d_1bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Conv2d_1bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_4.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_4_6faf5601", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_4_6faf5601.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_4bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_4bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_5.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_5_9edcbfcd", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_5_9edcbfcd.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_5bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_5bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_6_08bb4813", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_6_08bb4813.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_6.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_6_0909dd70", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_6_0909dd70.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_6bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_6bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_2conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_2conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2d_2bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Conv2d_2bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_8.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_0306fd97", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_0306fd97.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_8_05171b1c", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_8_05171b1c.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_8bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_8bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_9.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_9_b185726b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_9_b185726b.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_9bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_9bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_34edfbff", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_34edfbff.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_c2442fd6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_c2442fd6.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_10bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_10bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_3conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_3conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2d_3bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Conv2d_3bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_749f00a1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_749f00a1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_dfd1a27d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_dfd1a27d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_11bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_11bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_303b3c86", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_303b3c86.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_4ee40c40", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_4ee40c40.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_12bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_12bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_61ce1792", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_61ce1792.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_a85ca358", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_a85ca358.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_13bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_13bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_4conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_4conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2d_4bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Conv2d_4bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_f63cde89", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_f63cde89.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_70a03dc2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_70a03dc2.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_72c6547f", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_72c6547f.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_14bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_14bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_1b9fd017", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_1b9fd017.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_d01ce067", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_d01ce067.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_15bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_15bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_7e9971c1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_7e9971c1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_75db895d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_75db895d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_16bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_16bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_5conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_5conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2d_5bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Conv2d_5bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_90fdc921", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_90fdc921.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_70b8f72e", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_70b8f72e.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_17bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_17bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_a7079b99", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_a7079b99.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_f2ce61c9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_f2ce61c9.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_18bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_18bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_8ec95cbf", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_8ec95cbf.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_1_dac090c6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_1_dac090c6.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_19bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_19bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_6conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_6conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2d_6bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Conv2d_6bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_90e98147", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_90e98147.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_2ea7e0c0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_2ea7e0c0.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_294b40b0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_294b40b0.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_20bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_20bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_7conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_7conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelconv2d_7biasaddmod", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_7biasaddmod.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_57f81003", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_57f81003.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_d0fcba95", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_d0fcba95.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_21bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_21bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_c1d436ee", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_c1d436ee.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_5ce43f07", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_5ce43f07.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_23bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_23bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_8conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_8conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelconv2d_8biasaddmod", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_8biasaddmod.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_41884a31", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_41884a31.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_8e412014", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_8e412014.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_24bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_24bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_9conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_9conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2d_9bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Conv2d_9bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_10conv2d1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_10conv2d1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2d_10bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Conv2d_10bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelconv2d_11conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_11conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelconv2d_11biasaddmo", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_11biasaddmo.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_53045d95", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_53045d95.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modelseparable_conv2d_2_f68e0cfc", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_f68e0cfc.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_26bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Separable_conv2d_26bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "Model_2modeloutputconv2d1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modeloutputconv2d1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modeloutputbiasaddmodel", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modeloutputbiasaddmodel.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelconv2d_10conv2d_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_10conv2d_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S119_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S119_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S119_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S119_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S119_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S119_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S120_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S120_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S120_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S120_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00737 out: 0.00737  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S120_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S120_Infos.tensor", 1, 1, 8, 0)),
                // in q: -0.94<(i8-0.00)*0.00737075<0.94 forced out_q: 0.00<(i8--128.00)*0.00367092<0.94
                TCArgInfo("signed char * __restrict__", "S122_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S122_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S123_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S123_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S123_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S123_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00230 out: 0.00230  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S123_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S123_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00230397<0.59 out_q: -0.69<(i8-0.00)*0.00542670<0.69 forced
                TCArgInfo("signed char * __restrict__", "S124_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S124_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00230397<0.59 out_q: -0.59<(i8-0.00)*0.00462608<0.59 forced
                TCArgInfo("signed char * __restrict__", "S125_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S125_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelconv2d_10conv2d_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_10conv2d_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelconv2d_10conv2d_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelconv2d_10conv2d_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S129_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S129_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S129_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S129_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S129_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S129_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S130_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S130_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S130_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S130_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00240 out: 0.00240  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S130_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S130_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00239889<0.61 out_q: -0.69<(i8-0.00)*0.00542670<0.69 forced
                TCArgInfo("signed char * __restrict__", "S131_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S131_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00239889<0.61 out_q: -0.62<(i8-0.00)*0.00481666<0.61 forced
                TCArgInfo("signed char * __restrict__", "S132_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S132_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S133_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S133_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S133_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S133_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S133_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S133_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S134_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S134_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S134_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S134_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00543 out: 0.00543  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S134_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S134_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S135_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S135_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S135_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S135_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S135_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S135_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S136_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S136_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S136_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S136_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00543 out: 0.00543  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S136_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S136_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S138_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S138_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S139_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S139_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S139_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S139_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00881 out: 0.00881  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S139_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S139_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S140_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S140_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S140_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S140_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S140_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S140_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S141_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S141_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S141_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S141_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00837 out: 0.00837  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S141_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S141_Infos.tensor", 1, 1, 8, 0)),
                // in q: -0.69<(i8-0.00)*0.00542670<0.69 forced out_q: 0.00<(i8--128.00)*0.00740498<1.89
                TCArgInfo("signed char * __restrict__", "S142_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S142_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_4.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_5.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_6.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_0306fd97_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_0306fd97_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S149_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S149_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S149_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S149_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S149_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S149_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S150_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S150_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S150_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S150_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00881 out: 0.00881  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S150_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S150_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S151_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S151_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S151_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S151_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S151_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S151_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S152_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S152_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S152_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S152_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00881 out: 0.00881  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S152_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S152_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S154_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S154_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S154_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S154_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01670 out: 0.01670  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S154_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S154_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_0306fd97_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_0306fd97_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S156_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S156_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S156_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S156_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S156_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S156_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S157_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S157_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S157_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S157_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00653 out: 0.00653  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S157_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S157_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00653194<1.67 out_q: -2.14<(i8-0.00)*0.01669604<2.12 forced
                TCArgInfo("signed char * __restrict__", "S158_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S158_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00653194<1.67 out_q: -1.68<(i8-0.00)*0.01311532<1.67 forced
                TCArgInfo("signed char * __restrict__", "S159_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S159_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S160_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S160_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S160_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S160_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S160_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S160_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S161_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S161_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S161_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S161_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00367 out: 0.00367  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S161_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S161_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00367384<0.94 out_q: -2.14<(i8-0.00)*0.01669604<2.12 forced
                TCArgInfo("signed char * __restrict__", "S162_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S162_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00367384<0.94 out_q: -0.94<(i8-0.00)*0.00737660<0.94 forced
                TCArgInfo("signed char * __restrict__", "S163_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S163_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S164_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S164_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S164_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S164_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S164_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S164_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S165_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S165_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S165_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S165_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01670 out: 0.01670  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S165_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S165_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S167_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S167_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S168_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S168_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S168_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S168_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01658 out: 0.01658  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S168_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S168_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S169_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S169_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S169_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S169_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S169_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S169_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S170_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S170_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S170_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S170_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01761 out: 0.01761  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S170_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S170_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_0306fd97_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_0306fd97_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_0306fd97_4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_0306fd97_4.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_0306fd97_5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_0306fd97_5.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_70a03dc2_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_70a03dc2_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S175_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S175_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S175_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S175_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S175_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S175_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S176_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S176_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S176_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S176_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01338 out: 0.01338  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S176_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S176_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.01338032<3.41 out_q: -2.12<(i8-0.00)*0.01657579<2.11
                TCArgInfo("signed char * __restrict__", "S177_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S177_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.01338032<3.41 out_q: -3.44<(i8-0.00)*0.02686599<3.41 forced
                TCArgInfo("signed char * __restrict__", "S178_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S178_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S179_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S179_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S179_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S179_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S179_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S179_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S180_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S180_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S180_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S180_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01658 out: 0.01658  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S180_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S180_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S182_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S182_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S182_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S182_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04591 out: 0.04591  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S182_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S182_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_70a03dc2_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_70a03dc2_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S184_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S184_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S184_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S184_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S184_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S184_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S185_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S185_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S185_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S185_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02106 out: 0.02106  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S185_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S185_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.02106018<5.37 out_q: -5.88<(i8-0.00)*0.04591471<5.83 forced
                TCArgInfo("signed char * __restrict__", "S186_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S186_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.02106018<5.37 out_q: -5.41<(i8-0.00)*0.04228619<5.37 forced
                TCArgInfo("signed char * __restrict__", "S187_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S187_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S188_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S188_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S188_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S188_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S188_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S188_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S189_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S189_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S189_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S189_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00910 out: 0.00910  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S189_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S189_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00910447<2.32 out_q: -5.88<(i8-0.00)*0.04591471<5.83 forced
                TCArgInfo("signed char * __restrict__", "S190_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S190_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(i8--128.00)*0.00910447<2.32 out_q: -2.34<(i8-0.00)*0.01828063<2.32 forced
                TCArgInfo("signed char * __restrict__", "S191_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S191_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S192_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S192_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S192_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S192_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S192_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S192_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S193_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S193_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S193_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S193_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04591 out: 0.04591  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S193_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S193_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [5] GLOBAL_SUM_SCALEN: [9]
                TCArgInfo("signed char * __restrict__", "S195_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S195_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S197_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S197_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S197_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S197_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04126 out: 0.04126  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S197_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S197_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S198_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S198_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S198_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S198_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S198_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S198_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S199_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S199_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S199_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S199_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04435 out: 0.04435  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S199_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S199_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [49] GLOBAL_SUM_SCALEN: [12]
                TCArgInfo("signed char * __restrict__", "S200_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S200_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S202_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S202_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S202_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S202_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00787  actscale: [127] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S202_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S202_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S205_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S205_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S205_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S205_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S205_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S205_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S206_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S206_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S206_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S206_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01876 out: 0.01876  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S206_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S206_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Model_2modelseparable_conv2d_2_70a03dc2_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/Model_2modelseparable_conv2d_2_70a03dc2_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S208_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S208_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S208_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S208_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S208_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S208_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S209_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S209_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S209_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S209_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01358 out: 0.01358  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S209_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S209_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [71] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S210_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S210_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S212_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S212_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S212_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S212_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00787  actscale: [127] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S212_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S212_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S215_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S215_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S215_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S215_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S215_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S215_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S216_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S216_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S216_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S216_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00740 out: 0.00740  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S216_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S216_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S218_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S218_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S218_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S218_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01505 out: 0.01505  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S218_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S218_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [91] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S219_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S219_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S221_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S221_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S221_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S221_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00222 out: 0.00222  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S221_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S221_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S222_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S222_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S222_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S222_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00787  actscale: [127] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S222_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S222_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S224_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S224_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S224_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S224_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S224_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S224_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S225_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S225_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S225_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S225_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00732 out: 0.00732  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S225_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S225_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S226_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S226_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S226_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S226_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S226_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL/tensors/S226_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(73,
            TCArgInfo("signed char * __restrict__", "S4_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S7_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S10_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S11_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S17_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S20_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S23_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S119_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S121_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S122_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S123_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S125_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S129_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S130_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S132_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S133_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S135_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S137_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S138_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S140_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S141_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S149_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S151_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S153_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S156_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S157_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S159_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S160_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S161_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S163_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S164_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S166_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S167_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S169_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S170_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S175_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S176_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S178_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S179_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S181_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S184_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S185_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S187_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S188_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S189_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S191_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S192_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S194_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S195_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S197_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S198_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S199_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S200_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S202_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S203_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S204_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S205_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S206_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S208_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S209_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S210_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S212_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S213_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S214_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S215_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S217_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S218_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S219_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S221_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S222_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S223_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S224_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S225_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    AddStackedTensors("S4_Output", 2, "S1_Output", "S3_Output");
    AddStackedTensors("S121_Output", 3, "S14_Output", "S26_Output", "S120_Output");
    AddStackedTensors("S137_Output", 4, "S124_Output", "S131_Output", "S134_Output", "S136_Output");
    AddStackedTensors("S153_Output", 3, "S139_Output", "S150_Output", "S152_Output");
    AddStackedTensors("S166_Output", 4, "S154_Output", "S158_Output", "S162_Output", "S165_Output");
    AddStackedTensors("S181_Output", 3, "S168_Output", "S177_Output", "S180_Output");
    AddStackedTensors("S194_Output", 4, "S182_Output", "S186_Output", "S190_Output", "S193_Output");
    AddStackedTensors("S217_Output", 2, "S216_Output", "S142_Output");

    // Node input_1_copy inq -0.82<(i8-0.00)*0.00644342<0.82 outq -0.82<(i8-0.00)*0.00644342<0.82
    AddNode("S1_Op_input_1_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_OUT, "S1_Output", 0)
        )
    );
    // Node input_2_copy inq -0.82<(i8-0.00)*0.00644342<0.82 outq -0.82<(i8-0.00)*0.00644342<0.82
    AddNode("S3_Op_input_2_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "Input_2", 0),
            GNodeArg(GNA_OUT, "S3_Output", 0)
        )
    );
    // Node S7_Conv2d_2x1x3x3 inq -0.82<(i8-0.00)*0.00644342<0.82 forced weightsq chan<(i8-0.00)*chan<chan outq -0.55<(i8-3.00)*0.00420315<0.52 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S7_Conv2d_2x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S4_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2dse", 0),
            GNodeArg(GNA_IN, "Model_2modeloutputconv2d", 0),
            GNodeArg(GNA_OUT, "S7_Output", 0),
            GNodeArg(GNA_IN, "S7_Mul_scale", 0),
            GNodeArg(GNA_IN, "S7_Mul_shift", 0),
            GNodeArg(GNA_IN, "S7_Infos", 0)
        )
    );
    // Node S10_Conv2d_16x2x1x1_Relu inq -0.55<(i8-3.00)*0.00420315<0.52 weightsq chan<(i8-0.00)*chan<chan outq -0.57<(i8-0.00)*0.00447123<0.57 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S10_Conv2d_16x2x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S7_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2dse_1b6142d5", 0),
            GNodeArg(GNA_IN, "Separable_conv2dbias", 0),
            GNodeArg(GNA_OUT, "S10_Output", 0),
            GNodeArg(GNA_IN, "S10_Mul_scale", 0),
            GNodeArg(GNA_IN, "S10_Mul_shift", 0),
            GNodeArg(GNA_IN, "S10_Infos", 0)
        )
    );
    // Node AVERAGE_POOL_2D_0_3 inq -0.57<(i8-0.00)*0.00447123<0.57 forced outq -0.57<(i8-0.00)*0.00447123<0.57 forced
    AddNode("S11_AveragePool_3x3",
        Bindings(3,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_OUT, "S11_Output", 0),
            GNodeArg(GNA_IN, "S11_Infos", 0)
        )
    );
    // Node S14_Conv2d_32x16x1x1_Relu inq -0.57<(i8-0.00)*0.00447123<0.57 forced weightsq chan<(i8-0.00)*chan<chan outq -0.94<(i8-0.00)*0.00737075<0.94 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S14_Conv2d_32x16x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S11_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2dconv2d", 0),
            GNodeArg(GNA_IN, "Conv2dbias", 0),
            GNodeArg(GNA_OUT, "S14_Output", 0),
            GNodeArg(GNA_IN, "S14_Mul_scale", 0),
            GNodeArg(GNA_IN, "S14_Mul_shift", 0),
            GNodeArg(GNA_IN, "S14_Infos", 0)
        )
    );
    // Node S17_Conv2d_16x1x3x3 inq -0.57<(i8-0.00)*0.00447123<0.57 forced weightsq chan<(i8-0.00)*chan<chan outq -0.35<(i8--44.00)*0.00414742<0.71 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S17_Conv2d_16x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_10conv2d", 0),
            GNodeArg(GNA_OUT, "S17_Output", 0),
            GNodeArg(GNA_IN, "S17_Mul_scale", 0),
            GNodeArg(GNA_IN, "S17_Mul_shift", 0),
            GNodeArg(GNA_IN, "S17_Infos", 0)
        )
    );
    // Node S20_Conv2d_32x16x1x1_Relu inq -0.35<(i8--44.00)*0.00414742<0.71 weightsq chan<(i8-0.00)*chan<chan outq -0.91<(i8-0.00)*0.00710241<0.90 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S20_Conv2d_32x16x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S17_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_f875d3fa", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_1bias", 0),
            GNodeArg(GNA_OUT, "S20_Output", 0),
            GNodeArg(GNA_IN, "S20_Mul_scale", 0),
            GNodeArg(GNA_IN, "S20_Mul_shift", 0),
            GNodeArg(GNA_IN, "S20_Infos", 0)
        )
    );
    // Node S23_Conv2d_32x1x3x3 inq -0.91<(i8-0.00)*0.00710241<0.90 forced weightsq chan<(i8-0.00)*chan<chan outq -0.81<(i8-40.00)*0.00485005<0.42 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S23_Conv2d_32x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S20_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_6a286dd7", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2", 0),
            GNodeArg(GNA_OUT, "S23_Output", 0),
            GNodeArg(GNA_IN, "S23_Mul_scale", 0),
            GNodeArg(GNA_IN, "S23_Mul_shift", 0),
            GNodeArg(GNA_IN, "S23_Infos", 0)
        )
    );
    // Node S26_Conv2d_16x32x1x1_Relu inq -0.81<(i8-40.00)*0.00485005<0.42 weightsq chan<(i8-0.00)*chan<chan outq -0.94<(i8-0.00)*0.00737075<0.94 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S26_Conv2d_16x32x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S23_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_1d2f6f73", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_2bias", 0),
            GNodeArg(GNA_OUT, "S26_Output", 0),
            GNodeArg(GNA_IN, "S26_Mul_scale", 0),
            GNodeArg(GNA_IN, "S26_Mul_shift", 0),
            GNodeArg(GNA_IN, "S26_Infos", 0)
        )
    );
    // Node S119_Conv2d_16x1x3x3 inq -0.94<(i8-0.00)*0.00737075<0.94 forced weightsq chan<(i8-0.00)*chan<chan outq -0.83<(i8--9.00)*0.00700378<0.95 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S119_Conv2d_16x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S26_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_3", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_10conv2d_1", 0),
            GNodeArg(GNA_OUT, "S119_Output", 0),
            GNodeArg(GNA_IN, "S119_Mul_scale", 0),
            GNodeArg(GNA_IN, "S119_Mul_shift", 0),
            GNodeArg(GNA_IN, "S119_Infos", 0)
        )
    );
    // Node S120_Conv2d_16x16x1x1_Relu inq -0.83<(i8--9.00)*0.00700378<0.95 weightsq chan<(i8-0.00)*chan<chan outq -0.94<(i8-0.00)*0.00737075<0.94 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S120_Conv2d_16x16x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S119_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_3_34a7c8eb", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_3bias", 0),
            GNodeArg(GNA_OUT, "S120_Output", 0),
            GNodeArg(GNA_IN, "S120_Mul_scale", 0),
            GNodeArg(GNA_IN, "S120_Mul_shift", 0),
            GNodeArg(GNA_IN, "S120_Infos", 0)
        )
    );
    // Node CONV_2D_0_12_fusion_qin0 inq -0.94<(i8-0.00)*0.00737075<0.94 forced outq 0.00<(i8--128.00)*0.00367092<0.94
    AddNode("S122_Op_CONV_2D_0_12_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S121_Output", 0),
            GNodeArg(GNA_OUT, "S122_Output", 0),
            GNodeArg(GNA_IN, "S122_Infos", 0)
        )
    );
    // Node S123_Conv2d_32x64x1x1_Relu inq 0.00<(i8--128.00)*0.00367092<0.94 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00230397<0.59 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S123_Conv2d_32x64x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S122_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_1conv2d", 0),
            GNodeArg(GNA_IN, "Conv2d_1bias", 0),
            GNodeArg(GNA_OUT, "S123_Output", 0),
            GNodeArg(GNA_IN, "S123_Mul_scale", 0),
            GNodeArg(GNA_IN, "S123_Mul_shift", 0),
            GNodeArg(GNA_IN, "S123_Infos", 0)
        )
    );
    // Node CONCATENATION_0_19_qin0 inq 0.00<(i8--128.00)*0.00230397<0.59 outq -0.69<(i8-0.00)*0.00542670<0.69 forced
    AddNode("S124_Op_CONCATENATION_0_19_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S123_Output", 0),
            GNodeArg(GNA_OUT, "S124_Output", 0),
            GNodeArg(GNA_IN, "S124_Infos", 0)
        )
    );
    // Node DEPTHWISE_CONV_2D_0_13_qin0 inq 0.00<(i8--128.00)*0.00230397<0.59 outq -0.59<(i8-0.00)*0.00462608<0.59 forced
    AddNode("S125_Op_DEPTHWISE_CONV_2D_0_13_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S123_Output", 0),
            GNodeArg(GNA_OUT, "S125_Output", 0),
            GNodeArg(GNA_IN, "S125_Infos", 0)
        )
    );
    // Node S129_Conv2d_32x1x3x3 inq -0.59<(i8-0.00)*0.00462608<0.59 forced weightsq chan<(i8-0.00)*chan<chan outq -0.39<(i8--22.00)*0.00369950<0.55 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S129_Conv2d_32x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S125_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_4", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_1", 0),
            GNodeArg(GNA_OUT, "S129_Output", 0),
            GNodeArg(GNA_IN, "S129_Mul_scale", 0),
            GNodeArg(GNA_IN, "S129_Mul_shift", 0),
            GNodeArg(GNA_IN, "S129_Infos", 0)
        )
    );
    // Node S130_Conv2d_16x32x1x1_Relu inq -0.39<(i8--22.00)*0.00369950<0.55 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00239889<0.61 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S130_Conv2d_16x32x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S129_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_4_6faf5601", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_4bias", 0),
            GNodeArg(GNA_OUT, "S130_Output", 0),
            GNodeArg(GNA_IN, "S130_Mul_scale", 0),
            GNodeArg(GNA_IN, "S130_Mul_shift", 0),
            GNodeArg(GNA_IN, "S130_Infos", 0)
        )
    );
    // Node CONCATENATION_0_19_qin1 inq 0.00<(i8--128.00)*0.00239889<0.61 outq -0.69<(i8-0.00)*0.00542670<0.69 forced
    AddNode("S131_Op_CONCATENATION_0_19_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S130_Output", 0),
            GNodeArg(GNA_OUT, "S131_Output", 0),
            GNodeArg(GNA_IN, "S131_Infos", 0)
        )
    );
    // Node DEPTHWISE_CONV_2D_0_15_qin0 inq 0.00<(i8--128.00)*0.00239889<0.61 outq -0.62<(i8-0.00)*0.00481666<0.61 forced
    AddNode("S132_Op_DEPTHWISE_CONV_2D_0_15_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S130_Output", 0),
            GNodeArg(GNA_OUT, "S132_Output", 0),
            GNodeArg(GNA_IN, "S132_Infos", 0)
        )
    );
    // Node S133_Conv2d_16x1x3x3 inq -0.62<(i8-0.00)*0.00481666<0.61 forced weightsq chan<(i8-0.00)*chan<chan outq -0.58<(i8-22.00)*0.00383511<0.40 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S133_Conv2d_16x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S132_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_5", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_10conv2d_2", 0),
            GNodeArg(GNA_OUT, "S133_Output", 0),
            GNodeArg(GNA_IN, "S133_Mul_scale", 0),
            GNodeArg(GNA_IN, "S133_Mul_shift", 0),
            GNodeArg(GNA_IN, "S133_Infos", 0)
        )
    );
    // Node S134_Conv2d_8x16x1x1_Relu inq -0.58<(i8-22.00)*0.00383511<0.40 weightsq chan<(i8-0.00)*chan<chan outq -0.69<(i8-0.00)*0.00542670<0.69 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S134_Conv2d_8x16x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S133_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_5_9edcbfcd", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_5bias", 0),
            GNodeArg(GNA_OUT, "S134_Output", 0),
            GNodeArg(GNA_IN, "S134_Mul_scale", 0),
            GNodeArg(GNA_IN, "S134_Mul_shift", 0),
            GNodeArg(GNA_IN, "S134_Infos", 0)
        )
    );
    // Node S135_Conv2d_8x1x3x3 inq -0.69<(i8-0.00)*0.00542670<0.69 forced weightsq chan<(i8-0.00)*chan<chan outq -0.73<(i8-103.00)*0.00315181<0.08 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S135_Conv2d_8x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S134_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_6_08bb4813", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_6", 0),
            GNodeArg(GNA_OUT, "S135_Output", 0),
            GNodeArg(GNA_IN, "S135_Mul_scale", 0),
            GNodeArg(GNA_IN, "S135_Mul_shift", 0),
            GNodeArg(GNA_IN, "S135_Infos", 0)
        )
    );
    // Node S136_Conv2d_8x8x1x1_Relu inq -0.73<(i8-103.00)*0.00315181<0.08 weightsq chan<(i8-0.00)*chan<chan outq -0.69<(i8-0.00)*0.00542670<0.69 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S136_Conv2d_8x8x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S135_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_6_0909dd70", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_6bias", 0),
            GNodeArg(GNA_OUT, "S136_Output", 0),
            GNodeArg(GNA_IN, "S136_Mul_scale", 0),
            GNodeArg(GNA_IN, "S136_Mul_shift", 0),
            GNodeArg(GNA_IN, "S136_Infos", 0)
        )
    );
    // Node AVERAGE_POOL_2D_0_20 inq -0.69<(i8-0.00)*0.00542670<0.69 forced outq -0.69<(i8-0.00)*0.00542670<0.69 forced
    AddNode("S138_AveragePool_3x3",
        Bindings(3,
            GNodeArg(GNA_IN, "S137_Output", 0),
            GNodeArg(GNA_OUT, "S138_Output", 0),
            GNodeArg(GNA_IN, "S138_Infos", 0)
        )
    );
    // Node S139_Conv2d_64x64x1x1_Relu inq -0.69<(i8-0.00)*0.00542670<0.69 forced weightsq chan<(i8-0.00)*chan<chan outq -1.13<(i8-0.00)*0.00881390<1.12 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S139_Conv2d_64x64x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S138_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_2conv2d", 0),
            GNodeArg(GNA_IN, "Conv2d_2bias", 0),
            GNodeArg(GNA_OUT, "S139_Output", 0),
            GNodeArg(GNA_IN, "S139_Mul_scale", 0),
            GNodeArg(GNA_IN, "S139_Mul_shift", 0),
            GNodeArg(GNA_IN, "S139_Infos", 0)
        )
    );
    // Node S140_Conv2d_64x1x3x3 inq -0.69<(i8-0.00)*0.00542670<0.69 forced weightsq chan<(i8-0.00)*chan<chan outq -0.48<(i8--23.00)*0.00452755<0.68 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S140_Conv2d_64x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S137_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_8", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_0306fd97", 0),
            GNodeArg(GNA_OUT, "S140_Output", 0),
            GNodeArg(GNA_IN, "S140_Mul_scale", 0),
            GNodeArg(GNA_IN, "S140_Mul_shift", 0),
            GNodeArg(GNA_IN, "S140_Infos", 0)
        )
    );
    // Node S141_Conv2d_64x64x1x1_Relu inq -0.48<(i8--23.00)*0.00452755<0.68 weightsq chan<(i8-0.00)*chan<chan outq -1.07<(i8-0.00)*0.00836546<1.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S141_Conv2d_64x64x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S140_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_8_05171b1c", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_8bias", 0),
            GNodeArg(GNA_OUT, "S141_Output", 0),
            GNodeArg(GNA_IN, "S141_Mul_scale", 0),
            GNodeArg(GNA_IN, "S141_Mul_shift", 0),
            GNodeArg(GNA_IN, "S141_Infos", 0)
        )
    );
    // Node CONCATENATION_0_88_qin1 inq -0.69<(i8-0.00)*0.00542670<0.69 forced outq 0.00<(i8--128.00)*0.00740498<1.89
    AddNode("S142_Op_CONCATENATION_0_88_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S137_Output", 0),
            GNodeArg(GNA_OUT, "S142_Output", 0),
            GNodeArg(GNA_IN, "S142_Infos", 0)
        )
    );
    // Node S149_Conv2d_64x1x3x3 inq -1.07<(i8-0.00)*0.00836546<1.06 forced weightsq chan<(i8-0.00)*chan<chan outq -0.70<(i8--2.00)*0.00554092<0.71 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S149_Conv2d_64x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S141_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_9", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_0306fd97_1", 0),
            GNodeArg(GNA_OUT, "S149_Output", 0),
            GNodeArg(GNA_IN, "S149_Mul_scale", 0),
            GNodeArg(GNA_IN, "S149_Mul_shift", 0),
            GNodeArg(GNA_IN, "S149_Infos", 0)
        )
    );
    // Node S150_Conv2d_32x64x1x1_Relu inq -0.70<(i8--2.00)*0.00554092<0.71 weightsq chan<(i8-0.00)*chan<chan outq -1.13<(i8-0.00)*0.00881390<1.12 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S150_Conv2d_32x64x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S149_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_9_b185726b", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_9bias", 0),
            GNodeArg(GNA_OUT, "S150_Output", 0),
            GNodeArg(GNA_IN, "S150_Mul_scale", 0),
            GNodeArg(GNA_IN, "S150_Mul_shift", 0),
            GNodeArg(GNA_IN, "S150_Infos", 0)
        )
    );
    // Node S151_Conv2d_32x1x3x3 inq -1.13<(i8-0.00)*0.00881390<1.12 forced weightsq chan<(i8-0.00)*chan<chan outq -0.63<(i8-2.00)*0.00480955<0.60 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S151_Conv2d_32x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S150_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_34edfbff", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_2", 0),
            GNodeArg(GNA_OUT, "S151_Output", 0),
            GNodeArg(GNA_IN, "S151_Mul_scale", 0),
            GNodeArg(GNA_IN, "S151_Mul_shift", 0),
            GNodeArg(GNA_IN, "S151_Infos", 0)
        )
    );
    // Node S152_Conv2d_32x32x1x1_Relu inq -0.63<(i8-2.00)*0.00480955<0.60 weightsq chan<(i8-0.00)*chan<chan outq -1.13<(i8-0.00)*0.00881390<1.12 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S152_Conv2d_32x32x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S151_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_c2442fd6", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_10bias", 0),
            GNodeArg(GNA_OUT, "S152_Output", 0),
            GNodeArg(GNA_IN, "S152_Mul_scale", 0),
            GNodeArg(GNA_IN, "S152_Mul_shift", 0),
            GNodeArg(GNA_IN, "S152_Infos", 0)
        )
    );
    // Node S154_Conv2d_64x128x1x1_Relu inq -1.13<(i8-0.00)*0.00881390<1.12 forced weightsq chan<(i8-0.00)*chan<chan outq -2.14<(i8-0.00)*0.01669604<2.12 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S154_Conv2d_64x128x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S153_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_3conv2d", 0),
            GNodeArg(GNA_IN, "Conv2d_3bias", 0),
            GNodeArg(GNA_OUT, "S154_Output", 0),
            GNodeArg(GNA_IN, "S154_Mul_scale", 0),
            GNodeArg(GNA_IN, "S154_Mul_shift", 0),
            GNodeArg(GNA_IN, "S154_Infos", 0)
        )
    );
    // Node S156_Conv2d_64x1x3x3 inq -2.14<(i8-0.00)*0.01669604<2.12 forced weightsq chan<(i8-0.00)*chan<chan outq -1.13<(i8--13.00)*0.00981163<1.37 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S156_Conv2d_64x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S154_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_749f00a1", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_0306fd97_2", 0),
            GNodeArg(GNA_OUT, "S156_Output", 0),
            GNodeArg(GNA_IN, "S156_Mul_scale", 0),
            GNodeArg(GNA_IN, "S156_Mul_shift", 0),
            GNodeArg(GNA_IN, "S156_Infos", 0)
        )
    );
    // Node S157_Conv2d_32x64x1x1_Relu inq -1.13<(i8--13.00)*0.00981163<1.37 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00653194<1.67 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S157_Conv2d_32x64x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S156_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_dfd1a27d", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_11bias", 0),
            GNodeArg(GNA_OUT, "S157_Output", 0),
            GNodeArg(GNA_IN, "S157_Mul_scale", 0),
            GNodeArg(GNA_IN, "S157_Mul_shift", 0),
            GNodeArg(GNA_IN, "S157_Infos", 0)
        )
    );
    // Node CONCATENATION_0_36_qin1 inq 0.00<(i8--128.00)*0.00653194<1.67 outq -2.14<(i8-0.00)*0.01669604<2.12 forced
    AddNode("S158_Op_CONCATENATION_0_36_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S157_Output", 0),
            GNodeArg(GNA_OUT, "S158_Output", 0),
            GNodeArg(GNA_IN, "S158_Infos", 0)
        )
    );
    // Node DEPTHWISE_CONV_2D_0_32_qin0 inq 0.00<(i8--128.00)*0.00653194<1.67 outq -1.68<(i8-0.00)*0.01311532<1.67 forced
    AddNode("S159_Op_DEPTHWISE_CONV_2D_0_32_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S157_Output", 0),
            GNodeArg(GNA_OUT, "S159_Output", 0),
            GNodeArg(GNA_IN, "S159_Infos", 0)
        )
    );
    // Node S160_Conv2d_32x1x3x3 inq -1.68<(i8-0.00)*0.01311532<1.67 forced weightsq chan<(i8-0.00)*chan<chan outq -0.42<(i8--71.00)*0.00740684<1.47 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S160_Conv2d_32x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S159_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_303b3c86", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_3", 0),
            GNodeArg(GNA_OUT, "S160_Output", 0),
            GNodeArg(GNA_IN, "S160_Mul_scale", 0),
            GNodeArg(GNA_IN, "S160_Mul_shift", 0),
            GNodeArg(GNA_IN, "S160_Infos", 0)
        )
    );
    // Node S161_Conv2d_16x32x1x1_Relu inq -0.42<(i8--71.00)*0.00740684<1.47 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00367384<0.94 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S161_Conv2d_16x32x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S160_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_4ee40c40", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_12bias", 0),
            GNodeArg(GNA_OUT, "S161_Output", 0),
            GNodeArg(GNA_IN, "S161_Mul_scale", 0),
            GNodeArg(GNA_IN, "S161_Mul_shift", 0),
            GNodeArg(GNA_IN, "S161_Infos", 0)
        )
    );
    // Node CONCATENATION_0_36_qin2 inq 0.00<(i8--128.00)*0.00367384<0.94 outq -2.14<(i8-0.00)*0.01669604<2.12 forced
    AddNode("S162_Op_CONCATENATION_0_36_qin2",
        Bindings(3,
            GNodeArg(GNA_IN, "S161_Output", 0),
            GNodeArg(GNA_OUT, "S162_Output", 0),
            GNodeArg(GNA_IN, "S162_Infos", 0)
        )
    );
    // Node DEPTHWISE_CONV_2D_0_34_qin0 inq 0.00<(i8--128.00)*0.00367384<0.94 outq -0.94<(i8-0.00)*0.00737660<0.94 forced
    AddNode("S163_Op_DEPTHWISE_CONV_2D_0_34_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S161_Output", 0),
            GNodeArg(GNA_OUT, "S163_Output", 0),
            GNodeArg(GNA_IN, "S163_Infos", 0)
        )
    );
    // Node S164_Conv2d_16x1x3x3 inq -0.94<(i8-0.00)*0.00737660<0.94 forced weightsq chan<(i8-0.00)*chan<chan outq -1.24<(i8-54.00)*0.00682904<0.50 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S164_Conv2d_16x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S163_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_61ce1792", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_10conv2d_3", 0),
            GNodeArg(GNA_OUT, "S164_Output", 0),
            GNodeArg(GNA_IN, "S164_Mul_scale", 0),
            GNodeArg(GNA_IN, "S164_Mul_shift", 0),
            GNodeArg(GNA_IN, "S164_Infos", 0)
        )
    );
    // Node S165_Conv2d_16x16x1x1_Relu inq -1.24<(i8-54.00)*0.00682904<0.50 weightsq chan<(i8-0.00)*chan<chan outq -2.14<(i8-0.00)*0.01669604<2.12 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S165_Conv2d_16x16x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S164_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_a85ca358", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_13bias", 0),
            GNodeArg(GNA_OUT, "S165_Output", 0),
            GNodeArg(GNA_IN, "S165_Mul_scale", 0),
            GNodeArg(GNA_IN, "S165_Mul_shift", 0),
            GNodeArg(GNA_IN, "S165_Infos", 0)
        )
    );
    // Node AVERAGE_POOL_2D_0_37 inq -2.14<(i8-0.00)*0.01669604<2.12 forced outq -2.14<(i8-0.00)*0.01669604<2.12 forced
    AddNode("S167_AveragePool_3x3",
        Bindings(3,
            GNodeArg(GNA_IN, "S166_Output", 0),
            GNodeArg(GNA_OUT, "S167_Output", 0),
            GNodeArg(GNA_IN, "S167_Infos", 0)
        )
    );
    // Node S168_Conv2d_128x128x1x1_Relu inq -2.14<(i8-0.00)*0.01669604<2.12 forced weightsq chan<(i8-0.00)*chan<chan outq -2.12<(i8-0.00)*0.01657579<2.11 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S168_Conv2d_128x128x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S167_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_4conv2d", 0),
            GNodeArg(GNA_IN, "Conv2d_4bias", 0),
            GNodeArg(GNA_OUT, "S168_Output", 0),
            GNodeArg(GNA_IN, "S168_Mul_scale", 0),
            GNodeArg(GNA_IN, "S168_Mul_shift", 0),
            GNodeArg(GNA_IN, "S168_Infos", 0)
        )
    );
    // Node S169_Conv2d_128x1x3x3 inq -2.14<(i8-0.00)*0.01669604<2.12 forced weightsq chan<(i8-0.00)*chan<chan outq -1.32<(i8--16.00)*0.01181986<1.69 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S169_Conv2d_128x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S166_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_f63cde89", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_70a03dc2", 0),
            GNodeArg(GNA_OUT, "S169_Output", 0),
            GNodeArg(GNA_IN, "S169_Mul_scale", 0),
            GNodeArg(GNA_IN, "S169_Mul_shift", 0),
            GNodeArg(GNA_IN, "S169_Infos", 0)
        )
    );
    // Node S170_Conv2d_128x128x1x1_Relu inq -1.32<(i8--16.00)*0.01181986<1.69 weightsq chan<(i8-0.00)*chan<chan outq -2.25<(i8-0.00)*0.01760845<2.24 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S170_Conv2d_128x128x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S169_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_72c6547f", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_14bias", 0),
            GNodeArg(GNA_OUT, "S170_Output", 0),
            GNodeArg(GNA_IN, "S170_Mul_scale", 0),
            GNodeArg(GNA_IN, "S170_Mul_shift", 0),
            GNodeArg(GNA_IN, "S170_Infos", 0)
        )
    );
    // Node S175_Conv2d_128x1x3x3 inq -2.25<(i8-0.00)*0.01760845<2.24 forced weightsq chan<(i8-0.00)*chan<chan outq -3.12<(i8-38.00)*0.01878701<1.67 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S175_Conv2d_128x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S170_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_1b9fd017", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_70a03dc2_1", 0),
            GNodeArg(GNA_OUT, "S175_Output", 0),
            GNodeArg(GNA_IN, "S175_Mul_scale", 0),
            GNodeArg(GNA_IN, "S175_Mul_shift", 0),
            GNodeArg(GNA_IN, "S175_Infos", 0)
        )
    );
    // Node S176_Conv2d_64x128x1x1_Relu inq -3.12<(i8-38.00)*0.01878701<1.67 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.01338032<3.41 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S176_Conv2d_64x128x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S175_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_d01ce067", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_15bias", 0),
            GNodeArg(GNA_OUT, "S176_Output", 0),
            GNodeArg(GNA_IN, "S176_Mul_scale", 0),
            GNodeArg(GNA_IN, "S176_Mul_shift", 0),
            GNodeArg(GNA_IN, "S176_Infos", 0)
        )
    );
    // Node CONCATENATION_0_45_qin1 inq 0.00<(i8--128.00)*0.01338032<3.41 outq -2.12<(i8-0.00)*0.01657579<2.11
    AddNode("S177_Op_CONCATENATION_0_45_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S176_Output", 0),
            GNodeArg(GNA_OUT, "S177_Output", 0),
            GNodeArg(GNA_IN, "S177_Infos", 0)
        )
    );
    // Node DEPTHWISE_CONV_2D_0_43_qin0 inq 0.00<(i8--128.00)*0.01338032<3.41 outq -3.44<(i8-0.00)*0.02686599<3.41 forced
    AddNode("S178_Op_DEPTHWISE_CONV_2D_0_43_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S176_Output", 0),
            GNodeArg(GNA_OUT, "S178_Output", 0),
            GNodeArg(GNA_IN, "S178_Infos", 0)
        )
    );
    // Node S179_Conv2d_64x1x3x3 inq -3.44<(i8-0.00)*0.02686599<3.41 forced weightsq chan<(i8-0.00)*chan<chan outq -2.78<(i8-17.00)*0.01915484<2.11 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S179_Conv2d_64x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S178_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_7e9971c1", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_0306fd97_3", 0),
            GNodeArg(GNA_OUT, "S179_Output", 0),
            GNodeArg(GNA_IN, "S179_Mul_scale", 0),
            GNodeArg(GNA_IN, "S179_Mul_shift", 0),
            GNodeArg(GNA_IN, "S179_Infos", 0)
        )
    );
    // Node S180_Conv2d_64x64x1x1_Relu inq -2.78<(i8-17.00)*0.01915484<2.11 weightsq chan<(i8-0.00)*chan<chan outq -2.12<(i8-0.00)*0.01657579<2.11 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S180_Conv2d_64x64x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S179_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_75db895d", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_16bias", 0),
            GNodeArg(GNA_OUT, "S180_Output", 0),
            GNodeArg(GNA_IN, "S180_Mul_scale", 0),
            GNodeArg(GNA_IN, "S180_Mul_shift", 0),
            GNodeArg(GNA_IN, "S180_Infos", 0)
        )
    );
    // Node S182_Conv2d_128x256x1x1_Relu inq -2.12<(i8-0.00)*0.01657579<2.11 weightsq chan<(i8-0.00)*chan<chan outq -5.88<(i8-0.00)*0.04591471<5.83 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S182_Conv2d_128x256x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S181_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_5conv2d", 0),
            GNodeArg(GNA_IN, "Conv2d_5bias", 0),
            GNodeArg(GNA_OUT, "S182_Output", 0),
            GNodeArg(GNA_IN, "S182_Mul_scale", 0),
            GNodeArg(GNA_IN, "S182_Mul_shift", 0),
            GNodeArg(GNA_IN, "S182_Infos", 0)
        )
    );
    // Node S184_Conv2d_128x1x3x3 inq -5.88<(i8-0.00)*0.04591471<5.83 forced weightsq chan<(i8-0.00)*chan<chan outq -4.46<(i8-12.00)*0.03185714<3.66 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S184_Conv2d_128x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S182_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_90fdc921", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_70a03dc2_2", 0),
            GNodeArg(GNA_OUT, "S184_Output", 0),
            GNodeArg(GNA_IN, "S184_Mul_scale", 0),
            GNodeArg(GNA_IN, "S184_Mul_shift", 0),
            GNodeArg(GNA_IN, "S184_Infos", 0)
        )
    );
    // Node S185_Conv2d_64x128x1x1_Relu inq -4.46<(i8-12.00)*0.03185714<3.66 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02106018<5.37 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S185_Conv2d_64x128x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S184_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_70b8f72e", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_17bias", 0),
            GNodeArg(GNA_OUT, "S185_Output", 0),
            GNodeArg(GNA_IN, "S185_Mul_scale", 0),
            GNodeArg(GNA_IN, "S185_Mul_shift", 0),
            GNodeArg(GNA_IN, "S185_Infos", 0)
        )
    );
    // Node CONCATENATION_0_53_qin1 inq 0.00<(i8--128.00)*0.02106018<5.37 outq -5.88<(i8-0.00)*0.04591471<5.83 forced
    AddNode("S186_Op_CONCATENATION_0_53_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S185_Output", 0),
            GNodeArg(GNA_OUT, "S186_Output", 0),
            GNodeArg(GNA_IN, "S186_Infos", 0)
        )
    );
    // Node DEPTHWISE_CONV_2D_0_49_qin0 inq 0.00<(i8--128.00)*0.02106018<5.37 outq -5.41<(i8-0.00)*0.04228619<5.37 forced
    AddNode("S187_Op_DEPTHWISE_CONV_2D_0_49_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S185_Output", 0),
            GNodeArg(GNA_OUT, "S187_Output", 0),
            GNodeArg(GNA_IN, "S187_Infos", 0)
        )
    );
    // Node S188_Conv2d_64x1x3x3 inq -5.41<(i8-0.00)*0.04228619<5.37 forced weightsq chan<(i8-0.00)*chan<chan outq -2.49<(i8-17.00)*0.01720178<1.89 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S188_Conv2d_64x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S187_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_a7079b99", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_0306fd97_4", 0),
            GNodeArg(GNA_OUT, "S188_Output", 0),
            GNodeArg(GNA_IN, "S188_Mul_scale", 0),
            GNodeArg(GNA_IN, "S188_Mul_shift", 0),
            GNodeArg(GNA_IN, "S188_Infos", 0)
        )
    );
    // Node S189_Conv2d_32x64x1x1_Relu inq -2.49<(i8-17.00)*0.01720178<1.89 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00910447<2.32 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S189_Conv2d_32x64x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S188_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_f2ce61c9", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_18bias", 0),
            GNodeArg(GNA_OUT, "S189_Output", 0),
            GNodeArg(GNA_IN, "S189_Mul_scale", 0),
            GNodeArg(GNA_IN, "S189_Mul_shift", 0),
            GNodeArg(GNA_IN, "S189_Infos", 0)
        )
    );
    // Node CONCATENATION_0_53_qin2 inq 0.00<(i8--128.00)*0.00910447<2.32 outq -5.88<(i8-0.00)*0.04591471<5.83 forced
    AddNode("S190_Op_CONCATENATION_0_53_qin2",
        Bindings(3,
            GNodeArg(GNA_IN, "S189_Output", 0),
            GNodeArg(GNA_OUT, "S190_Output", 0),
            GNodeArg(GNA_IN, "S190_Infos", 0)
        )
    );
    // Node DEPTHWISE_CONV_2D_0_51_qin0 inq 0.00<(i8--128.00)*0.00910447<2.32 outq -2.34<(i8-0.00)*0.01828063<2.32 forced
    AddNode("S191_Op_DEPTHWISE_CONV_2D_0_51_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S189_Output", 0),
            GNodeArg(GNA_OUT, "S191_Output", 0),
            GNodeArg(GNA_IN, "S191_Infos", 0)
        )
    );
    // Node S192_Conv2d_32x1x3x3 inq -2.34<(i8-0.00)*0.01828063<2.32 forced weightsq chan<(i8-0.00)*chan<chan outq -1.59<(i8--10.00)*0.01345786<1.84 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S192_Conv2d_32x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S191_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_8ec95cbf", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_4", 0),
            GNodeArg(GNA_OUT, "S192_Output", 0),
            GNodeArg(GNA_IN, "S192_Mul_scale", 0),
            GNodeArg(GNA_IN, "S192_Mul_shift", 0),
            GNodeArg(GNA_IN, "S192_Infos", 0)
        )
    );
    // Node S193_Conv2d_32x32x1x1_Relu inq -1.59<(i8--10.00)*0.01345786<1.84 weightsq chan<(i8-0.00)*chan<chan outq -5.88<(i8-0.00)*0.04591471<5.83 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S193_Conv2d_32x32x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S192_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_1_dac090c6", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_19bias", 0),
            GNodeArg(GNA_OUT, "S193_Output", 0),
            GNodeArg(GNA_IN, "S193_Mul_scale", 0),
            GNodeArg(GNA_IN, "S193_Mul_shift", 0),
            GNodeArg(GNA_IN, "S193_Infos", 0)
        )
    );
    // Node MEAN_0_54 inq -5.88<(i8-0.00)*0.04591471<5.83 forced outq -4.69<(i8-0.00)*0.03666924<4.66
    AddNode("S195_Op_MEAN_0_54",
        Bindings(3,
            GNodeArg(GNA_IN, "S194_Output", 0),
            GNodeArg(GNA_OUT, "S195_Output", 0),
            GNodeArg(GNA_IN, "S195_Infos", 0)
        )
    );
    // Node S197_Conv2d_32x256x1x1_Relu inq -4.69<(i8-0.00)*0.03666924<4.66 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.04125773<10.52 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S197_Conv2d_32x256x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S195_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_6conv2d", 0),
            GNodeArg(GNA_IN, "Conv2d_6bias", 0),
            GNodeArg(GNA_OUT, "S197_Output", 0),
            GNodeArg(GNA_IN, "S197_Mul_scale", 0),
            GNodeArg(GNA_IN, "S197_Mul_shift", 0),
            GNodeArg(GNA_IN, "S197_Infos", 0)
        )
    );
    // Node S198_Conv2d_256x1x3x3 inq -5.88<(i8-0.00)*0.04591471<5.83 forced weightsq chan<(i8-0.00)*chan<chan outq -4.67<(i8-7.00)*0.03460640<4.15 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S198_Conv2d_256x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S194_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_90e98147", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_2ea7e0c0", 0),
            GNodeArg(GNA_OUT, "S198_Output", 0),
            GNodeArg(GNA_IN, "S198_Mul_scale", 0),
            GNodeArg(GNA_IN, "S198_Mul_shift", 0),
            GNodeArg(GNA_IN, "S198_Infos", 0)
        )
    );
    // Node S199_Conv2d_32x256x1x1_Relu inq -4.67<(i8-7.00)*0.03460640<4.15 weightsq chan<(i8-0.00)*chan<chan outq -5.68<(i8-0.00)*0.04434614<5.63 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S199_Conv2d_32x256x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S198_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_294b40b0", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_20bias", 0),
            GNodeArg(GNA_OUT, "S199_Output", 0),
            GNodeArg(GNA_IN, "S199_Mul_scale", 0),
            GNodeArg(GNA_IN, "S199_Mul_shift", 0),
            GNodeArg(GNA_IN, "S199_Infos", 0)
        )
    );
    // Node MEAN_0_62 inq -5.68<(i8-0.00)*0.04434614<5.63 forced outq -3.71<(i8-0.00)*0.02896121<3.68
    AddNode("S200_Op_MEAN_0_62",
        Bindings(3,
            GNodeArg(GNA_IN, "S199_Output", 0),
            GNodeArg(GNA_OUT, "S200_Output", 0),
            GNodeArg(GNA_IN, "S200_Infos", 0)
        )
    );
    // Node S202_Conv2d_32x32x1x1_Sigmoid inq -3.71<(i8-0.00)*0.02896121<3.68 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S202_Conv2d_32x32x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S200_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_7conv2d", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_7biasaddmod", 0),
            GNodeArg(GNA_OUT, "S202_Output", 0),
            GNodeArg(GNA_IN, "S202_Mul_scale", 0),
            GNodeArg(GNA_IN, "S202_Mul_shift", 0),
            GNodeArg(GNA_IN, "S202_Infos", 0)
        )
    );
    // Node expr_2 in_qs [-5.68<(i8-0.00)*0.04434614<5.63 forced,-1.01<(i8-0.00)*0.00787402<1.00,0.00<(i8--128.00)*0.04125773<10.52] out_qs [-16.57<(i8-0.00)*0.12943257<16.44]
    AddNode("S203_Op_expr_2",
        Bindings(4,
            GNodeArg(GNA_IN, "S199_Output", 0),
            GNodeArg(GNA_IN, "S202_Output", 0),
            GNodeArg(GNA_IN, "S197_Output", 0),
            GNodeArg(GNA_OUT, "S203_Output", 0)
        )
    );
    // Node RESIZE_BILINEAR_0_71 inq -16.57<(i8-0.00)*0.12943257<16.44 outq -16.57<(i8-0.00)*0.12943257<16.44 forced
    AddNode("S204_Op_RESIZE_BILINEAR_0_71",
        Bindings(2,
            GNodeArg(GNA_IN, "S203_Output", 0),
            GNodeArg(GNA_OUT, "S204_Output", 0)
        )
    );
    // Node S205_Conv2d_32x1x3x3 inq -16.57<(i8-0.00)*0.12943257<16.44 forced weightsq chan<(i8-0.00)*chan<chan outq -8.63<(i8-11.00)*0.06209885<7.20 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S205_Conv2d_32x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S204_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_57f81003", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_5", 0),
            GNodeArg(GNA_OUT, "S205_Output", 0),
            GNodeArg(GNA_IN, "S205_Mul_scale", 0),
            GNodeArg(GNA_IN, "S205_Mul_shift", 0),
            GNodeArg(GNA_IN, "S205_Infos", 0)
        )
    );
    // Node S206_Conv2d_32x32x1x1_Relu inq -8.63<(i8-11.00)*0.06209885<7.20 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.01876499<4.79 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S206_Conv2d_32x32x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S205_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_d0fcba95", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_21bias", 0),
            GNodeArg(GNA_OUT, "S206_Output", 0),
            GNodeArg(GNA_IN, "S206_Mul_scale", 0),
            GNodeArg(GNA_IN, "S206_Mul_shift", 0),
            GNodeArg(GNA_IN, "S206_Infos", 0)
        )
    );
    // Node S208_Conv2d_128x1x3x3 inq -2.14<(i8-0.00)*0.01669604<2.12 forced weightsq chan<(i8-0.00)*chan<chan outq -1.53<(i8-2.00)*0.01179407<1.47 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S208_Conv2d_128x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S166_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_c1d436ee", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_70a03dc2_3", 0),
            GNodeArg(GNA_OUT, "S208_Output", 0),
            GNodeArg(GNA_IN, "S208_Mul_scale", 0),
            GNodeArg(GNA_IN, "S208_Mul_shift", 0),
            GNodeArg(GNA_IN, "S208_Infos", 0)
        )
    );
    // Node S209_Conv2d_32x128x1x1_Relu inq -1.53<(i8-2.00)*0.01179407<1.47 weightsq chan<(i8-0.00)*chan<chan outq -1.74<(i8-0.00)*0.01357732<1.72 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S209_Conv2d_32x128x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S208_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_5ce43f07", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_23bias", 0),
            GNodeArg(GNA_OUT, "S209_Output", 0),
            GNodeArg(GNA_IN, "S209_Mul_scale", 0),
            GNodeArg(GNA_IN, "S209_Mul_shift", 0),
            GNodeArg(GNA_IN, "S209_Infos", 0)
        )
    );
    // Node MEAN_0_76 inq -1.74<(i8-0.00)*0.01357732<1.72 forced outq -1.56<(i8-0.00)*0.01222099<1.55
    AddNode("S210_Op_MEAN_0_76",
        Bindings(3,
            GNodeArg(GNA_IN, "S209_Output", 0),
            GNodeArg(GNA_OUT, "S210_Output", 0),
            GNodeArg(GNA_IN, "S210_Infos", 0)
        )
    );
    // Node S212_Conv2d_32x32x1x1_Sigmoid inq -1.56<(i8-0.00)*0.01222099<1.55 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S212_Conv2d_32x32x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S210_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_8conv2d", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_8biasaddmod", 0),
            GNodeArg(GNA_OUT, "S212_Output", 0),
            GNodeArg(GNA_IN, "S212_Mul_scale", 0),
            GNodeArg(GNA_IN, "S212_Mul_shift", 0),
            GNodeArg(GNA_IN, "S212_Infos", 0)
        )
    );
    // Node expr_0 in_qs [-1.74<(i8-0.00)*0.01357732<1.72 forced,-1.01<(i8-0.00)*0.00787402<1.00,0.00<(i8--128.00)*0.01876499<4.79] out_qs [-4.82<(i8-0.00)*0.03767773<4.79]
    AddNode("S213_Op_expr_0",
        Bindings(4,
            GNodeArg(GNA_IN, "S209_Output", 0),
            GNodeArg(GNA_IN, "S212_Output", 0),
            GNodeArg(GNA_IN, "S206_Output", 0),
            GNodeArg(GNA_OUT, "S213_Output", 0)
        )
    );
    // Node RESIZE_BILINEAR_0_85 inq -4.82<(i8-0.00)*0.03767773<4.79 outq -4.82<(i8-0.00)*0.03767773<4.79 forced
    AddNode("S214_Op_RESIZE_BILINEAR_0_85",
        Bindings(2,
            GNodeArg(GNA_IN, "S213_Output", 0),
            GNodeArg(GNA_OUT, "S214_Output", 0)
        )
    );
    // Node S215_Conv2d_32x1x3x3 inq -4.82<(i8-0.00)*0.03767773<4.79 forced weightsq chan<(i8-0.00)*chan<chan outq -3.93<(i8-32.00)*0.02453824<2.33 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S215_Conv2d_32x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S214_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_41884a31", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_6", 0),
            GNodeArg(GNA_OUT, "S215_Output", 0),
            GNodeArg(GNA_IN, "S215_Mul_scale", 0),
            GNodeArg(GNA_IN, "S215_Mul_shift", 0),
            GNodeArg(GNA_IN, "S215_Infos", 0)
        )
    );
    // Node S216_Conv2d_32x32x1x1_Relu inq -3.93<(i8-32.00)*0.02453824<2.33 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00740498<1.89 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S216_Conv2d_32x32x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S215_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_8e412014", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_24bias", 0),
            GNodeArg(GNA_OUT, "S216_Output", 0),
            GNodeArg(GNA_IN, "S216_Mul_scale", 0),
            GNodeArg(GNA_IN, "S216_Mul_shift", 0),
            GNodeArg(GNA_IN, "S216_Infos", 0)
        )
    );
    // Node S218_Conv2d_64x96x1x1_Relu inq 0.00<(i8--128.00)*0.00740498<1.89 weightsq chan<(i8-0.00)*chan<chan outq -1.93<(i8-0.00)*0.01505481<1.91 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S218_Conv2d_64x96x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S217_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_9conv2d", 0),
            GNodeArg(GNA_IN, "Conv2d_9bias", 0),
            GNodeArg(GNA_OUT, "S218_Output", 0),
            GNodeArg(GNA_IN, "S218_Mul_scale", 0),
            GNodeArg(GNA_IN, "S218_Mul_shift", 0),
            GNodeArg(GNA_IN, "S218_Infos", 0)
        )
    );
    // Node MEAN_0_90 inq -1.93<(i8-0.00)*0.01505481<1.91 forced outq -1.36<(i8-0.00)*0.01059234<1.35
    AddNode("S219_Op_MEAN_0_90",
        Bindings(3,
            GNodeArg(GNA_IN, "S218_Output", 0),
            GNodeArg(GNA_OUT, "S219_Output", 0),
            GNodeArg(GNA_IN, "S219_Infos", 0)
        )
    );
    // Node S221_Conv2d_16x64x1x1_Relu inq -1.36<(i8-0.00)*0.01059234<1.35 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00222415<0.57 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S221_Conv2d_16x64x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S219_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_10conv2d1", 0),
            GNodeArg(GNA_IN, "Conv2d_10bias", 0),
            GNodeArg(GNA_OUT, "S221_Output", 0),
            GNodeArg(GNA_IN, "S221_Mul_scale", 0),
            GNodeArg(GNA_IN, "S221_Mul_shift", 0),
            GNodeArg(GNA_IN, "S221_Infos", 0)
        )
    );
    // Node S222_Conv2d_64x16x1x1_Sigmoid inq 0.00<(i8--128.00)*0.00222415<0.57 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S222_Conv2d_64x16x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S221_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_11conv2d", 0),
            GNodeArg(GNA_IN, "Model_2modelconv2d_11biasaddmo", 0),
            GNodeArg(GNA_OUT, "S222_Output", 0),
            GNodeArg(GNA_IN, "S222_Mul_scale", 0),
            GNodeArg(GNA_IN, "S222_Mul_shift", 0),
            GNodeArg(GNA_IN, "S222_Infos", 0)
        )
    );
    // Node expr_1 in_qs [-1.93<(i8-0.00)*0.01505481<1.91 forced,-1.01<(i8-0.00)*0.00787402<1.00] out_qs [-2.88<(i8-0.00)*0.02248147<2.86]
    AddNode("S223_Op_expr_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S218_Output", 0),
            GNodeArg(GNA_IN, "S222_Output", 0),
            GNodeArg(GNA_OUT, "S223_Output", 0)
        )
    );
    // Node S224_Conv2d_64x1x3x3 inq -2.88<(i8-0.00)*0.02248147<2.86 forced weightsq chan<(i8-0.00)*chan<chan outq -2.25<(i8-32.00)*0.01404283<1.33 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S224_Conv2d_64x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S223_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_53045d95", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_0306fd97_5", 0),
            GNodeArg(GNA_OUT, "S224_Output", 0),
            GNodeArg(GNA_IN, "S224_Mul_scale", 0),
            GNodeArg(GNA_IN, "S224_Mul_shift", 0),
            GNodeArg(GNA_IN, "S224_Infos", 0)
        )
    );
    // Node S225_Conv2d_64x64x1x1_Relu inq -2.25<(i8-32.00)*0.01404283<1.33 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00731766<1.87 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S225_Conv2d_64x64x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S224_Output", 0),
            GNodeArg(GNA_IN, "Model_2modelseparable_conv2d_2_f68e0cfc", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_26bias", 0),
            GNodeArg(GNA_OUT, "S225_Output", 0),
            GNodeArg(GNA_IN, "S225_Mul_scale", 0),
            GNodeArg(GNA_IN, "S225_Mul_shift", 0),
            GNodeArg(GNA_IN, "S225_Infos", 0)
        )
    );
    // Node S226_Conv2d_2x64x1x1 inq 0.00<(i8--128.00)*0.00731766<1.87 weightsq chan<(i8-0.00)*chan<chan outq -1.59<(i8-0.00)*0.01240211<1.58 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S226_Conv2d_2x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S225_Output", 0),
            GNodeArg(GNA_IN, "Model_2modeloutputconv2d1", 0),
            GNodeArg(GNA_IN, "Model_2modeloutputbiasaddmodel", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_IN, "S226_Mul_scale", 0),
            GNodeArg(GNA_IN, "S226_Mul_shift", 0),
            GNodeArg(GNA_IN, "S226_Infos", 0)
        )
    );
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    nanoflownet_unquantizedModel(64000, 300000, 8000000, 64*1024*1024);
    GenerateTilingCode();
    return 0;
}

/*
* Copyright (C) 2017 GreenWaves Technologies
* All rights reserved.
*
* This software may be modified and distributed under the terms
* of the BSD license.  See the LICENSE file for details.
*/

#include "stdio.h"

/* PMSIS includes */
#include "pmsis.h"

/* BSP includes */
#include "bsp/bsp.h"
#include "bsp/camera.h"

/* Demo includes */
#include "gaplib/ImgIO.h"

#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE 1024
#endif

/* Camera resolution definitions */
#if defined(QVGA_IMG)   /* QVGA */
#define YRES          224
#define XRES          162
#elif defined(QQVGA_IMG) /* QQVGA */
#define YRES          162
#define XRES          162
#else                   /* default */
#define YRES          324
#define XRES          324
#endif

/* Region-of-interest (slice) mode */
#if defined(SLICE_MODE)
#if defined(QVGA_IMG)
    #define ROI_WIDTH    162
    #define ROI_HEIGHT   112
#elif defined(QQVGA_IMG)
    #define ROI_WIDTH     64   /* QQVGA下縮至64×64 */
    #define ROI_HEIGHT    64
#else
    #define ROI_WIDTH    162
    #define ROI_HEIGHT   112
#endif
#define X             ((XRES - ROI_WIDTH)  / 2)
#define Y             ((YRES - ROI_HEIGHT) / 2)
#define CAMERA_WIDTH  ROI_WIDTH
#define CAMERA_HEIGHT ROI_HEIGHT
#endif

/* Global peripherals and buffers */
static struct pi_device      cam;        /* Camera device */
static uint8_t              *imgBuff0;   /* Frame buffer 0 */
static uint8_t              *imgBuff1;   /* Frame buffer 1 */
static struct pi_device      uart;       /* UART device */
static struct pi_uart_conf   uart_conf;
static struct pi_device      gpio_device;/* GPIO device for LED */
static int                   led_val = 0;/* LED state toggle */

static uint8_t              *Input_1;    /* 最新影像緩衝區 */
static uint8_t              *Input_2;    /* 前一幀影像緩衝區 */
static uint8_t              *drawImg = NULL; /* 用於繪製光流結果 */

static float *shared_flow_vectors_x;
static float *shared_flow_vectors_y;

static uint8_t               g_flow_error = 0;
static char                  imgName[50];
static uint32_t              idx = 0;

static float smoothed_error = 0.0f;   // 指數平滑狀態
float alpha = 0.3f;                   // 平滑係數

static int32_t raw_error; 
static int8_t final_error_u8;

/*──────────────────────────────────────────────────────────────────────────*/
/*                           EXPERIMENTAL PARAMETERS                       */
/*──────────────────────────────────────────────────────────────────────────*/
#define PYRAMID_LEVELS  2    /* 金字塔層數 */
#define GRID_SIZE_X     8   /* 光流網格橫向點數 */
#define GRID_SIZE_Y     8   /* 光流網格縱向點數 */
#define CLUSTER_CORES   8    /* Cluster 計算核心數 */

#define STR_HELPER(x)   #x
#define STR(x)          STR_HELPER(x)
 



/*──────────────────────────────────────────────────────────────────────────*/
/*                           open_camera_himax                              */
/*──────────────────────────────────────────────────────────────────────────*/
static int32_t open_camera_himax(struct pi_device *device){
    struct pi_himax_conf cam_conf;
    pi_himax_conf_init(&cam_conf);

#ifdef SLICE_MODE
    cam_conf.roi.slice_en = 1;
    cam_conf.roi.x = X;
    cam_conf.roi.y = Y;
    cam_conf.roi.w = CAMERA_WIDTH;
    cam_conf.roi.h = CAMERA_HEIGHT;
#endif

#ifdef QQVGA_IMG
    cam_conf.format = PI_CAMERA_QQVGA;
#endif

    pi_open_from_conf(device, &cam_conf);
    if (pi_camera_open(device)){
        return -1;
    }

    // 旋轉影像設定
    pi_camera_control(&cam, PI_CAMERA_CMD_START, 0);
    uint8_t set_value = 3;
    uint8_t reg_value;
    pi_camera_reg_set(&cam, IMG_ORIENTATION, &set_value);
    pi_time_wait_us(1000000);
    pi_camera_reg_get(&cam, IMG_ORIENTATION, &reg_value);
    if (set_value != reg_value){
        printf("Failed to rotate camera image\n");
        return -1;
    }
    pi_camera_control(&cam, PI_CAMERA_CMD_STOP, 0);

    /* 啟動自動曝光控制 */
    pi_camera_control(&cam, PI_CAMERA_CMD_AEG_INIT, 0);

    return 0;
}

/*──────────────────────────────────────────────────────────────────────────*/
/*                     Lukas-Kanade Pyramid 光流相關函數                      */
/*──────────────────────────────────────────────────────────────────────────*/
//=======================================================================================================================================================================================================================================
// (1) Parallel Resize

// 定義平行縮放參數結構
typedef struct {
    uint8_t* src;
    int src_width;
    int src_height;
    uint8_t* dst;
    int dst_width;
    int dst_height;
} resize_args_t;

// 每個核心計算自己負責的行範圍
void parallel_resize_worker(void *arg) {
    resize_args_t* args = (resize_args_t*) arg;
    int core_id = pi_core_id();
    int nb_cores = CLUSTER_CORES;
    int total_rows = args->dst_height;
    int rows_per_core = total_rows / nb_cores;
    int start = core_id * rows_per_core;
    int end = (core_id == nb_cores - 1) ? total_rows : start + rows_per_core;
    for (int j = start; j < end; j++){
        float src_y = j * (args->src_height - 1) / (float)(args->dst_height - 1);
        int y0 = (int)src_y;
        int y1 = y0 + 1;
        if (y1 >= args->src_height) y1 = args->src_height - 1;
        float dy = src_y - y0;
        for (int i = 0; i < args->dst_width; i++){
            float src_x = i * (args->src_width - 1) / (float)(args->dst_width - 1);
            int x0 = (int)src_x;
            int x1 = x0 + 1;
            if (x1 >= args->src_width) x1 = args->src_width - 1;
            float dx = src_x - x0;
            float top = (1 - dx) * args->src[y0 * args->src_width + x0] + dx * args->src[y0 * args->src_width + x1];
            float bottom = (1 - dx) * args->src[y1 * args->src_width + x0] + dx * args->src[y1 * args->src_width + x1];
            float value = (1 - dy) * top + dy * bottom;
            args->dst[j * args->dst_width + i] = (uint8_t)(value + 0.5f);
        }
    }
}

// 平行版本的縮放函式，利用集群所有核心共同完成
/* 自行實作的 bilinear 插值縮放函式 */
void resize_image_uint8_linear_parallel(uint8_t* src, int src_width, int src_height, 
                                          uint8_t* dst, int dst_width, int dst_height) {
    resize_args_t args;
    args.src = src;
    args.src_width = src_width;
    args.src_height = src_height;
    args.dst = dst;
    args.dst_width = dst_width;
    args.dst_height = dst_height;
    pi_cl_team_fork(CLUSTER_CORES, parallel_resize_worker, &args);
}


//=======================================================================================================================================================================================================================================
// (2) Parallel Build Pyramid

/* 建立圖像金字塔 */
unsigned char** build_pyramid_parallel(unsigned char* image, int width, int height, int channels, int levels, float scale, int* out_sizes) {
    unsigned char** pyramid = (unsigned char**) pmsis_l2_malloc(levels * sizeof(unsigned char*));
    if (!pyramid) {
        printf("Memory allocation failed for image pyramid.\n");
        return NULL;
    }
    int alloc_sizes[levels];
    int current_width = width;
    int current_height = height;
    alloc_sizes[0] = width * height * channels;
    pyramid[0] = (unsigned char*) pmsis_l2_malloc(alloc_sizes[0]);
    if (!pyramid[0]) {
        printf("Memory allocation failed for level 0 of pyramid.\n");
        pi_l2_free(pyramid, levels * sizeof(unsigned char*));
        return NULL;
    }
    memcpy(pyramid[0], image, alloc_sizes[0]);
    for (int level = 1; level < levels; level++) {
        int new_width = (int)(current_width * scale);
        int new_height = (int)(current_height * scale);
        alloc_sizes[level] = new_width * new_height * channels;
        pyramid[level] = (unsigned char*) pmsis_l2_malloc(alloc_sizes[level]);
        if (!pyramid[level]) {
            printf("Memory allocation failed for level %d of pyramid.\n", level);
            for (int i = 0; i < level; i++) {
                pi_l2_free(pyramid[i], alloc_sizes[i]);
            }
            pi_l2_free(pyramid, levels * sizeof(unsigned char*));
            return NULL;
        }
        // 使用平行化的縮放函式
        resize_image_uint8_linear_parallel(pyramid[level - 1], current_width, current_height, pyramid[level], new_width, new_height);
        out_sizes[level * 2]     = new_width;
        out_sizes[level * 2 + 1] = new_height;
        current_width = new_width;
        current_height = new_height;
    }
    return pyramid;
}


//=======================================================================================================================================================================================================================================
// (3) Nearest neighbor interpolation function
float nearest_neighbor_interpolate(unsigned char* img, float x, float y, int width, int height) {
    int nearest_x = (int)round(x);
    int nearest_y = (int)round(y);

    // Check if coordinates are within the bounds of the image
    if (nearest_x < 0 || nearest_x >= width || nearest_y < 0 || nearest_y >= height) {
        return 0.0;  // Return zero if out of bounds
    }

    return (float)img[nearest_y * width + nearest_x];
}

//=======================================================================================================================================================================================================================================
// (4) 取得像素值 (檢查邊界)
int get_value_int(unsigned char* img, int x, int y, int width, int height) {

    // Check if coordinates are within the bounds of the image
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0;  // Return zero if out of bounds
    }

    return img[y * width + x];
}

//=======================================================================================================================================================================================================================================
// (5) 光流計算：對單一點在兩張影像間估計位移(dx,dy)
void calculate_optical_flow(unsigned char* img1, unsigned char* img2, int width, int height, int x, int y, float* dx, float* dy, int half_patch_size, int iterations) {
    float lastCost = 0.0, cost = 0.0;
    float H[2][2], b[2], J[2];
    float update[2];

    for (int it = 0; it < iterations; it++) {
        memset(H, 0, 4 * sizeof(float));
        memset(b, 0, 2 * sizeof(float));
        cost = 0.0;

        for (int i = -half_patch_size; i <= half_patch_size; i++) {
            for (int j = -half_patch_size; j <= half_patch_size; j++) {
                int Ix = x + i, Iy = y + j;
                float Dx = x + *dx + i, Dy = y + *dy + j;void calculate_optical_flow(unsigned char* img1, unsigned char* img2, int width, int height, int x, int y, float* dx, float* dy, int half_patch_size, int iterations) {
                    float lastCost = 0.0, cost = 0.0;
                    float H[2][2], b[2], J[2];
                    float update[2];
                
                    for (int it = 0; it < iterations; it++) {
                        memset(H, 0, 4 * sizeof(float));
                        memset(b, 0, 2 * sizeof(float));
                        cost = 0.0;
                
                        for (int i = -half_patch_size; i <= half_patch_size; i++) {
                            for (int j = -half_patch_size; j <= half_patch_size; j++) {
                                int Ix = x + i, Iy = y + j;
                                float Dx = x + *dx + i, Dy = y + *dy + j;
                
                                float error = get_value_int(img1, Ix, Iy, width, height) - nearest_neighbor_interpolate(img2, Dx, Dy, width, height);
                                if (it == 0) {
                                    J[0] = -0.5 * (get_value_int(img2, Ix + 1, Iy, width, height) - get_value_int(img2, Ix - 1, Iy, width, height));
                                    J[1] = -0.5 * (get_value_int(img2, Ix, Iy + 1, width, height) - get_value_int(img2, Ix, Iy - 1, width, height));
                                }
                
                                b[0] -= error * J[0];
                                b[1] -= error * J[1];
                                //cost += error * error;  // Sum of Squared Difference
                                cost += fabs(error);  // Sum of Absolute Difference (faster)
                                H[0][0] += J[0] * J[0];
                                H[0][1] += J[0] * J[1];
                                H[1][0] += J[1] * J[0];
                                H[1][1] += J[1] * J[1];
                            }
                        }
                
                        float det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
                        if (fabs(det) < 1e-6) {
                            break;  // Exit if matrix is singular
                        }
                
                        float invH[2][2];  // Inverse of H
                        invH[0][0] = H[1][1] / det;
                        invH[0][1] = -H[0][1] / det;
                        invH[1][0] = -H[1][0] / det;
                        invH[1][1] = H[0][0] / det;
                
                        update[0] = invH[0][0] * b[0] + invH[0][1] * b[1];
                        update[1] = invH[1][0] * b[0] + invH[1][1] * b[1];
                
                        if (fabs(update[0]) < 1e-2 && fabs(update[1]) < 1e-2) {
                            break;  // Break if updates are small
                        }
                
                        *dx += update[0];
                        *dy += update[1];
                
                        if (it > 0 && cost > lastCost) {
                            break;  // Break if cost increases
                        }
                        lastCost = cost;
                    }
                }

                float error = get_value_int(img1, Ix, Iy, width, height) - nearest_neighbor_interpolate(img2, Dx, Dy, width, height);
                if (it == 0) {
                    J[0] = -0.5 * (get_value_int(img2, Ix + 1, Iy, width, height) - get_value_int(img2, Ix - 1, Iy, width, height));
                    J[1] = -0.5 * (get_value_int(img2, Ix, Iy + 1, width, height) - get_value_int(img2, Ix, Iy - 1, width, height));
                }

                b[0] -= error * J[0];
                b[1] -= error * J[1];
                //cost += error * error;  // Sum of Squared Difference
                cost += fabs(error);  // Sum of Absolute Difference (faster)
                H[0][0] += J[0] * J[0];
                H[0][1] += J[0] * J[1];
                H[1][0] += J[1] * J[0];
                H[1][1] += J[1] * J[1];
            }
        }

        float det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
        if (fabs(det) < 1e-6) {
            break;  // Exit if matrix is singular
        }

        float invH[2][2];  // Inverse of H
        invH[0][0] = H[1][1] / det;
        invH[0][1] = -H[0][1] / det;
        invH[1][0] = -H[1][0] / det;
        invH[1][1] = H[0][0] / det;

        update[0] = invH[0][0] * b[0] + invH[0][1] * b[1];
        update[1] = invH[1][0] * b[0] + invH[1][1] * b[1];

        if (fabs(update[0]) < 1e-2 && fabs(update[1]) < 1e-2) {
            break;  // Break if updates are small
        }

        *dx += update[0];
        *dy += update[1];

        if (it > 0 && cost > lastCost) {
            break;  // Break if cost increases
        }
        lastCost = cost;
    }
}

//=======================================================================================================================================================================================================================================
// (6)  Parallel Optical Flow Pyramid

// 定義平行光流計算參數結構
typedef struct {
    int start_index;      // 此層要處理的起始網格索引
    int end_index;        // 此層要處理的結束網格索引（不含）
    int grid_count_x;     // 橫向網格數
    int grid_spacing;     // 網格間距
    int scaled_width;     // 當前層影像寬度
    int scaled_height;    // 當前層影像高度
    unsigned char* img1;  // 金字塔此層影像1（前幀）
    unsigned char* img2;  // 金字塔此層影像2（當前幀）
    float* flow_x;        // 此層初始 x 光流向量
    float* flow_y;        // 此層初始 y 光流向量
    int half_patch_size;  // LK 半視窗大小
    int iterations;       // LK 迭代次數
} flow_args_t;

// 每個核心負責計算部分網格點的光流
void parallel_flow_worker(void* arg) {
    flow_args_t *f = (flow_args_t*)arg;
    int nb_cores = CLUSTER_CORES;        // 總核心數
    int core_id  = pi_core_id();                    // 這個核心的編號
    int total    = f->end_index - f->start_index;
    int per_core = total / nb_cores;                // 每核心處理的點數

    // 計算這個核心要處理的區間
    int start = f->start_index + core_id * per_core;
    int end   = (core_id == nb_cores - 1) ? f->end_index : start + per_core;

    // 印出這個核心負責的區間
    // printf("  [Core %d] handling idx %d..%d\n", core_id, start, end - 1);

    for (int idx = start; idx < end; idx++) {
        int row = idx / f->grid_count_x;
        int col = idx % f->grid_count_x;
        int x = f->grid_spacing/2 + col * f->grid_spacing;
        int y = f->grid_spacing/2 + row * f->grid_spacing;

        // 讀入初始光流
        float dx = f->flow_x[idx];
        float dy = f->flow_y[idx];
        // printf("    [Core %d] idx=%2d @(%3d,%3d) in=(%6.2f,%6.2f)\n", core_id, idx, x, y, dx, dy);

        // 呼叫 LK 計算
        calculate_optical_flow(
            f->img1, f->img2,
            f->scaled_width, f->scaled_height,
            x, y, &dx, &dy,
            f->half_patch_size, f->iterations
        );

        // 寫回結果
        f->flow_x[idx] = dx;
        f->flow_y[idx] = dy;
        // printf("    [Core %d] idx=%2d out=(%6.2f,%6.2f)\n", core_id, idx, dx, dy);
    }
}

/* 計算金字塔層級光流 */
void calculate_optical_flow_pyramid_parallel(
    unsigned char** pyr1, unsigned char** pyr2,
    int width, int height, int channels,
    int levels, int grid_w, int grid_h,
    float** flow_vectors_x, float** flow_vectors_y
){
    float scale = 0.5f;
    int grid_pts = grid_w * grid_h;

    float *prev_x = NULL, *prev_y = NULL;
    int prev_pts = 0;

    for (int l = levels - 1; l >= 0; l--) {
        int sw = (int)(width  * powf(scale, l));
        int sh = (int)(height * powf(scale, l));
        int gs = sw / grid_w;

        int gx = 0, gy = 0;
        for (int x = gs / 2; x < sw; x += gs) gx++;
        for (int y = gs / 2; y < sh; y += gs) gy++;
        int pts = grid_pts;

        float *lvl_x = pmsis_l2_malloc(grid_pts * sizeof(float));
        float *lvl_y = pmsis_l2_malloc(grid_pts * sizeof(float));
        if (!lvl_x || !lvl_y) {
            printf("[L=%d] Memory allocation failed\n", l);
            return;
        }
        memset(lvl_x, 0, grid_pts * sizeof(float));
        memset(lvl_y, 0, grid_pts * sizeof(float));

        if (prev_x && l != levels - 1) {
            pi_l2_free(prev_x, grid_pts * sizeof(float));
            pi_l2_free(prev_y, grid_pts * sizeof(float));
        }

        if (prev_x) {
            for (int i = 0; i < grid_pts; i++) {
                lvl_x[i] = prev_x[i] * 2.0f;
                lvl_y[i] = prev_y[i] * 2.0f;
            }
        }

        if (!pyr1 || !pyr2) {
            printf("Build pyramid failed\n");
            return;
        }

        flow_args_t fargs = {
            .start_index    = 0,
            .end_index      = pts,
            .grid_count_x   = gx,
            .grid_spacing   = gs,
            .scaled_width   = sw,
            .scaled_height  = sh,
            .img1           = pyr1[l],
            .img2           = pyr2[l],
            .flow_x         = lvl_x,
            .flow_y         = lvl_y,
            .half_patch_size= 4,
            .iterations     = 10
        };

        pi_cl_team_fork(CLUSTER_CORES, parallel_flow_worker, &fargs);

        if (l == 0) {
            memcpy(*flow_vectors_x, lvl_x, pts * sizeof(float));
            memcpy(*flow_vectors_y, lvl_y, pts * sizeof(float));
        }

        prev_x = lvl_x;
        prev_y = lvl_y;
        prev_pts = pts;
    }

    if (prev_x) pi_l2_free(prev_x, prev_pts * sizeof(float));
    if (prev_y) pi_l2_free(prev_y, prev_pts * sizeof(float));
}


//===========================================
// (7) 使用 Bresenham 演算法繪製直線
void draw_line(uint8_t* img, int width, int height, int x0, int y0, int x1, int y1, uint8_t color) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    for (;;) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height)
            img[y0 * width + x0] = color;
        if (x0 == x1 && y0 == y1)
            break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

typedef struct {
    float* flow_x;
    float* flow_y;
} cluster_args_t;

static void cluster(void *args) {
    cluster_args_t *cl_args = (cluster_args_t*) args;

    int sizes[4] = {
        CAMERA_WIDTH, CAMERA_HEIGHT,
        CAMERA_WIDTH / 2, CAMERA_HEIGHT / 2
    };
    int levels = PYRAMID_LEVELS;

    unsigned char** pyr1 = build_pyramid_parallel(Input_2, CAMERA_WIDTH, CAMERA_HEIGHT, 1, levels, 0.5f, sizes);
    unsigned char** pyr2 = build_pyramid_parallel(Input_1, CAMERA_WIDTH, CAMERA_HEIGHT, 1, levels, 0.5f, sizes);

    if (!pyr1 || !pyr2) {
        printf("[Cluster] Build pyramid failed\n");
        return;
    }

    calculate_optical_flow_pyramid_parallel(
        pyr1, pyr2,
        CAMERA_WIDTH, CAMERA_HEIGHT, 1,
        PYRAMID_LEVELS, GRID_SIZE_X, GRID_SIZE_Y,
        &cl_args->flow_x, &cl_args->flow_y
    );

    for(int lev = 0; lev < levels; lev++) {
        int w = sizes[2 * lev];
        int h = sizes[2 * lev + 1];
        pi_l2_free(pyr1[lev], w * h * sizeof(uint8_t));
        pi_l2_free(pyr2[lev], w * h * sizeof(uint8_t));
    }
    pi_l2_free(pyr1, levels * sizeof(unsigned char*));
    pi_l2_free(pyr2, levels * sizeof(unsigned char*));
}





int test_lk_pyramid_flow(void){
    printf("Entering main controller\n");

    /* LK PYRAMID INITIAL PARTITION*/
    // unsigned char** pyr1 = NULL;
    // unsigned char** pyr2 = NULL;
    // int levels = 2;                     // 金字塔層數 (本例中建立兩層影像金字塔)
    // int sizes[4] = {64, 64, 32, 32};    // 金字塔各層尺寸：Level 0 為 64x64，Level 1 為 32x32
    // int frame_size = 64 * 64;           // 每張影像總像素數

#ifndef __EMUL__
    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, (void *)&cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC * 1000 * 1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL * 1000 * 1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE * 1000 * 1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
    printf("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIIPH Frequency = %d Hz\n",
           pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

    if (PMU_set_voltage(1200, 0))
    {
        printf("Failed to set voltage\n");
        pmsis_exit(-4);
    }
    printf("Set voltage to 1.2V\n");
#endif

    /* 開啟攝影機 */
    if (open_camera_himax(&cam)){
        printf("Failed to open camera\n");
        pmsis_exit(-2);
    }
    printf("Camera opened\n");
    printf("Will capture images at %d x %d resolution\n", CAMERA_WIDTH, CAMERA_HEIGHT);

    /* 配置影像緩衝區 */
    imgBuff0 = (uint8_t *)pmsis_l2_malloc((CAMERA_WIDTH * CAMERA_HEIGHT) * sizeof(uint8_t));
    if (imgBuff0 == NULL){
        printf("Failed to allocate Memory for Image, asking for: %d x %d\n", CAMERA_WIDTH, CAMERA_HEIGHT);
        pmsis_exit(-1);
    }
    imgBuff1 = (uint8_t *)pmsis_l2_malloc((CAMERA_WIDTH * CAMERA_HEIGHT) * sizeof(uint8_t));
    if (imgBuff1 == NULL){
        printf("Failed to allocate Memory for Image, asking for: %d x %d\n", CAMERA_WIDTH, CAMERA_HEIGHT);
        pmsis_exit(-1);
    }

    /* Init and open UART */
    pi_uart_conf_init(&uart_conf);
    uart_conf.enable_tx = 1;
    uart_conf.enable_rx = 1;
    uart_conf.baudrate_bps = 115200;
    pi_open_from_conf(&uart, &uart_conf);
    if (pi_uart_open(&uart)){
        printf("UART failed to open!\n");
        pmsis_exit(-1);
    }


    shared_flow_vectors_x = pmsis_l2_malloc(GRID_SIZE_X * GRID_SIZE_Y * sizeof(float));
    shared_flow_vectors_y = pmsis_l2_malloc(GRID_SIZE_X * GRID_SIZE_Y * sizeof(float));
    if (!shared_flow_vectors_x || !shared_flow_vectors_y) {
        printf("Failed to allocate flow vectors\n");
        pmsis_exit(-1);
    }

    cluster_args_t cl_args = {
        .flow_x = shared_flow_vectors_x,
        .flow_y = shared_flow_vectors_y
    };

    printf("Call cluster\n");
#ifndef __EMUL__
    struct pi_cluster_task task;
    pi_cluster_task(&task, NULL, NULL);
    task.entry = cluster;
    task.arg = &cl_args;
    task.stack_size = (unsigned int)STACK_SIZE;
    task.slave_stack_size = (unsigned int)SLAVE_STACK_SIZE;
#endif


    /*
     * 設定 Input_1 與 Input_2 用於儲存攝影機捕獲的影像。
     * 第一輪迴圈中，由於沒有前一幀，僅捕獲影像存入 Input_1。
     * 隨後藉由交換指標，保持新舊兩幀分別位於 Input_1 與 Input_2。
     */
    Input_1 = imgBuff0;
    Input_2 = imgBuff1;

    /* 配置 LED 輸出 */
    pi_gpio_pin_configure(&gpio_device, 2, PI_GPIO_OUTPUT);

    /*
     * 主循環：持續捕獲攝影機影像，並藉由交換緩衝區指標更新前後幀資料。
     */
    while (1){
        // 切換 LED 狀態
        pi_gpio_pin_write(&gpio_device, 2, led_val);
        led_val ^= 1;

        // 捕獲新影像，存入 Input_1 (假設 Input_1 為最新影像緩衝區)
        if (pi_camera_control(&cam, PI_CAMERA_CMD_START, 0)){
            printf("Failed to start camera\n");
            pmsis_exit(-3);
        }

        pi_camera_capture(&cam, Input_1, CAMERA_WIDTH * CAMERA_HEIGHT);

        if (pi_camera_control(&cam, PI_CAMERA_CMD_STOP, 0)){
            printf("Failed to stop camera\n");
            pmsis_exit(-3);
        }

        // printf("\n------------------------------------------------------------------------------------------------------------------\n");
        // printf("Frame %lu captured.\n", idx);
        


#ifndef __EMUL__
        // uint32_t time_before = pi_time_get_us();
        pi_cluster_send_task_to_cl(&cluster_dev, &task);
        // uint32_t time_after = pi_time_get_us();
        // printf("Computation time: %dus \n", time_after - time_before);
#else
        cluster();
#endif

        // printf("\nFlow vectors:\n");
        // for (int i = 0; i < GRID_SIZE_X*GRID_SIZE_Y; i++) {
        //     printf("( %6.2f , %6.2f ) ", shared_flow_vectors_x[i], shared_flow_vectors_y[i]);
        //     if ((i+1) % GRID_SIZE_X == 0)
        //         printf("\n");
        // }



        // ……迴圈裡……
       if (idx > 15) {
            // 改為方向判定
            double out_sum_left = 0.0f;
            double out_sum_right = 0.0f;
            int32_t direction = flow_state_from_direction(
                shared_flow_vectors_x,
                shared_flow_vectors_y,
                GRID_SIZE_X,
                GRID_SIZE_Y,
                &out_sum_left,
                &out_sum_right
            );

            // 若要平滑方向，也可以用過去方向進行統計投票（可選）
            final_error_u8 = (uint8_t)direction;

            // // 顯示資訊
            // const char* dir_str[] = {"Forward", "Right", "Left"};
            // printf("img %3d: direction = %d (%s), left_sum = %.2f, right_sum = %.2f\n", idx, direction, dir_str[direction], out_sum_left, out_sum_right);
        }
        else {
            final_error_u8 = 0;  // warming up時預設方向為0 (Right)
            // printf("img %3d: warming up… final = %3u\n", idx, final_error_u8);
        }

        // 一律傳 1-byte 無符號整數
        pi_uart_write(&uart, &final_error_u8, 1);


        // if (idx > 15) {
        //     int32_t direction = flow_state_from_direction_with_avgdy(
        //         shared_flow_vectors_x,
        //         shared_flow_vectors_y,
        //         GRID_SIZE_X,
        //         GRID_SIZE_Y
        //     );

        //     if (direction == -1) {
        //         final_error_u8 = 255;
        //     } else {
        //         final_error_u8 = (uint8_t)direction;
        //         const char* dir_str[] = {"Right", "Down", "Left", "Up"};
        //         printf("img %3d: dir = %d (%s)\n", idx, direction, dir_str[direction]);
        //     }
        // }
        // else {
        //     final_error_u8 = 0;
        //     printf("img %3d: warming up… final = %3u\n", idx, final_error_u8);
        // }


        // // 一律傳 1-byte 無符號整數
        // pi_uart_write(&uart, &final_error_u8, 1);

        
        

        // printf("img %d error = %d \n", idx, error);

        /* save raw image*/
        // 建立資料夾名稱與檔名前綴
        // const char *folder = "img" STR(GRID_SIZE_X) "_" STR(GRID_SIZE_Y);
        // const char *prefix = "draw_flow_" STR(GRID_SIZE_X) "_" STR(GRID_SIZE_Y);
        // {
        //     char filename[64];
        //     snprintf(filename, sizeof(filename),
        //              "../../../%s/img_raw_%lu.ppm",
        //              folder, idx);
        //     WriteImageToFile(filename,
        //                      CAMERA_WIDTH, CAMERA_HEIGHT,
        //                      sizeof(uint8_t),
        //                      Input_1,
        //                      GRAY_SCALE_IO);
        // }

        // // ==== 這裡是新增的 FC 上畫線與儲存流程 ====
        // if (!drawImg)
        //     drawImg = (uint8_t*)pmsis_l2_malloc(CAMERA_WIDTH * CAMERA_HEIGHT);
        // memcpy(drawImg, Input_1, CAMERA_WIDTH * CAMERA_HEIGHT);

        // int grid_w = GRID_SIZE_X, grid_h = GRID_SIZE_Y;
        // int grid_spacing = CAMERA_WIDTH / grid_w;
        // for (int i = 0; i < grid_w * grid_h; i++) {
        //     int gx = (i % grid_w) * grid_spacing + grid_spacing / 2;
        //     int gy = (i / grid_w) * grid_spacing + grid_spacing / 2;
        //     int dx = (int)(shared_flow_vectors_x[i]);
        //     int dy = (int)(shared_flow_vectors_y[i]);
        //     draw_line(drawImg, CAMERA_WIDTH, CAMERA_HEIGHT, gx, gy, gx - dx, gy - dy, 255);
        // }

        // {
        //     char draw_fn[128];
        //     snprintf(draw_fn, sizeof(draw_fn),
        //             "../../../%s/%s_%lu.ppm",
        //             folder, prefix, idx);
        //     WriteImageToFile(draw_fn,
        //                     CAMERA_WIDTH, CAMERA_HEIGHT,
        //                     sizeof(uint8_t),
        //                     drawImg,
        //                     GRAY_SCALE_IO);
        //     pi_l2_free(drawImg, CAMERA_WIDTH * CAMERA_HEIGHT);
        //     drawImg = NULL;
        // }
        // ==================================================

        

        // 指標交換：使原本存有前一幀資料的 Input_2 成為下一輪的前一幀。
        {
            uint8_t *temp = Input_1;
            Input_1 = Input_2;
            Input_2 = temp;
        }


        idx++;


        
    
    }

    // 理論上 while(1) 不會跳出，若結束前可釋放資源：
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[]){
    printf("\n\n\t *** LK Pyramid Optical Flow Example ***\n\n");
#ifdef __EMUL__
    test_lk_pyramid_flow();
#else
    return pmsis_kickoff((void *)test_lk_pyramid_flow);
#endif
    return 0;
}

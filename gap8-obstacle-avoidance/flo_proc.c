#include "flo_proc.h"
#include <math.h>


int32_t flow_state_from_dy_focus_ycenter(
    float* flow_x, 
    float* flow_y, 
    int grid_w, 
    int grid_h, 
    float* avg_out
) {
    float sum = 0.0f;
    int count = 0;

    int y_start = grid_h / 2 - 1;
    int y_end   = grid_h / 2 + 2;

    if (y_start < 0) y_start = 0;
    if (y_end > grid_h) y_end = grid_h;

    for (int y = y_start; y < y_end; y++) {
        for (int x = 0; x < grid_w; x++) {
            int idx = y * grid_w + x;
            sum += fabsf(flow_y[idx]);
            count++;
        }
    }

    if (count == 0) {
        *avg_out = 0.0f;
        return 0;
    }

    float avg = sum / (float)count;
    *avg_out = avg;

    int32_t result = (int32_t)(avg * 100.0f);
    if (result > 127) result = 127;

    return result;
}









#include <math.h>
#include <stdint.h>

// int32_t flow_state_from_direction(float* flow_vectors_x, float* flow_vectors_y, int grid_cols, int grid_rows) {
//     double sum_left = 0, sum_right = 0, sum_up = 0, sum_down = 0;

//     for (int y = 0; y < grid_rows; y++) {
//         for (int x = 0; x < grid_cols; x++) {
//             int idx = y * grid_cols + x;
//             float dx = flow_vectors_x[idx];
//             float dy = flow_vectors_y[idx];
//             double mag = sqrt(dx * dx + dy * dy);
//             if (mag < 1e-3) continue; // 忽略非常小的光流向量

//             double angle = atan2(dy, dx) * 180.0 / M_PI; // 角度制 -180 ~ 180

//             if ((angle >= -45 && angle <= 45)) {
//                 sum_right += mag;er
//             } else if ((angle >= 135 || angle <= -135)) {
//                 sum_left += mag;
//             } else if ((angle > 45 && angle < 135)) {
//                 sum_down += mag;
//             } else if ((angle < -45 && angle > -135)) {
//                 sum_up += mag;
//             }
//         }
//     }

//     // 回傳方向編碼：0=右, 1=下, 2=左, 3=上
//     double max_val = sum_right;
//     int dir = 0;
//     if (sum_down > max_val) { max_val = sum_down; dir = 1; }
//     if (sum_left > max_val) { max_val = sum_left; dir = 2; }
//     if (sum_up > max_val)   { max_val = sum_up;   dir = 3; }

//     return dir;
// }


int32_t flow_state_from_direction(
        float* flow_vectors_x, 
        float* flow_vectors_y, 
        int grid_cols, 
        int grid_rows, 
        double* out_sum_left, 
        double* out_sum_right
    ) {
    double sum_left = 0, sum_right = 0;
    float threshold = 100;
    for (int y = 0; y < grid_rows; y++) {
        for (int x = 0; x < grid_cols; x++) {
            int idx = y * grid_cols + x;
            float dx = flow_vectors_x[idx];
            float dy = flow_vectors_y[idx];
            double mag = sqrt(dx * dx + dy * dy);
            if (mag < 1e-3) continue; // 忽略非常小的光流向量

            // double angle = atan2(dy, dx) * 180.0 / M_PI; // 角度制 -180 ~ 180
            double angle = atan2(dy, dx);
            // if ((angle >= -45 && angle <= 45)) {
            //     sum_right += mag;
            // } else if ((angle >= 135 || angle <= -135)) {
            //     sum_left += mag;
            // }
            // // 忽略上下方向的流
            // if (angle > -90 && angle <= 90) {  // 右半平面 (-90°, 90°]
            //     sum_right += mag;
            // } else {                          // 左半平面 (90°, 180°] 和 [-180°, -90°]
            //     sum_left += mag;
            // }
            if (angle > -M_PI_2 && angle <= M_PI_2) {
                sum_right += mag;
            } else {
                sum_left += mag;
            }
        }
    }
    // 透過指標回傳 sum_left 和 sum_right
    // printf("Debug - sum_left: %.3f, sum_right: %.3f\n", sum_left, sum_right);
    *out_sum_left = sum_left;
    *out_sum_right = sum_right;
    // 如果總流動量不足閾值，返回前進
    // if ((sum_left + sum_right) < threshold) {
    //     return 0; // 假設0代表前進
    // }
    
    // 否則返回主要流動方向
    // return (sum_left > sum_right) ? 2 : 1; // 2=左, 0=右
    return (sum_left > sum_right) ? 
        ((sum_left > threshold) ? 2 : 0) : 
        ((sum_right > threshold) ? 1 : 0);
}



int32_t flow_state_from_direction_with_avgdy(
    float* flow_x,
    float* flow_y,
    int grid_w,
    int grid_h
) {
    // Step 1: 先算 avg_dy 中心三列
    float avg_dy = 0.0f;
    int32_t raw_error = flow_state_from_dy_focus_ycenter(flow_x, flow_y, grid_w, grid_h, &avg_dy);
    // printf("raw error = %d\n", raw_error);

    // Step 2: 判斷是否低於門檻
    if (raw_error < 100) {
        printf("no obstacle\n");
        return -1;  // 無障礙物
    }

    // Step 3: 正常方向統計
    double sum_left = 0, sum_right = 0, sum_up = 0, sum_down = 0;

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            int idx = y * grid_w + x;
            float dx = flow_x[idx];
            float dy = flow_y[idx];
            double mag = sqrt(dx * dx + dy * dy);
            if (mag < 1e-3) continue;

            double angle = atan2(dy, dx) * 180.0 / M_PI;

            if ((angle >= -45 && angle <= 45)) {
                sum_right += mag;
            } else if ((angle >= 135 || angle <= -135)) {
                sum_left += mag;
            } else if ((angle > 45 && angle < 135)) {
                sum_down += mag;
            } else if ((angle < -45 && angle > -135)) {
                sum_up += mag;
            }
        }
    }

    // Step 4: 回傳方向（0=右, 1=下, 2=左, 3=上）
    double max_val = sum_right;
    int dir = 0;
    if (sum_down > max_val) { max_val = sum_down; dir = 1; }
    if (sum_left > max_val) { max_val = sum_left; dir = 2; }
    if (sum_up > max_val)   { max_val = sum_up;   dir = 3; }

    return dir;
}

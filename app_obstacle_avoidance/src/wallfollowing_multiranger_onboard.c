/*
* wall_follower_multi_ranger_onboard.c
*
*  Created on: Aug 7, 2018
*      Author: knmcguire
The same wallfollowing strategy was used in the following paper:
@article{mcguire2019minimal,
title={Minimal navigation solution for a swarm of tiny flying robots to explore an unknown environment},
author={McGuire, KN and De Wagter, Christophe and Tuyls, Karl and Kappen, HJ and de Croon, Guido CHE},
journal={Science Robotics},
volume={4},
number={35},
year={2019},
publisher={Science Robotics}
*/

#include "wallfollowing_multiranger_onboard.h"
#include <math.h>
#include "debug.h"

// variables
static float maxForwardSpeed = 0.15f;
// static float maxTurnRate = 0.5f;
static float firstRun = false;
static float prevHeading = 0.0f;
static bool aroundCornerBackTrack = false;
static float stateStartTime;
static const float inCornerAngle = 0.8f;
static const float waitForMeasurementSeconds = 5.0f;

static StateWF stateWF = forward;
float timeNow = 0.0f;

//====================================================================
//  EXPERIMENT
int count = 0;
int heightBit = 0;

const double OBSTACLE_THRESH = 110.0f;        // 設定光流避障用的閾值
const double OBSTACLE_CLEAR_THRESHOLD = 110.0f; // Error小於1.0才算清除
const float AVOID_DISTANCE_THRESHOLD = 0.40f; // m
static int avoidDirection = 0; // -1 left, 1 right





void wallFollowerInit(float maxForwardSpeed_ref, StateWF initState){
    // 僅在第一次初始化時更新 maxForwardSpeed
    // 避免進入hover回到forward時不會往前
    if (firstRun) {
        maxForwardSpeed = maxForwardSpeed_ref;
    }
    firstRun = true;
    stateWF = initState;
}

static void commandHover(float *cmdVelX, float *cmdVelY, float *cmdAngW){
    *cmdVelX = 0.0f;
    *cmdVelY = 0.0f;
    *cmdAngW = 0.0f;
}

static StateWF transition(StateWF newState){
    stateStartTime = timeNow;
    return newState;
}


 StateWF wallFollower(float *cmdVelX, float *cmdVelY, float *cmdAngW, float *cmdHeight, 
    float currentHeading, float timeOuter, double yawRate)
{
    timeNow = timeOuter;

    if (firstRun){
        prevHeading = currentHeading;
        aroundCornerBackTrack = false;
        firstRun = false;
    }

    // DEBUG_PRINT("yawRate = %f", yawRate);

    /***********************************************************
    * 處理狀態轉換（如果需要）
//     ***********************************************************/
//    switch (stateWF)
//    {
//        case forward:        DEBUG_PRINT("🚀 State: FORWARD\n"); break;
//        case hover:          DEBUG_PRINT("🛑 State: HOVER\n"); break;
//        case hover_left:     DEBUG_PRINT("🛑 State: HOVER_LEFT\n"); break;
//        case hover_right:    DEBUG_PRINT("🛑 State: HOVER_RIGHT\n"); break;
//        case avoid_phase2:   DEBUG_PRINT("🛑 State: AVOID_PHASE2\n"); break; 
//        case avoid_left:     DEBUG_PRINT("⬅️  State: AVOID_LEFT\n"); break;
//        case avoid_right:    DEBUG_PRINT("➡️  State: AVOID_RIGHT\n"); break;
//        case turn_around:    DEBUG_PRINT("↩️  State: TURN AROUND\n"); break;
//        case turn_back:      DEBUG_PRINT("↩️  State: TURN BACK\n"); break;
//        default:             DEBUG_PRINT("❓ State: UNKNOWN → HOVER\n"); stateWF = transition(hover); break;
//    }
   

    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;
    float cmdHeightTemp = 0.0f;       

    /***********************************************************
    * 狀態機主邏輯
    ***********************************************************/
    switch (stateWF)
    {
        case forward:
        {
            cmdVelXTemp = maxForwardSpeed;
            cmdVelYTemp = 0.0f;
            cmdHeightTemp = 0.4f;
        
            // if (fabs(yawRate) == 1) {
            //     // 只在检测到障碍时切到 hover
            //     stateWF = transition(hover_left);
            // }
            // else if (yawRate == 2) {
            //     stateWF = transition(hover_right);
            // }
            // 否则什么都不做，继续留在 forward
             if (fabs(yawRate) == 2) {
                // 只在检测到障碍时切到 hover
                stateWF = transition(avoid_left);
            }
            else if (yawRate == 1) {
                stateWF = transition(avoid_right);
            }
            avoidDirection = 0;
        }
        break;

        case hover_left:
        {
            // 完全懸停
            commandHover(&cmdVelXTemp, &cmdVelYTemp, &cmdAngWTemp);
            cmdHeightTemp = 0.4f;
        
            // 檢查懸停時間
            if (timeNow - stateStartTime >= 0.1f) {
                stateWF = transition(avoid_left);
            }
        }
        break;
        
        case hover_right:
        {
            // 完全懸停
            commandHover(&cmdVelXTemp, &cmdVelYTemp, &cmdAngWTemp);
            cmdHeightTemp = 0.4f;
        
            // 檢查懸停時間
            if (timeNow - stateStartTime >= 0.1f) {
                stateWF = transition(avoid_right);
            }
        }
        break;
        

    
        case avoid_left:
        case avoid_right:
        {
            float moveSpeed = (stateWF == avoid_left) ? -0.2f : 0.2f;
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = moveSpeed;
            cmdAngWTemp = 0.0f;
            cmdHeightTemp = 0.4f;
    
            float moved_dist = fabsf((timeNow - stateStartTime) * moveSpeed);
    
            if (moved_dist >= AVOID_DISTANCE_THRESHOLD) {
                // stateWF = transition(forward);
                stateWF = transition(avoid_check);
            }
        }
        break;

        case avoid_check:
        {
            // 完全懸停
            commandHover(&cmdVelXTemp, &cmdVelYTemp, &cmdAngWTemp);
            cmdHeightTemp = 0.8f;
        
            // 檢查懸停時間
            if (timeNow - stateStartTime >= 0.3f) {
                stateWF = transition(forward);
            }
        }
        break;

        // case avoid_phase2:
        //     commandHover(&cmdVelXTemp, &cmdVelYTemp, &cmdAngWTemp);
        //     cmdHeightTemp = 0.4f;
    
        //     if (timeNow - stateStartTime >= 1.0f) {
        //         if (fabs(yawRate) < OBSTACLE_CLEAR_THRESHOLD) {
        //             DEBUG_PRINT("✅ Obstacle cleared, going FORWARD\n");
        //             stateWF = transition(forward);
        //         }
        //         else {
        //             DEBUG_PRINT("❗ Obstacle still detected, retry avoidance\n");
        //             if (avoidDirection == -1) {
        //                 stateWF = transition(avoid_left);
        //             } else {
        //                 stateWF = transition(avoid_right);
        //             }
        //         }
        //     }
        //     break;
    
        case turn_around:
        case turn_back:
            commandHover(&cmdVelXTemp, &cmdVelYTemp, &cmdAngWTemp);
            cmdHeightTemp = 0.4f;
            // 現在不觸發轉向
            break;
    
        default:
            commandHover(&cmdVelXTemp, &cmdVelYTemp, &cmdAngWTemp);
            cmdHeightTemp = 0.4f;
            break;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;
    *cmdHeight = cmdHeightTemp;

    return stateWF;
}

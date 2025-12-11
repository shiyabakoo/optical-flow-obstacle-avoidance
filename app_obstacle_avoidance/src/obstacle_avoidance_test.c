/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2021 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * based on wall_follower.c - App layer application of the wall following demo. Modified for optical flow-based obstacle avoidance.
 *
 * The same wallfollowing strategy was used in the following paper:

 @article{mcguire2019minimal,
  title={Minimal navigation solution for a swarm of tiny flying robots to explore an unknown environment},
  author={McGuire, KN and De Wagter, Christophe and Tuyls, Karl and Kappen, HJ and de Croon, Guido CHE},
  journal={Science Robotics},
  volume={4},
  number={35},
  year={2019},
  publisher={Science Robotics}
}

 */


// #include <string.h>
// #include <stdint.h>
// #include <stdbool.h>

// #include "app.h"

// #include "commander.h"
// #include "supervisor.h"
// #include "FreeRTOS.h"
// #include "task.h"

// #include "debug.h"

// #include "log.h"
// #include "param.h"
// #include <math.h>
// #include "usec_time.h"

// #include "deck.h"

// #include "uart1.h"

// #include "wallfollowing_multiranger_onboard.h"

// #include "system.h"

// // static uint8_t byte;
// // static int8_t yaw_command, temp;
// static int8_t yaw_command;
// // double yaw_rate;

// float heightEstimate;

// float maxForwardSpeed = 0.2f;
// static const float hoverHeight = 0.5f;

// float cmdVelX = 0.0f;
// float cmdVelY = 0.0f;
// float cmdHeight = 0.5f;
// float cmdAngWRad = 0.0f;
// float cmdAngWDeg = 0.0f;

// float stateStartTime, timeNow;
// const float AVOID_DISTANCE_THRESHOLD = 0.40f; // m


// // static void receiveByteTask(void *param)
// // {
// //   systemWaitStart();
// //   vTaskDelay(M2T(10));


// //   // Read out the byte the Gap8 sends and immediately send it to the console.
// //   while (1)
// //   {
// //     uart1GetDataWithTimeout(&byte, portMAX_DELAY);
// //     // DEBUG_PRINT("AI-deck -> byte = %u\n", byte);
// //     // consolePutchar(byte);
// //   }
// // }

// static void setVelocitySetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate)
// {
//   setpoint->mode.z = modeAbs;
//   setpoint->position.z = z;
//   setpoint->mode.yaw = modeVelocity;
//   setpoint->attitudeRate.yaw = yawrate;
//   setpoint->mode.x = modeVelocity;
//   setpoint->mode.y = modeVelocity;
//   setpoint->velocity.x = vx;
//   setpoint->velocity.y = vy;
//   setpoint->velocity_body = true;
// }

// static void commandHover(float *cmdVelX, float *cmdVelY, float *cmdAngW){
//     *cmdVelX = 0.0f;
//     *cmdVelY = 0.0f;
//     *cmdAngW = 0.0f;
// }

// static StateWF transition(StateWF newState){
//     stateStartTime = timeNow;
//     return newState;
// }

// // States
// StateWF stateInnerLoop = forward;
// StateWF stateWF;
// void appMain()
// {
//   // set baudrate
// //   vTaskDelay(M2T(1000));
// //   uart1Init(115200);
// //   vTaskDelay(M2T(1000));
//   // arm
//   vTaskDelay(M2T(2000));
//   supervisorRequestArming(true);
//   vTaskDelay(M2T(2000));
//   // Getting the Logging IDs of the state estimates
//   // logVarId_t idStabilizerYaw = logGetVarId("stabilizer", "yaw");
//   logVarId_t idHeightEstimate = logGetVarId("stateEstimate", "z");

//   // // Getting Param IDs of the deck driver initialization
//   // paramVarId_t idAIdeck = paramGetVarId("deck", "bcAI");
// //   paramVarId_t idPositioningDeck = paramGetVarId("deck", "bcFlow2");
//   // check flow deck work correctly
// //   uint8_t positioningInit = paramGetUint(idPositioningDeck);

//   // Initialize the wall follower state machine
//   stateWF = TAKING_OFF;

//   // Intialize the setpoint structure
//   setpoint_t setpoint;

//   DEBUG_PRINT("Waiting for activation ...\n");

//   yaw_command = 0;

//     // Pull reset for GAP8/ESP32
// //   pinMode(DECK_GPIO_IO4, OUTPUT);
// //   digitalWrite(DECK_GPIO_IO4, LOW);

// //   xTaskCreate(receiveByteTask, "AI DECK UART READOUT", AI_DECK_TASK_STACKSIZE, NULL,
// //             AI_DECK_TASK_PRI, NULL);

// //   vTaskDelay(M2T(100));


// //   // Release reset for GAP8/ESP32
// //   digitalWrite(DECK_GPIO_IO4, HIGH);
// //   pinMode(DECK_GPIO_IO4, INPUT_PULLUP);

//   vTaskDelay(M2T(100));

//   while (1)
//   {
//     vTaskDelay(M2T(10));

//     // if (positioningInit)
//     // {
//     DEBUG_PRINT("STATE = %d\n", stateWF);
//       // Check if AI deck is properly mounted
//       // uint8_t aiInit = paramGetUint(idAIdeck);

//       // uart1GetDataWithTimeout(&byte, M2T(10));
//     //   temp = byte;
//       yaw_command = 0;

//       // Get Height estimate
//       heightEstimate = logGetFloat(idHeightEstimate);

//       cmdVelX = maxForwardSpeed;
//       cmdVelY = 0.0f;
//       cmdAngWRad = 0.0f;
//       cmdAngWDeg = 0.0f;
//       cmdHeight = hoverHeight;
//       // The wall-following state machine which outputs velocity commands
//       timeNow = usecTimestamp() / 1e6;
//       /***********************************************************
//       *                         state machine
//       ***********************************************************/
//      switch (stateWF)
//       {
//           case TAKING_OFF:
//           {
//               cmdVelX = 0;
//               cmdVelY = 0.0f;
//               cmdHeight = 0.5f;
//               if (timeNow - stateStartTime >= 2.0f) {
//                   stateWF = transition(forward);
//               }
//           }
//           break;
//           case forward:
//           {
//               cmdVelX = maxForwardSpeed;
//               cmdVelY = 0.0f;
//               cmdHeight = 0.4f;
//               if (fabs(yaw_command) == 2) {

//                   stateWF = transition(avoid_left);
//               }
//               else if (yaw_command == 1) {
//                   stateWF = transition(avoid_right);
//               }
//           }
//           break;

//           case hover_left:
//           {
//               // 完全懸停
//               commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
//               cmdHeight = 0.4f;
          
//               // 檢查懸停時間
//               if (timeNow - stateStartTime >= 0.1f) {
//                   stateWF = transition(avoid_left);
//               }
//           }
//           break;
          
//           case hover_right:
//           {
//               // 完全懸停
//               commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
//               cmdHeight = 0.4f;
          
//               // 檢查懸停時間
//               if (timeNow - stateStartTime >= 0.1f) {
//                   stateWF = transition(avoid_right);
//               }
//           }
//           break;
          

      
//           case avoid_left:
//           case avoid_right:
//           {
//               float moveSpeed = (stateWF == avoid_left) ? -0.2f : 0.2f;
//               cmdVelX = 0.0f;
//               cmdVelY = moveSpeed;
//               cmdAngWRad = 0.0f;
//               cmdHeight = 0.4f;
      
//               float moved_dist = fabsf((timeNow - stateStartTime) * moveSpeed);
      
//               if (moved_dist >= AVOID_DISTANCE_THRESHOLD) {
//                   // stateWF = transition(forward);
//                   stateWF = transition(avoid_check);
//               }
//           }
//           break;

//           case avoid_check:
//           {
//               // 完全懸停
//               commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
//               cmdHeight = 0.8f;
          
//               // 檢查懸停時間
//               if (timeNow - stateStartTime >= 0.3f) {
//                   stateWF = transition(forward);
//               }
//           }
//           break;
      
//           case turn_around:
//           case turn_back:
//               commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
//               cmdHeight = 0.4f;
//               // 現在不觸發轉向
//               break;
      
//           default:
//               commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
//               cmdHeight = 0.4f;
//               break;
//       }
//       cmdAngWDeg = cmdAngWRad * 180.0f / (float)M_PI;
//       // Turn velocity commands into setpoints and send it to the commander
//       setVelocitySetpoint(&setpoint, cmdVelX, cmdVelY, cmdHeight, cmdAngWDeg);
//       commanderSetSetpoint(&setpoint, 3);
//     // }
    
//   }
// }

// PARAM_GROUP_START(app)
// // PARAM_ADD(PARAM_UINT8, goLeft, &goLeft)
// // PARAM_ADD(PARAM_FLOAT, distanceWall, &distanceToWall)
// // PARAM_ADD(PARAM_FLOAT, yaw_gain_proportional, &yaw_gain_proportional)
// // PARAM_ADD(PARAM_FLOAT, yaw_gain_derivative, &yaw_gain_derivative)
// PARAM_GROUP_STOP(app)

// LOG_GROUP_START(app)
// LOG_ADD(LOG_INT8, yaw_command, &yaw_command)
// LOG_ADD(LOG_FLOAT, cmdVelX, &cmdVelX)
// LOG_ADD(LOG_FLOAT, cmdVelY, &cmdVelY)
// LOG_ADD(LOG_FLOAT, cmdHeight, &cmdHeight)
// LOG_ADD(LOG_FLOAT, cmdAngWDeg, &cmdAngWDeg)
// LOG_ADD(LOG_FLOAT, heightEstimate, &heightEstimate)
// LOG_GROUP_STOP(app)



#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "commander.h"
#include "supervisor.h"
#include "FreeRTOS.h"
#include "task.h"
#include "debug.h"
#include "log.h"
#include "param.h"
#include <math.h>
#include "usec_time.h"
// #include <string.h>
// #include <stdint.h>
// #include <stdbool.h>

// #include "app.h"

// #include "commander.h"
// #include "supervisor.h"
// #include "FreeRTOS.h"
// #include "task.h"

// #include "debug.h"

// #include "log.h"
// #include "param.h"
// #include <math.h>
// #include "usec_time.h"

// #include "deck.h"

// #include "uart1.h"


// #include "system.h"
static int8_t yaw_command;

float heightEstimate;

float maxForwardSpeed = 0.2f;
static const float hoverHeight = 0.5f;

float cmdVelX = 0.0f;
float cmdVelY = 0.0f;
float cmdHeight = 0.5f;
float cmdAngWRad = 0.0f;
float cmdAngWDeg = 0.0f;

float stateStartTime, timeNow;
const float AVOID_DISTANCE_THRESHOLD = 0.40f; // m

typedef enum {
    TAKING_OFF,
    forward,
    hover,
    hover_left,
    hover_right,
    left,
    avoid_left,
    avoid_right,
    avoid_check,
    avoid_phase2,  // 新增：避障後持續 5 秒再做決策
    turn_around,
    turn_back
} StateWF;

static void setVelocitySetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate)
{
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;
  setpoint->mode.yaw = modeVelocity;
  setpoint->attitudeRate.yaw = yawrate;
  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;
  setpoint->velocity_body = true;
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

// States
// StateWF stateInnerLoop = forward;
StateWF stateWF = TAKING_OFF;
void appMain()
{
  // arm
  vTaskDelay(M2T(2000));
  supervisorRequestArming(true);
  vTaskDelay(M2T(2000));
  logVarId_t idHeightEstimate = logGetVarId("stateEstimate", "z");


  // Initialize the wall follower state machine
  stateWF = TAKING_OFF;

  // Intialize the setpoint structure
  setpoint_t setpoint;

  DEBUG_PRINT("Waiting for activation ...\n");

  yaw_command = 0;

  vTaskDelay(M2T(100));

  while (1)
  {
    vTaskDelay(M2T(10));

    // if (positioningInit)
    // {
    DEBUG_PRINT("STATE = %d\n", stateWF);
    memset(&setpoint, 0, sizeof(setpoint_t));
      // Check if AI deck is properly mounted
      // uint8_t aiInit = paramGetUint(idAIdeck);

      // uart1GetDataWithTimeout(&byte, M2T(10));
    //   temp = byte;
      yaw_command = 0;

      // Get Height estimate
      heightEstimate = logGetFloat(idHeightEstimate);

      cmdVelX = maxForwardSpeed;
      cmdVelY = 0.0f;
      cmdAngWRad = 0.0f;
      cmdAngWDeg = 0.0f;
      cmdHeight = hoverHeight;
      // The wall-following state machine which outputs velocity commands
      timeNow = usecTimestamp() / 1e6;
      /***********************************************************
      *                         state machine
      ***********************************************************/
     switch (stateWF)
      {
          case TAKING_OFF:
          {
              cmdVelX = 0;
              cmdVelY = 0.0f;
              cmdHeight = 0.5f;
              if (timeNow - stateStartTime >= 2.0f) {
                  stateWF = transition(forward);
              }
          }
          break;
          case forward:
          {
              cmdVelX = maxForwardSpeed;
              cmdVelY = 0.0f;
              cmdHeight = 0.4f;
              if (fabs(yaw_command) == 2) {

                  stateWF = transition(avoid_left);
              }
              else if (yaw_command == 1) {
                  stateWF = transition(avoid_right);
              }
          }
          break;

          case hover_left:
          {
              // 完全懸停
              commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
              cmdHeight = 0.4f;
          
              // 檢查懸停時間
              if (timeNow - stateStartTime >= 0.1f) {
                  stateWF = transition(avoid_left);
              }
          }
          break;
          
          case hover_right:
          {
              // 完全懸停
              commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
              cmdHeight = 0.4f;
          
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
              cmdVelX = 0.0f;
              cmdVelY = moveSpeed;
              cmdAngWRad = 0.0f;
              cmdHeight = 0.4f;
      
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
              commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
              cmdHeight = 0.8f;
          
              // 檢查懸停時間
              if (timeNow - stateStartTime >= 0.3f) {
                  stateWF = transition(forward);
              }
          }
          break;
      
          case turn_around:
          case turn_back:
              commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
              cmdHeight = 0.4f;
              // 現在不觸發轉向
              break;
      
          default:
              commandHover(&cmdVelX, &cmdVelY, &cmdAngWRad);
              cmdHeight = 0.4f;
              break;
      }
      cmdAngWDeg = cmdAngWRad * 180.0f / (float)M_PI;
      // Turn velocity commands into setpoints and send it to the commander
      setVelocitySetpoint(&setpoint, cmdVelX, cmdVelY, cmdHeight, cmdAngWDeg);
    //   double  v_x = setpoint.velocity.x;
    //   double  v_y = setpoint.velocity.y;
    //   double  h_z = setpoint.position.z;
    //   DEBUG_PRINT("cmdVelX = %f cmdVelY = %f cmdHeight = %f\n", v_x, v_y, h_z);
      commanderSetSetpoint(&setpoint, 3);
    // }
    
  }
}

PARAM_GROUP_START(app)
PARAM_GROUP_STOP(app)

LOG_GROUP_START(app)
LOG_ADD(LOG_INT8, yaw_command, &yaw_command)
LOG_ADD(LOG_FLOAT, cmdVelX, &cmdVelX)
LOG_ADD(LOG_FLOAT, cmdVelY, &cmdVelY)
LOG_ADD(LOG_FLOAT, cmdHeight, &cmdHeight)
LOG_ADD(LOG_FLOAT, cmdAngWDeg, &cmdAngWDeg)
LOG_ADD(LOG_FLOAT, heightEstimate, &heightEstimate)
LOG_GROUP_STOP(app)




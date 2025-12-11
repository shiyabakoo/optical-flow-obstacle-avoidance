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

#include "deck.h"

#include "uart1.h"

#include "wallfollowing_multiranger_onboard.h"

#include "system.h"

static uint8_t byte;
static int8_t yaw_command;
static int8_t prev_yaw_command;

float yaw_command_derivative;
float yaw_gain_proportional;
float yaw_gain_derivative;

double yaw_rate;

float heightEstimate;

float maxForwardSpeed = 0.2f;
static const float hoverHeight = 0.5f;

float cmdVelX = 0.0f;
float cmdVelY = 0.0f;
float cmdHeight = 0.5f;
float cmdAngWRad = 0.0f;
float cmdAngWDeg = 0.0f;

static void receiveByteTask(void *param)
{
  systemWaitStart();
  vTaskDelay(M2T(10));

  // Read out the byte the Gap8 sends and immediately send it to the console.
  while (1)
  {
    uart1GetDataWithTimeout(&byte, portMAX_DELAY);
    // DEBUG_PRINT("AI-deck -> byte = %u\n", byte);
    // consolePutchar(byte);
  }
}

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

// States
StateWF stateInnerLoop = forward;

void appMain()
{

  vTaskDelay(M2T(2000));
  supervisorRequestArming(true);
  vTaskDelay(M2T(2000));
  uart1Init(115200);

  // Getting the Logging IDs of the state estimates
  logVarId_t idStabilizerYaw = logGetVarId("stabilizer", "yaw");
  logVarId_t idHeightEstimate = logGetVarId("stateEstimate", "z");

  // // Getting Param IDs of the deck driver initialization
  // paramVarId_t idAIdeck = paramGetVarId("deck", "bcAI");
  paramVarId_t idPositioningDeck = paramGetVarId("deck", "bcFlow2");

  uint8_t positioningInit = paramGetUint(idPositioningDeck);

  // Initialize the wall follower state machine
  wallFollowerInit(maxForwardSpeed, stateInnerLoop);

  // Intialize the setpoint structure
  setpoint_t setpoint;

  DEBUG_PRINT("Waiting for activation ...\n");

  yaw_gain_proportional = 0.063;  // 0.0126
  yaw_gain_derivative = 0.009;  // 0.0018

  yaw_command = 0;

    // Pull reset for GAP8/ESP32
  pinMode(DECK_GPIO_IO4, OUTPUT);
  digitalWrite(DECK_GPIO_IO4, LOW);

  xTaskCreate(receiveByteTask, "AI DECK UART READOUT", AI_DECK_TASK_STACKSIZE, NULL,
            AI_DECK_TASK_PRI, NULL);

  vTaskDelay(M2T(100));


  // Release reset for GAP8/ESP32
  digitalWrite(DECK_GPIO_IO4, HIGH);
  pinMode(DECK_GPIO_IO4, INPUT_PULLUP);

  vTaskDelay(M2T(100));

  while (1)
  {
    vTaskDelay(M2T(10));
    DEBUG_PRINT("in the loop\n");
    if (positioningInit)
    {
      // Check if AI deck is properly mounted
      // uint8_t aiInit = paramGetUint(idAIdeck);

      // uart1GetDataWithTimeout(&byte, M2T(10));

      yaw_command = byte;
      //yaw_command = 0;
      yaw_command_derivative = (prev_yaw_command - yaw_command) / 0.2; // approx 5 fps

      yaw_rate  = 1 * yaw_gain_proportional * yaw_command + 1 * yaw_gain_derivative * yaw_command_derivative;
      prev_yaw_command = yaw_command;

      // DEBUG_PRINT("yaw Rate = %f\n", yaw_rate);

      // Get Height estimate
      heightEstimate = logGetFloat(idHeightEstimate);

      // Get the heading and convert it to rad
      float estYawDeg = logGetFloat(idStabilizerYaw);
      float estYawRad = estYawDeg * (float)M_PI / 180.0f;

      cmdVelX = maxForwardSpeed;
      cmdVelY = 0.0f;
      cmdAngWRad = 0.0f;
      cmdAngWDeg = 0.0f;
      cmdHeight = hoverHeight;

      // Only go to the state machine if the crazyflie has reached a certain height
      // if (heightEstimate > hoverHeight - 0.1f)
      // {
        // The wall-following state machine which outputs velocity commands
        float timeNow = usecTimestamp() / 1e6;
        stateInnerLoop = wallFollower(&cmdVelX, &cmdVelY, &cmdAngWRad, &cmdHeight, estYawRad, timeNow, yaw_command);
        cmdAngWDeg = cmdAngWRad * 180.0f / (float)M_PI;
      // }
      // Turn velocity commands into setpoints and send it to the commander
      setVelocitySetpoint(&setpoint, cmdVelX, cmdVelY, cmdHeight, cmdAngWDeg);
      commanderSetSetpoint(&setpoint, 3);
    }
    
  }
}

PARAM_GROUP_START(app)
// PARAM_ADD(PARAM_UINT8, goLeft, &goLeft)
// PARAM_ADD(PARAM_FLOAT, distanceWall, &distanceToWall)
PARAM_ADD(PARAM_FLOAT, yaw_gain_proportional, &yaw_gain_proportional)
PARAM_ADD(PARAM_FLOAT, yaw_gain_derivative, &yaw_gain_derivative)
PARAM_GROUP_STOP(app)

LOG_GROUP_START(app)
LOG_ADD(LOG_INT8, yaw_command, &yaw_command)
LOG_ADD(LOG_FLOAT, yaw_rate, &yaw_rate)
LOG_ADD(LOG_FLOAT, cmdVelX, &cmdVelX)
LOG_ADD(LOG_FLOAT, cmdVelY, &cmdVelY)
LOG_ADD(LOG_FLOAT, cmdHeight, &cmdHeight)
LOG_ADD(LOG_FLOAT, cmdAngWDeg, &cmdAngWDeg)
LOG_ADD(LOG_FLOAT, heightEstimate, &heightEstimate)
LOG_GROUP_STOP(app)

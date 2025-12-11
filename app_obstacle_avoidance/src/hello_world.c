/**
 * Simplified Takeoff and Landing Demo
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

#define DEBUG_MODULE "TAKEOFF_LANDING"

// States
typedef enum {
    IDLE,
    TAKING_OFF,
    HOVERING,
    FORWARD,
    LANDING,
    STOPPED
} FlightState;

static FlightState flightState = IDLE;

// Flight parameters
static float TAKEOFF_HEIGHT = 0.4f;    // 起飞高度
static float LANDING_HEIGHT = 0.1f;    // 降落高度  
static float HOVER_DURATION = 5.0f;    // 悬停时间

static float currentHeight = 0.0f;
static float stateStartTime = 0.0f;

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

void appMain()
{
  DEBUG_PRINT("Takeoff and Landing Demo Started\n");
  vTaskDelay(M2T(2000));
  
  // 请求解锁
  supervisorRequestArming(true);
  vTaskDelay(M2T(2000));

  // 获取高度估计
  logVarId_t idHeightEstimate = logGetVarId("stateEstimate", "z");

  // 初始化setpoint结构
  setpoint_t setpoint;

  flightState = TAKING_OFF;
  stateStartTime = usecTimestamp() / 1e6;

  while (1)
  {
    vTaskDelay(M2T(10));
    
    // 获取当前高度
    currentHeight = logGetFloat(idHeightEstimate);
    float currentTime = usecTimestamp() / 1e6;

    memset(&setpoint, 0, sizeof(setpoint_t));
    
    switch (flightState)
    {
      case IDLE:
        DEBUG_PRINT("IDLE - Ready for takeoff\n");
        setVelocitySetpoint(&setpoint, 0, 0, 0.0f, 0);
        break;
        
      case TAKING_OFF:
        setVelocitySetpoint(&setpoint, 0, 0, TAKEOFF_HEIGHT, 0);
        
        break;
        
      case HOVERING:
        {
          float hoverTime = currentTime - stateStartTime;
          setVelocitySetpoint(&setpoint, 0, 0, TAKEOFF_HEIGHT, 0);
          
          if (hoverTime >= HOVER_DURATION) {
            flightState = FORWARD;
            stateStartTime = currentTime;
            DEBUG_PRINT("Hover complete, starting forwarding\n");
          }
        }
        break;
      case FORWARD:
        {
          float forwardTime = currentTime - stateStartTime;
          setVelocitySetpoint(&setpoint, 0.1f, 0, TAKEOFF_HEIGHT, 0);
          
          if (forwardTime >= HOVER_DURATION) {
            flightState = LANDING;
            stateStartTime = currentTime;
            DEBUG_PRINT("Forward complete, starting landing\n");
          }
        }
        break;        
      case LANDING:
        setVelocitySetpoint(&setpoint, 0, 0, LANDING_HEIGHT, 0);
        
        // 检查是否降落到地面
        if (currentHeight <= LANDING_HEIGHT + 0.05f) {
          flightState = STOPPED;
          DEBUG_PRINT("Landing complete\n");
        }
        break;
        
      case STOPPED:
        // 完全停止，关闭电机
        memset(&setpoint, 0, sizeof(setpoint_t));
        DEBUG_PRINT("STOPPED - Mission complete\n");
        vTaskDelay(M2T(1000)); // 减少打印频率
        break;
    }
    
    // 发送setpoint给commander
    commanderSetSetpoint(&setpoint, 3);
  }
}

// 参数配置
PARAM_GROUP_START(app)
PARAM_ADD(PARAM_FLOAT, takeoffHeight, &TAKEOFF_HEIGHT)
PARAM_ADD(PARAM_FLOAT, landingHeight, &LANDING_HEIGHT)
PARAM_ADD(PARAM_FLOAT, hoverDuration, &HOVER_DURATION)
PARAM_GROUP_STOP(app)

LOG_GROUP_START(app)
LOG_ADD(LOG_FLOAT, currentHeight, &currentHeight)
LOG_ADD(LOG_UINT8, flightState, &flightState)
LOG_GROUP_STOP(app)

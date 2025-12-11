/*
 * wallfollowing_multirange_onboard.h
 *
 *  Created on: Aug 7, 2018
 *      Author: knmcguire
 */

#ifndef SRC_WALLFOLLOWING_MULTIRANGER_ONBOARD_H_
#define SRC_WALLFOLLOWING_MULTIRANGER_ONBOARD_H_
#include <stdint.h>
#include <stdbool.h>


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



StateWF wallFollower(float *cmdVelX, float *cmdVelY, float *cmdAngW, float *cmdHeight, float currentHeading, float timeOuter, double yawRate);

void adjustDistanceWall(float distanceWallNew);

void wallFollowerInit(float maxForwardSpeed_ref, StateWF initState);
#endif /* SRC_WALLFOLLOWING_MULTIRANGER_ONBOARD_H_ */

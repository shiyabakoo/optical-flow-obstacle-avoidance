#!/bin/bash

# Source the GAP SDK environment
source ~/Desktop/Greenwaves/gap_riscv_toolchain_ubuntu/gap_sdk/sourceme.sh <<< "3"

# Source the AI-deck configuration
source ~/Desktop/Greenwaves/gap_riscv_toolchain_ubuntu/gap_sdk/configs/ai_deck.sh

# Export the OpenOCD cable configuration
export GAPY_OPENOCD_CABLE=~/Desktop/Greenwaves/gap8_openocd/tcl/interface/ftdi/olimex-arm-usb-tiny-h.cfg

# Build and run the project
make all run PMSIS_OS=pulpos platform=board


include parameters.mk
include app_path.mk
HB_HAMMERBENCH_PATH:=$(shell git rev-parse --show-toplevel)
override BSG_MACHINE_PATH = $(REPLICANT_PATH)/machines/pod_X1Y1_ruche_X16Y8_hbm_one_pseudo_channel 
include $(HB_HAMMERBENCH_PATH)/mk/environment.mk

# Tile Group Dimensions
TILE_GROUP_DIM_X ?= 16
TILE_GROUP_DIM_Y ?= 8

vpath %.c   $(APP_PATH)
vpath %.cpp $(APP_PATH)

# Source files
TEST_SOURCES = main.c

# Get SIZE from vector-size parameter
SIZE = $(vector-size)

# Compiler flags
DEFINES += -D_XOPEN_SOURCE=500 -D_BSD_SOURCE -D_DEFAULT_SOURCE
DEFINES += -DSIZE=$(SIZE)
CDEFINES += -Dbsg_tiles_X=$(TILE_GROUP_DIM_X) -Dbsg_tiles_Y=$(TILE_GROUP_DIM_Y)
CXXDEFINES += 

# Optimization flags
FLAGS     = -g -O3 -Wall -Wno-unused-function -Wno-unused-variable
CFLAGS   += -std=c99 $(FLAGS)
CXXFLAGS += -std=c++11 $(FLAGS)

include $(EXAMPLES_PATH)/compilation.mk

# Linker flags
LDFLAGS +=
include $(EXAMPLES_PATH)/link.mk

# RISC-V compilation
RISCV_CCPPFLAGS += -O3 -std=c++14
RISCV_CCPPFLAGS += -Dbsg_tiles_X=$(TILE_GROUP_DIM_X)
RISCV_CCPPFLAGS += -Dbsg_tiles_Y=$(TILE_GROUP_DIM_Y)
RISCV_CCPPFLAGS += -DCACHE_LINE_WORDS=$(BSG_MACHINE_VCACHE_LINE_WORDS)
RISCV_LDFLAGS += -flto
RISCV_TARGET_OBJECTS = kernel.rvo
BSG_MANYCORE_KERNELS = main.riscv

include $(EXAMPLES_PATH)/cuda/riscv.mk

# Execution parameters
C_ARGS ?= $(BSG_MANYCORE_KERNELS) hybrid_sort
SIM_ARGS ?=

include $(EXAMPLES_PATH)/execution.mk

# Regression testing
regression: exec.log
	@grep "BSG REGRESSION TEST .*PASSED.*" $< > /dev/null

.DEFAULT_GOAL := help
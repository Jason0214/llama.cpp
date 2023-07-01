#!/bin/bash

# Warmup
# ./bin/benchmark -i 10 1>/dev/null 2>/dev/null

export GGML_VK_TH_GROUP_SIZE_X=4 GGML_VK_TH_GROUP_SIZE_Y=4 GGML_VK_TH_TILE_BYTES_X=32 GGML_VK_TH_TILE_BYTES_Y=32; ./bin/benchmark -i 3 2>/dev/null | grep "Average" | awk '{print "th_sz '$GGML_VK_TH_GROUP_SIZE_X'x'$GGML_VK_TH_GROUP_SIZE_Y', tile_sz '$GGML_VK_TH_TILE_BYTES_X'x'$GGML_VK_TH_TILE_BYTES_Y' gFlops: " $2}'
export GGML_VK_TH_GROUP_SIZE_X=8 GGML_VK_TH_GROUP_SIZE_Y=8 GGML_VK_TH_TILE_BYTES_X=32 GGML_VK_TH_TILE_BYTES_Y=32; ./bin/benchmark -i 3 2>/dev/null | grep "Average" | awk '{print "th_sz '$GGML_VK_TH_GROUP_SIZE_X'x'$GGML_VK_TH_GROUP_SIZE_Y', tile_sz '$GGML_VK_TH_TILE_BYTES_X'x'$GGML_VK_TH_TILE_BYTES_Y' gFlops: " $2}'
export GGML_VK_TH_GROUP_SIZE_X=16 GGML_VK_TH_GROUP_SIZE_Y=16 GGML_VK_TH_TILE_BYTES_X=32 GGML_VK_TH_TILE_BYTES_Y=32; ./bin/benchmark -i 3 2>/dev/null | grep "Average" | awk '{print "th_sz '$GGML_VK_TH_GROUP_SIZE_X'x'$GGML_VK_TH_GROUP_SIZE_Y', tile_sz '$GGML_VK_TH_TILE_BYTES_X'x'$GGML_VK_TH_TILE_BYTES_Y' gFlops: " $2}'
export GGML_VK_TH_GROUP_SIZE_X=8 GGML_VK_TH_GROUP_SIZE_Y=8 GGML_VK_TH_TILE_BYTES_X=16 GGML_VK_TH_TILE_BYTES_Y=16; ./bin/benchmark -i 3 2>/dev/null | grep "Average" | awk '{print "th_sz '$GGML_VK_TH_GROUP_SIZE_X'x'$GGML_VK_TH_GROUP_SIZE_Y', tile_sz '$GGML_VK_TH_TILE_BYTES_X'x'$GGML_VK_TH_TILE_BYTES_Y' gFlops: " $2}'
export GGML_VK_TH_GROUP_SIZE_X=8 GGML_VK_TH_GROUP_SIZE_Y=8 GGML_VK_TH_TILE_BYTES_X=8 GGML_VK_TH_TILE_BYTES_Y=128; ./bin/benchmark -i 3 2>/dev/null | grep "Average" | awk '{print "th_sz '$GGML_VK_TH_GROUP_SIZE_X'x'$GGML_VK_TH_GROUP_SIZE_Y', tile_sz '$GGML_VK_TH_TILE_BYTES_X'x'$GGML_VK_TH_TILE_BYTES_Y' gFlops: " $2}'

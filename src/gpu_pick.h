#pragma once
#include <vector>

bool gpu_pick_block_mse(
    const float* h_srcLum, int srcW, int srcH,
    const float* h_dstLum, int blockW, int blockH,
    int overlapW, int overlapH,
    int dstX, int dstY,
    float tolerance,
    unsigned int rngSeed,
    int& outX, int& outY,
    float& outMinMSE
);


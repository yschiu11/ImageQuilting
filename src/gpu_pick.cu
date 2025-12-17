#include "gpu_pick.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cmath>
#include <limits>
#include <vector>
#include <cstdio>

static bool cuda_ok(cudaError_t e, const char* msg)
{
    if(e != cudaSuccess)
    {
        std::fprintf(stderr, "[CUDA] %s: %s\n", msg, cudaGetErrorString(e));
        return false;
    }
    return true;
}

__device__ __forceinline__ float sqr(float x) { return x * x; }

__global__ void kernel_mse_overlap(
    const float* __restrict__ srcLum, int srcW, int srcH,
    const float* __restrict__ dstLum,
    int blockW, int blockH,
    int overlapW, int overlapH,
    int xLen, int yLen,
    int dstX, int dstY,
    float* __restrict__ outMSE
)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = xLen * yLen;
    if(tid >= total) return;

    int srcX = tid % xLen;
    int srcY = tid / xLen;

    float sse = 0.0f;
    int pixelCnt = 0;

    if(dstX <= 0 && dstY <= 0)
    {
        outMSE[tid] = 0.0f;
        return;
    }

    if(dstX > 0 && dstY <= 0)
    {
        pixelCnt = overlapW * blockH;
        for(int iy = 0; iy < blockH; ++iy)
        {
            int srcRow = (srcY + iy) * srcW;
            int dstRow = iy * blockW;
            for(int ix = 0; ix < overlapW; ++ix)
            {
                float d = srcLum[srcRow + (srcX + ix)] - dstLum[dstRow + ix];
                sse += sqr(d);
            }
        }
        outMSE[tid] = sse / (float)pixelCnt;
        return;
    }

    if(dstX <= 0 && dstY > 0)
    {
        pixelCnt = overlapH * blockW;
        for(int iy = 0; iy < overlapH; ++iy)
        {
            int srcRow = (srcY + iy) * srcW;
            int dstRow = iy * blockW;
            for(int ix = 0; ix < blockW; ++ix)
            {
                float d = srcLum[srcRow + (srcX + ix)] - dstLum[dstRow + ix];
                sse += sqr(d);
            }
        }
        outMSE[tid] = sse / (float)pixelCnt;
        return;
    }

    // A: overlapW*overlapH
    // B: overlapW*(blockH-overlapH)
    // C: overlapH*(blockW-overlapW)
    pixelCnt = overlapW * overlapH
             + overlapW * (blockH - overlapH)
             + overlapH * (blockW - overlapW);

    // A
    for(int iy = 0; iy < overlapH; ++iy)
    {
        int srcRow = (srcY + iy) * srcW;
        int dstRow = iy * blockW;
        for(int ix = 0; ix < overlapW; ++ix)
        {
            float d = srcLum[srcRow + (srcX + ix)] - dstLum[dstRow + ix];
            sse += sqr(d);
        }
    }
    // B
    for(int iy = overlapH; iy < blockH; ++iy)
    {
        int srcRow = (srcY + iy) * srcW;
        int dstRow = iy * blockW;
        for(int ix = 0; ix < overlapW; ++ix)
        {
            float d = srcLum[srcRow + (srcX + ix)] - dstLum[dstRow + ix];
            sse += sqr(d);
        }
    }
    // C
    for(int iy = 0; iy < overlapH; ++iy)
    {
        int srcRow = (srcY + iy) * srcW;
        int dstRow = iy * blockW;
        for(int ix = overlapW; ix < blockW; ++ix)
        {
            float d = srcLum[srcRow + (srcX + ix)] - dstLum[dstRow + ix];
            sse += sqr(d);
        }
    }

    outMSE[tid] = sse / (float)pixelCnt;
}

bool gpu_pick_block_mse(
    const float* h_srcLum, int srcW, int srcH,
    const float* h_dstLum, int blockW, int blockH,
    int overlapW, int overlapH,
    int dstX, int dstY,
    float tolerance,
    unsigned int rngSeed,
    int& outX, int& outY,
    float& outMinMSE
)
{
    int xLen = srcW - blockW;
    int yLen = srcH - blockH;
    if(xLen <= 0 || yLen <= 0) return false;

    int total = xLen * yLen;

    float *d_src = nullptr, *d_dst = nullptr, *d_mse = nullptr;
    size_t srcBytes = (size_t)srcW * (size_t)srcH * sizeof(float);
    size_t dstBytes = (size_t)blockW * (size_t)blockH * sizeof(float);
    size_t mseBytes = (size_t)total * sizeof(float);

    if(!cuda_ok(cudaMalloc(&d_src, srcBytes), "cudaMalloc d_src")) return false;
    if(!cuda_ok(cudaMalloc(&d_dst, dstBytes), "cudaMalloc d_dst")) { cudaFree(d_src); return false; }
    if(!cuda_ok(cudaMalloc(&d_mse, mseBytes), "cudaMalloc d_mse")) { cudaFree(d_src); cudaFree(d_dst); return false; }

    bool ok = true;

    ok = ok && cuda_ok(cudaMemcpy(d_src, h_srcLum, srcBytes, cudaMemcpyHostToDevice), "Memcpy srcLum H2D");
    ok = ok && cuda_ok(cudaMemcpy(d_dst, h_dstLum, dstBytes, cudaMemcpyHostToDevice), "Memcpy dstLum H2D");
    if(!ok)
    {
        cudaFree(d_src); cudaFree(d_dst); cudaFree(d_mse);
        return false;
    }

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    kernel_mse_overlap<<<blocks, threads>>>(
        d_src, srcW, srcH,
        d_dst,
        blockW, blockH,
        overlapW, overlapH,
        xLen, yLen,
        dstX, dstY,
        d_mse
    );

    ok = ok && cuda_ok(cudaGetLastError(), "kernel launch");
    ok = ok && cuda_ok(cudaDeviceSynchronize(), "kernel sync");
    if(!ok)
    {
        cudaFree(d_src); cudaFree(d_dst); cudaFree(d_mse);
        return false;
    }

    thrust::device_ptr<float> p(d_mse);
    auto it = thrust::min_element(p, p + total);
    int minIdx = (int)(it - p);

    float minMSE = 0.0f;
    ok = ok && cuda_ok(cudaMemcpy(&minMSE, d_mse + minIdx, sizeof(float), cudaMemcpyDeviceToHost), "Memcpy minMSE D2H");

    std::vector<float> h_mse(total);
    ok = ok && cuda_ok(cudaMemcpy(h_mse.data(), d_mse, mseBytes, cudaMemcpyDeviceToHost), "Memcpy mse array D2H");

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_mse);
    if(!ok) return false;

    float limit = minMSE * (1.0f + tolerance);
    const float eps = 0.001f;

    std::vector<int> allowed;
    allowed.reserve(4096);

    if(dstX == 0 && dstY == 0)
    {
        allowed.resize(total);
        for(int i = 0; i < total; ++i) allowed[i] = i;
    }
    else
    {
        for(int i = 0; i < total; ++i)
        {
            float mse = h_mse[i];
            if(mse > eps && mse <= limit)
                allowed.push_back(i);
        }
        if(allowed.empty()) allowed.push_back(minIdx);
    }

    unsigned int x = rngSeed ? rngSeed : 1234567u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    int pick = allowed[(int)(x % (unsigned int)allowed.size())];

    outX = pick % xLen;
    outY = pick / xLen;
    outMinMSE = minMSE;
    return true;
}


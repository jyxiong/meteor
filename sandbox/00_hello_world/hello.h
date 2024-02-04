#pragma once

#include <cuda_runtime.h>

struct Params
{
    uchar4* image;
    unsigned int width;
};

struct RayGenData
{
    float3 color;
};

#include <optix.h>

#include "hello.h"

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__() {
  uint3 launch_index = optixGetLaunchIndex();
  RayGenData *rtData = (RayGenData *)optixGetSbtDataPointer();
  params.image[launch_index.y * params.width + launch_index.x] =
      make_uchar4(rtData->color.x * 255, rtData->color.y * 255,
                  rtData->color.z * 255, 255);
}

extern "C" __global__ void __miss__() {

}

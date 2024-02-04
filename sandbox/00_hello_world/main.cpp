#include "hello.h"
#include <otkHelloKernelCuda.h>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stack_size.h>

#include "stb_image_write.h"

#include "meteor/core/macro.h"
#include "meteor/cuda/buffer.h"
#include "meteor/optix/context.h"
#include "meteor/optix/module.h"
#include "meteor/optix/pipeline.h"
#include "meteor/optix/program.h"
#include "meteor/optix/sbt.h"

using namespace meteor;

template <typename T> struct SbtRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<int> MissSbtRecord;

void test_otk(CUDABuffer<uchar4> d_image, size_t width) {

  // Initialize CUDA and create OptiX context
  //
  OptixDeviceContext context = nullptr;
  {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
  }

  //
  // Create module
  //
  OptixModule module = nullptr;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  {
    OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_NONE; // TODO: should be
                                   // OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreate(
        context, &module_compile_options, &pipeline_compile_options,
        helloCudaText(), helloCudaSize, log, &sizeof_log, &module));
  }

  //
  // Create program groups, including NULL miss and hitgroups
  //
  OptixProgramGroup raygen_prog_group = nullptr;
  OptixProgramGroup miss_prog_group = nullptr;
  {
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log,
                                            &sizeof_log, &raygen_prog_group));

    // Leave miss group's module and entryfunc name null
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    OPTIX_CHECK(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                        1, // num program groups
                                        &program_group_options, log,
                                        &sizeof_log, &miss_prog_group));
  }

  //
  // Link pipeline
  //
  OptixPipeline pipeline = nullptr;
  {
    char log[2048];
    size_t sizeof_log = sizeof(log);

    const uint32_t max_trace_depth = 0;
    OptixProgramGroup program_groups[] = {raygen_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    OPTIX_CHECK_LOG(optixPipelineCreate(
        context, &pipeline_compile_options, &pipeline_link_options,
        program_groups, sizeof(program_groups) / sizeof(program_groups[0]), log,
        &sizeof_log, &pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
#if OPTIX_VERSION < 70700
      CUDA_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
#else
      OPTIX_CHECK(
          optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
#endif
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, max_trace_depth,
        0, // maxCCDepth
        0, // maxDCDEpth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        2 // maxTraversableDepth
        ));
  }

  //
  // Set up shader binding table
  //
  OptixShaderBindingTable sbt = {};
  {
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record),
                          raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    rg_sbt.data = {0.462f, 0.725f, 0.f};
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record), &rg_sbt,
                          raygen_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    RayGenSbtRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt,
                          miss_record_size, cudaMemcpyHostToDevice));

    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
  }

  //
  // launch
  //
  {
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    Params params;
    params.image = d_image.deviceData();
    params.width = width;

    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_param), &params,
                          sizeof(params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt,
                            width, width, /*depth=*/1));

    CUDA_SYNC_CHECK();
  }
}

void test(CUDABuffer<uchar4> &d_image, size_t width) {
  Context context;

  CUDA_CHECK(cudaFree(0));
  OPTIX_CHECK(optixInit());
  context.create();

  Pipeline pipeline;
  pipeline.setLaunchVariableName("params");
  pipeline.setDirectCallableDepth(5);
  pipeline.setContinuationCallableDepth(5);
  pipeline.setNumPayloads(5);
  pipeline.setNumAttributes(5);

  Module module = pipeline.createModuleFromPtxSource(context, helloCudaText());

  Params params;
  params.image = d_image.deviceData();
  params.width = width;

  CUDABuffer<Params> d_params;
  d_params.allocate(1);
  d_params.copyToDevice(&params, 1);

  ProgramGroup raygen_prg =
      pipeline.createRaygenProgram(context, module, "__raygen__");

  ProgramGroup miss_prg =
      pipeline.createMissProgram(context, module, "__miss__");

  pipeline.create(context);

  ShaderBindingTable<Record<RayGenData>, Record<int>, emptyRecord, emptyRecord,
                     emptyRecord, 1>
      sbt;
  Record<RayGenData> raygen_record;
  raygen_prg.recordPackHeader(&raygen_record);
  raygen_record.data.color = make_float3(0.462f, 0.725f, 0.0f);
  sbt.setRaygenRecord(raygen_record);

  Record<int> miss_record;
  miss_prg.recordPackHeader(&miss_record);
  sbt.setMissRecord({miss_record});

  sbt.createOnDevice();

  CUstream stream;
  cudaStreamCreate(&stream);

  OPTIX_CHECK(optixLaunch(static_cast<OptixPipeline>(pipeline), stream,
                          d_params.devicePtr(), sizeof(Params), &sbt.sbt(),
                          params.width, params.width, 1));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_SYNC_CHECK();
}

int main() {

  size_t width = 512;
  CUDABuffer<uchar4> d_image;
  d_image.allocate(width * width * sizeof(uchar4));

  test_otk(d_image, width);

  std::vector<uchar4> h_image(width * width);
  cudaMemcpy(h_image.data(), d_image.deviceData(),
             width * width * sizeof(uchar4), cudaMemcpyDeviceToHost);

  stbi_write_png("output.png", width, width, 4, h_image.data(), width * 4);

  return 0;
}
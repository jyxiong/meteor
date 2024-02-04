#include "meteor/optix/module.h"

#include <map>

#include "meteor/core/macro.h"

namespace meteor {

// ------------------------------------------------------------------
Module::Module() {
  m_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  m_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#if OPTIX_VERSION < 70400
  m_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
  m_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif
}

Module::Module(const OptixModuleCompileOptions &options) : m_options(options) {}

void Module::createFromPtxSource(const Context &ctx, const std::string &source,
                                 OptixPipelineCompileOptions pipeline_options) {
  char log[2048];
  size_t sizeof_log = sizeof(log);

#if OPTIX_VERSION < 70700
  OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
      static_cast<OptixDeviceContext>(ctx), &m_options, &pipeline_options,
      source.c_str(), source.size(), log, &sizeof_log, &m_module));
#else
  OPTIX_CHECK_LOG(optixModuleCreate(
      static_cast<OptixDeviceContext>(ctx), &m_options, &pipeline_options,
      source.c_str(), source.size(), log, &sizeof_log, &m_module));
#endif
}

void Module::destroy() { OPTIX_CHECK(optixModuleDestroy(m_module)); }

// ------------------------------------------------------------------
void Module::setOptimizationLevel(OptixCompileOptimizationLevel optLevel) {
  m_options.optLevel = optLevel;
}

void Module::setDebugLevel(OptixCompileDebugLevel debugLevel) {
  m_options.debugLevel = debugLevel;
}

// ------------------------------------------------------------------
void Module::setBoundValues(size_t offset_in_bytes, size_t size_in_bytes,
                            void *bound_value_ptr, const char *annotation) {
  OptixModuleCompileBoundValueEntry *bound_values =
      new OptixModuleCompileBoundValueEntry();
  bound_values->pipelineParamOffsetInBytes = offset_in_bytes;
  bound_values->sizeInBytes = size_in_bytes;
  bound_values->boundValuePtr = bound_value_ptr;
  bound_values->annotation = annotation;
  m_options.boundValues = bound_values;
}

void Module::setBoundValues(OptixModuleCompileBoundValueEntry *bound_values) {
  m_options.boundValues = bound_values;
}

void Module::setNumBounds(unsigned int num_bound) {
  m_options.numBoundValues = num_bound;
}

const OptixModuleCompileOptions &Module::compileOptions() const {
  return m_options;
}

OptixModuleCompileOptions &Module::compileOptions() { return m_options; }

} // namespace meteor
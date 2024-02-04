#pragma once

#include <memory>
#include <vector>

#include <optix.h>

#include "meteor/core/shape.h"
#include "meteor/optix/context.h"

namespace meteor {

/// @todo: Support Opacity micro map

class GeometryAccel {
public:
  GeometryAccel() = default;
  GeometryAccel(ShapeType shape_type);
  ~GeometryAccel();

  void addShape(const std::shared_ptr<Shape> &shape);
  std::vector<std::shared_ptr<Shape>> shapes() const;

  void build(const Context &ctx, CUstream stream);
  void update(const Context &ctx, CUstream stream);
  // void relocate() // TODO
  void free();

  void setFlags(const uint32_t build_flags);

  void allowUpdate();
  void allowCompaction();
  void preferFastTrace();
  void preferFastBuild();
  void allowRandomVertexAccess();

  void disableUpdate();
  void disableCompaction();
  void disableFastTrace();
  void disableFastBuild();
  void disableRandomVertexAccess();

  void setMotionOptions(const OptixMotionOptions &motion_options);

  uint32_t count() const;
  OptixTraversableHandle handle() const;
  CUdeviceptr deviceBuffer() const;
  size_t deviceBufferSize() const;
  bool isBuilded() const;

private:
  ShapeType m_shape_type;
  OptixTraversableHandle m_handle{0};
  OptixAccelBuildOptions m_options{};
  uint32_t m_count{0};

  std::vector<std::shared_ptr<Shape>> m_shapes;
  std::vector<OptixBuildInput> m_build_inputs;

  CUdeviceptr d_buffer{0};
  size_t d_buffer_size{0};
};

} // namespace meteor
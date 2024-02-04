#pragma once

#include <optix.h>
#include <optix_stubs.h>

namespace meteor {

class Context {
public:
  Context();
  explicit Context(const OptixDeviceContextOptions &options);
  explicit Context(unsigned int device_id);
  explicit Context(unsigned int device_id,
                   const OptixDeviceContextOptions &options);

  explicit operator OptixDeviceContext() const { return m_ctx; }
  explicit operator OptixDeviceContext &() { return m_ctx; }

  void create();
  void destroy();

  // Setter for context options
  void setOptions(const OptixDeviceContextOptions &options);
  void setLogCallbackFunction(OptixLogCallback callback_func);
  void setLogCallbackData(void *callback_data);
  void setLogCallbackLevel(int callback_level);
  void enableValidation();
  void disableValidation();
  OptixDeviceContextOptions options() const;

  void setDeviceId(const unsigned int device_id);
  unsigned int deviceId() const;

private:
  unsigned int m_device_id{0};
  OptixDeviceContext m_ctx{nullptr};
  OptixDeviceContextOptions m_options;
};

} // namespace meteor
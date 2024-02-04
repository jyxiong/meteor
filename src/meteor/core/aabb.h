#pragma once

#include <iostream>

#include <optix.h>

#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"

namespace meteor {

class AABB {
public:
  AABB() : m_min(glm::vec3(0.f)), m_max(glm::vec3(0.f)) {}
  AABB(glm::vec3 min, glm::vec3 max) : m_min(min), m_max(max) {}
  const glm::vec3 &min() const { return m_min; }
  const glm::vec3 &max() const { return m_max; }

  explicit operator OptixAabb() {
    return {m_min[0], m_min[1], m_min[2], m_max[0], m_max[1], m_max[2]};
  }

  float surfaceArea() {
    float dx = m_max[0] - m_min[0];
    float dy = m_max[1] - m_max[1];
    float dz = m_max[2] - m_max[2];
    return 2 * (dx * dy + dy * dz + dz * dx);
  }

  static AABB merge(AABB box0, AABB box1) {
    glm::vec3 min_box = glm::vec3(fmin(box0.min()[0], box1.min()[0]),
                          fmin(box0.min()[1], box1.min()[1]),
                          fmin(box0.min()[2], box1.min()[2]));

    glm::vec3 max_box = glm::vec3(fmax(box0.max()[0], box1.max()[0]),
                          fmax(box0.max()[1], box1.max()[1]),
                          fmax(box0.max()[2], box1.max()[2]));

    return AABB(min_box, max_box);
  }

private:
  glm::vec3 m_min, m_max;
};

inline std::ostream &operator<<(std::ostream &out, const AABB &aabb) {
  out << "min: " << glm::to_string(aabb.min()) << ", max: " << glm::to_string(aabb.max());
  return out;
}

} // namespace meteor
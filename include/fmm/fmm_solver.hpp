#pragma once
#include <vector>
#include "fmm/direct_sum.hpp"

namespace fmm {

// Full Fast Multipole Method evaluation of phi(x_i) = sum_{j!=i} q_j/|x_i-x_j|
// for all particles, following the uniform (non-adaptive) algorithm of
// GR RR-1115, Section 6: a hierarchy of 8^l boxes over the bounding cube,
// upward pass (P2M at finest level, M2M up), downward pass (L2L from parent
// plus M2L from each box's interaction list), then L2P plus direct P2P over
// the 27 near neighbors at the finest level.
//
// `levels` is the finest refinement level (>= 2; levels 0 and 1 have no
// well-separated boxes). `order` is the expansion order p; worst-case
// truncation error decays like (sqrt(3)/3)^p (their Section 4).
std::vector<double> fmm_evaluate(const std::vector<Particle>& particles,
                                  int order, int levels);

}  // namespace fmm
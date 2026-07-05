#pragma once
#include <array>
#include <vector>

namespace fmm {

// A point mass/charge in 3D. `q` is mass (gravity) or charge (Coulomb) --
// the 1/r kernel is the same either way, only the physical interpretation
// of q and phi differs.
struct Particle {
    double x, y, z;
    double q;
};

// Direct O(N^2) evaluation of phi(x_i) = sum_{j != i} q_j / |x_i - x_j|
// for every particle i. This is the ground truth that every FMM stage is
// validated against -- correct by construction, since it's just the
// definition of the potential with no approximation.
std::vector<double> direct_sum(const std::vector<Particle>& particles);

// Same potential, but evaluated at a set of arbitrary target points rather
// than at the particles themselves (targets need not coincide with sources).
// Useful for validating multipole expansions at points far from a node,
// where "far from a node" doesn't necessarily mean "at another particle".
std::vector<double> direct_sum_at_targets(
    const std::vector<Particle>& sources,
    const std::vector<std::array<double, 3>>& targets);

}  // namespace fmm
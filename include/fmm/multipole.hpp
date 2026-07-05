#pragma once
#include <complex>
#include <vector>
#include "fmm/direct_sum.hpp"
#include "fmm/octree.hpp"
#include "fmm/spherical_harmonics.hpp"

namespace fmm {

// Multipole expansion in the Greengard-Rokhlin convention (RR-1115,
// "A New Version of the Fast Multipole Method for the Laplace Equation in
// Three Dimensions", 1996), Theorem 3.2:
//
//   Phi(P) = sum_{n=0}^{p} sum_{m=-n}^{n}  M_n^m / r^{n+1} * Y_n^m(theta, phi)
//   M_n^m  = sum_i q_i * rho_i^n * Y_n^{-m}(alpha_i, beta_i)
//
// where Y_n^m is the GR-normalized harmonic (spherical_harmonic_gr, their
// eq. 34), (r, theta, phi) are the target's spherical coordinates relative
// to the expansion center, and (rho_i, alpha_i, beta_i) each source's.
// Coefficients are stored flat via sh_index(n, m).
struct MultipoleExpansion {
    int order;
    double center_x, center_y, center_z;
    std::vector<std::complex<double>> M;  // size (order+1)^2

    double evaluate(double x, double y, double z) const;
};

// P2M (GR Theorem 3.2, eq. 38): build a multipole expansion of the given
// order about (center_x, center_y, center_z) from the listed particles.
MultipoleExpansion particles_to_multipole(
    const std::vector<Particle>& particles,
    const std::vector<std::size_t>& particle_indices,
    double center_x, double center_y, double center_z,
    int order);

// M2M (GR Theorem 5.1, eq. 45): translate a multipole expansion centered
// at C_old into an equivalent expansion of the same order centered at
// (new_cx, new_cy, new_cz). With (rho, alpha, beta) the spherical
// coordinates of C_old relative to the new center:
//
//   M_j^k = sum_{n=0}^{j} sum_{m=-n}^{n}, |k-m| <= j-n :
//       O_{j-n}^{k-m} * i^{|k|-|m|-|k-m|} * A_n^m * A_{j-n}^{k-m}
//           * rho^n * Y_n^{-m}(alpha, beta)  /  A_j^k
//
// with A_n^m = (-1)^n / sqrt((n-m)! (n+m)!) (their eq. 33, signed m).
// Per the paper, this translation is EXACT for truncated expansions --
// no additional truncation error is introduced by shifting the center.
// Validated in tests both at the coefficient level (translated
// coefficients match a direct P2M about the new center to machine
// precision) and at the evaluation level (against direct summation).
MultipoleExpansion multipole_to_multipole(
    const MultipoleExpansion& child,
    double new_cx, double new_cy, double new_cz);

// Upward pass (GR Section 6, Steps 1-2): P2M at leaves, then true M2M
// coefficient translation merging children's expansions into parents,
// level by level up the tree. Each node's expansion reuses its children's
// coefficients rather than reprocessing the underlying particles.
struct NodeExpansion {
    const OctreeNode* node;
    MultipoleExpansion expansion;
};

std::vector<NodeExpansion> build_upward_pass(const Octree& tree, int order);

}  // namespace fmm
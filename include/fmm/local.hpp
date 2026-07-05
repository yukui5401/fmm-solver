#pragma once
#include <complex>
#include <vector>
#include "fmm/multipole.hpp"

namespace fmm {

// Local expansion in the Greengard-Rokhlin convention (RR-1115, eq. 47):
//
//   Phi(P) = sum_{j=0}^{p} sum_{k=-j}^{j}  L_j^k * Y_j^k(theta, phi) * r^j
//
// valid inside a sphere around the expansion center, describing the field
// due to sources OUTSIDE that sphere. (r, theta, phi) are the evaluation
// point's spherical coordinates relative to the center. Coefficients are
// stored flat via sh_index(j, k).
struct LocalExpansion {
    int order;
    double center_x, center_y, center_z;
    std::vector<std::complex<double>> L;  // size (order+1)^2

    double evaluate(double x, double y, double z) const;  // L2P
};

// M2L (GR Theorem 5.2, eq. 48, truncated at the expansion order): convert a
// multipole expansion into a local expansion about a well-separated center.
// Validity requires the separation condition of the theorem: with the
// sources inside a sphere of radius a about the multipole center, the
// distance rho between centers must satisfy rho > (c+1)a with c > 1, and
// the local expansion then converges inside radius a about its center.
//
// Unlike M2M and L2L, this operator DOES introduce truncation error (their
// eq. 49, worst case ~ (1/c)^{p+1}); the test suite verifies the error
// decreases with increasing order.
LocalExpansion multipole_to_local(
    const MultipoleExpansion& multipole,
    double local_cx, double local_cy, double local_cz);

// L2L (GR Theorem 5.3, eq. 52): shift a local expansion's center. This is
// a finite sum (n = j..p) and, like M2M, EXACT for truncated expansions --
// verified in tests to machine precision.
LocalExpansion local_to_local(
    const LocalExpansion& parent,
    double new_cx, double new_cy, double new_cz);

}  // namespace fmm
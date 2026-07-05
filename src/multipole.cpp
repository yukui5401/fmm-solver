#include "fmm/multipole.hpp"
#include <cmath>
#include <unordered_map>

namespace fmm {

namespace {

struct SphericalCoords {
    double rho, theta, phi;
};

SphericalCoords to_spherical(double x, double y, double z,
                              double cx, double cy, double cz) {
    double dx = x - cx, dy = y - cy, dz = z - cz;
    double rho = std::sqrt(dx * dx + dy * dy + dz * dz);
    double theta = (rho > 1e-15) ? std::acos(dz / rho) : 0.0;
    double phi = std::atan2(dy, dx);
    return {rho, theta, phi};
}

double factorial(int n) {
    double result = 1.0;
    for (int k = 2; k <= n; ++k) result *= static_cast<double>(k);
    return result;
}

// A_n^m = (-1)^n / sqrt((n-m)! (n+m)!)   (GR eq. 33, signed m)
double A_coeff(int n, int m) {
    double sign = (n % 2 == 0) ? 1.0 : -1.0;
    return sign / std::sqrt(factorial(n - m) * factorial(n + m));
}

// i^k for possibly-negative integer k.
std::complex<double> ipow(int k) {
    switch (((k % 4) + 4) % 4) {
        case 0: return {1.0, 0.0};
        case 1: return {0.0, 1.0};
        case 2: return {-1.0, 0.0};
        default: return {0.0, -1.0};
    }
}

}  // namespace

MultipoleExpansion particles_to_multipole(
    const std::vector<Particle>& particles,
    const std::vector<std::size_t>& particle_indices,
    double center_x, double center_y, double center_z,
    int order) {
    MultipoleExpansion expansion;
    expansion.order = order;
    expansion.center_x = center_x;
    expansion.center_y = center_y;
    expansion.center_z = center_z;
    expansion.M.assign((order + 1) * (order + 1), std::complex<double>(0.0, 0.0));

    for (std::size_t idx : particle_indices) {
        const auto& p = particles[idx];
        auto sc = to_spherical(p.x, p.y, p.z, center_x, center_y, center_z);
        auto Y = spherical_harmonics_gr_all(order, sc.theta, sc.phi);

        double rho_pow_n = 1.0;
        for (int n = 0; n <= order; ++n) {
            for (int m = -n; m <= n; ++m) {
                // M_n^m += q * rho^n * Y_n^{-m}(alpha, beta)   (GR eq. 38)
                expansion.M[sh_index(n, m)] += p.q * rho_pow_n * Y[sh_index(n, -m)];
            }
            rho_pow_n *= sc.rho;
        }
    }
    return expansion;
}

double MultipoleExpansion::evaluate(double x, double y, double z) const {
    auto sc = to_spherical(x, y, z, center_x, center_y, center_z);
    auto Y = spherical_harmonics_gr_all(order, sc.theta, sc.phi);

    std::complex<double> total(0.0, 0.0);
    double r_pow_np1 = sc.rho;  // r^{n+1}, starting at r^1 for n = 0
    for (int n = 0; n <= order; ++n) {
        for (int m = -n; m <= n; ++m) {
            // Phi += M_n^m / r^{n+1} * Y_n^m(theta, phi)   (GR eq. 37)
            total += M[sh_index(n, m)] / r_pow_np1 * Y[sh_index(n, m)];
        }
        r_pow_np1 *= sc.rho;
    }
    return total.real();
}

MultipoleExpansion multipole_to_multipole(
    const MultipoleExpansion& child,
    double new_cx, double new_cy, double new_cz) {
    const int p = child.order;

    MultipoleExpansion out;
    out.order = p;
    out.center_x = new_cx;
    out.center_y = new_cy;
    out.center_z = new_cz;
    out.M.assign((p + 1) * (p + 1), std::complex<double>(0.0, 0.0));

    // (rho, alpha, beta): old center's spherical coords relative to the new
    // center (the vector Q in GR Theorem 5.1).
    auto sc = to_spherical(child.center_x, child.center_y, child.center_z,
                            new_cx, new_cy, new_cz);

    if (sc.rho < 1e-15) {
        out.M = child.M;  // zero shift: expansion unchanged
        return out;
    }

    auto Y = spherical_harmonics_gr_all(p, sc.theta, sc.phi);

    // Precompute rho^n.
    std::vector<double> rho_pow(p + 1);
    rho_pow[0] = 1.0;
    for (int n = 1; n <= p; ++n) rho_pow[n] = rho_pow[n - 1] * sc.rho;

    for (int j = 0; j <= p; ++j) {
        for (int k = -j; k <= j; ++k) {
            std::complex<double> total(0.0, 0.0);
            const double A_jk = A_coeff(j, k);

            for (int n = 0; n <= j; ++n) {
                const int deg = j - n;
                for (int m = -n; m <= n; ++m) {
                    const int km = k - m;
                    if (std::abs(km) > deg) continue;

                    // GR eq. 45 term:
                    // O_{j-n}^{k-m} * i^{|k|-|m|-|k-m|} * A_n^m * A_{j-n}^{k-m}
                    //   * rho^n * Y_n^{-m}(alpha, beta) / A_j^k
                    total += child.M[sh_index(deg, km)]
                             * ipow(std::abs(k) - std::abs(m) - std::abs(km))
                             * A_coeff(n, m) * A_coeff(deg, km)
                             * rho_pow[n] * Y[sh_index(n, -m)] / A_jk;
                }
            }
            out.M[sh_index(j, k)] = total;
        }
    }
    return out;
}

namespace {

// Recursive upward pass: leaves get P2M; internal nodes get the M2M-merged
// sum of their children's expansions. Appends this node's NodeExpansion to
// `out` and returns a pointer to it (stable because `out` is a deque-like
// usage -- we reserve exact size up front by counting nodes first).
const MultipoleExpansion* upward(const OctreeNode* node,
                                  const Octree& tree,
                                  int order,
                                  std::vector<NodeExpansion>& out) {
    if (node->is_leaf) {
        auto expansion = particles_to_multipole(
            tree.particles(), node->particle_indices,
            node->box.cx, node->box.cy, node->box.cz, order);
        out.push_back(NodeExpansion{node, std::move(expansion)});
        return &out.back().expansion;
    }

    // Internal node: recurse into children first, then merge via M2M.
    std::vector<const MultipoleExpansion*> child_expansions;
    for (const auto& child : node->children) {
        if (child) {
            child_expansions.push_back(upward(child.get(), tree, order, out));
        }
    }

    MultipoleExpansion merged;
    merged.order = order;
    merged.center_x = node->box.cx;
    merged.center_y = node->box.cy;
    merged.center_z = node->box.cz;
    merged.M.assign((order + 1) * (order + 1), std::complex<double>(0.0, 0.0));

    for (const auto* child_exp : child_expansions) {
        auto shifted = multipole_to_multipole(
            *child_exp, node->box.cx, node->box.cy, node->box.cz);
        for (std::size_t i = 0; i < merged.M.size(); ++i) {
            merged.M[i] += shifted.M[i];
        }
    }

    out.push_back(NodeExpansion{node, std::move(merged)});
    return &out.back().expansion;
}

std::size_t count_tree_nodes(const OctreeNode* node) {
    if (!node) return 0;
    std::size_t total = 1;
    for (const auto& child : node->children) total += count_tree_nodes(child.get());
    return total;
}

}  // namespace

std::vector<NodeExpansion> build_upward_pass(const Octree& tree, int order) {
    std::vector<NodeExpansion> result;
    result.reserve(count_tree_nodes(tree.root()));  // keep pointers stable
    upward(tree.root(), tree, order, result);
    return result;
}

}  // namespace fmm
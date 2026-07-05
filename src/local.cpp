#include "fmm/local.hpp"
#include <cmath>

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

double A_coeff(int n, int m) {
    double sign = (n % 2 == 0) ? 1.0 : -1.0;
    return sign / std::sqrt(factorial(n - m) * factorial(n + m));
}

std::complex<double> ipow(int k) {
    switch (((k % 4) + 4) % 4) {
        case 0: return {1.0, 0.0};
        case 1: return {0.0, 1.0};
        case 2: return {-1.0, 0.0};
        default: return {0.0, -1.0};
    }
}

}  // namespace

double LocalExpansion::evaluate(double x, double y, double z) const {
    auto sc = to_spherical(x, y, z, center_x, center_y, center_z);
    auto Y = spherical_harmonics_gr_all(order, sc.theta, sc.phi);

    std::complex<double> total(0.0, 0.0);
    double r_pow_j = 1.0;  // r^j, starting at r^0
    for (int j = 0; j <= order; ++j) {
        for (int k = -j; k <= j; ++k) {
            // Phi += L_j^k * Y_j^k(theta, phi) * r^j   (GR eq. 47)
            total += L[sh_index(j, k)] * Y[sh_index(j, k)] * r_pow_j;
        }
        r_pow_j *= sc.rho;
    }
    return total.real();
}

LocalExpansion multipole_to_local(
    const MultipoleExpansion& multipole,
    double local_cx, double local_cy, double local_cz) {
    const int p = multipole.order;

    LocalExpansion out;
    out.order = p;
    out.center_x = local_cx;
    out.center_y = local_cy;
    out.center_z = local_cz;
    out.L.assign((p + 1) * (p + 1), std::complex<double>(0.0, 0.0));

    // (rho, alpha, beta): multipole center relative to the local center
    // (the vector Q in GR Theorem 5.2).
    auto sc = to_spherical(multipole.center_x, multipole.center_y, multipole.center_z,
                            local_cx, local_cy, local_cz);

    // Y up to degree 2p is needed (index j+n reaches 2p).
    auto Y = spherical_harmonics_gr_all(2 * p, sc.theta, sc.phi);

    // rho^{j+n+1} up to exponent 2p+1.
    std::vector<double> rho_pow(2 * p + 2);
    rho_pow[0] = 1.0;
    for (int e = 1; e <= 2 * p + 1; ++e) rho_pow[e] = rho_pow[e - 1] * sc.rho;

    for (int j = 0; j <= p; ++j) {
        for (int k = -j; k <= j; ++k) {
            std::complex<double> total(0.0, 0.0);
            const double A_jk = A_coeff(j, k);

            for (int n = 0; n <= p; ++n) {
                const double neg1_n = (n % 2 == 0) ? 1.0 : -1.0;
                for (int m = -n; m <= n; ++m) {
                    const int mk = m - k;
                    if (std::abs(mk) > j + n) continue;

                    // GR eq. 48 term:
                    // O_n^m * i^{|k-m|-|k|-|m|} * A_n^m * A_j^k * Y_{j+n}^{m-k}
                    //   / ( (-1)^n * A_{j+n}^{m-k} * rho^{j+n+1} )
                    std::complex<double> num =
                        multipole.M[sh_index(n, m)]
                        * ipow(std::abs(k - m) - std::abs(k) - std::abs(m))
                        * A_coeff(n, m) * A_jk
                        * Y[sh_index(j + n, mk)];
                    double den = neg1_n * A_coeff(j + n, mk) * rho_pow[j + n + 1];
                    total += num / den;
                }
            }
            out.L[sh_index(j, k)] = total;
        }
    }
    return out;
}

LocalExpansion local_to_local(
    const LocalExpansion& parent,
    double new_cx, double new_cy, double new_cz) {
    const int p = parent.order;

    LocalExpansion out;
    out.order = p;
    out.center_x = new_cx;
    out.center_y = new_cy;
    out.center_z = new_cz;
    out.L.assign((p + 1) * (p + 1), std::complex<double>(0.0, 0.0));

    // (rho, alpha, beta): old local center relative to the new center
    // (the vector Q in GR Theorem 5.3).
    auto sc = to_spherical(parent.center_x, parent.center_y, parent.center_z,
                            new_cx, new_cy, new_cz);

    if (sc.rho < 1e-15) {
        out.L = parent.L;
        return out;
    }

    auto Y = spherical_harmonics_gr_all(p, sc.theta, sc.phi);

    std::vector<double> rho_pow(p + 1);
    rho_pow[0] = 1.0;
    for (int e = 1; e <= p; ++e) rho_pow[e] = rho_pow[e - 1] * sc.rho;

    for (int j = 0; j <= p; ++j) {
        for (int k = -j; k <= j; ++k) {
            std::complex<double> total(0.0, 0.0);
            const double A_jk = A_coeff(j, k);

            for (int n = j; n <= p; ++n) {
                const double neg1_nj = ((n + j) % 2 == 0) ? 1.0 : -1.0;
                for (int m = -n; m <= n; ++m) {
                    const int mk = m - k;
                    if (std::abs(mk) > n - j) continue;

                    // GR eq. 52 term:
                    // O_n^m * i^{|m|-|m-k|-|k|} * A_{n-j}^{m-k} * A_j^k
                    //   * Y_{n-j}^{m-k} * rho^{n-j}  /  ( (-1)^{n+j} * A_n^m )
                    std::complex<double> num =
                        parent.L[sh_index(n, m)]
                        * ipow(std::abs(m) - std::abs(mk) - std::abs(k))
                        * A_coeff(n - j, mk) * A_jk
                        * Y[sh_index(n - j, mk)] * rho_pow[n - j];
                    double den = neg1_nj * A_coeff(n, m);
                    total += num / den;
                }
            }
            out.L[sh_index(j, k)] = total;
        }
    }
    return out;
}

}  // namespace fmm
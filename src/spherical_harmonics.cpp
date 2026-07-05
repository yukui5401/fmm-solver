#include "fmm/spherical_harmonics.hpp"
#include <algorithm>
#include <cmath>

namespace fmm {

namespace {
constexpr double PI = 3.14159265358979323846;

// (n - m)! / (n + m)!  computed as a product of reciprocals rather than
// via factorial(n)/factorial(n+m) directly, to avoid overflow for larger n.
double factorial_ratio(int n, int m) {
    double result = 1.0;
    for (int k = n - m + 1; k <= n + m; ++k) {
        result /= static_cast<double>(k);
    }
    return result;
}
}  // namespace

std::vector<double> associated_legendre(int n_max, double x) {
    const int size = (n_max + 1) * (n_max + 2) / 2;
    std::vector<double> P(size, 0.0);

    double somx2 = std::sqrt(std::max(0.0, 1.0 - x * x));

    // Seed diagonal terms P_m^m via the standard recurrence
    // P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}, built incrementally.
    P[legendre_index(0, 0)] = 1.0;
    double pmm = 1.0;
    for (int m = 1; m <= n_max; ++m) {
        pmm *= -(2 * m - 1) * somx2;
        P[legendre_index(m, m)] = pmm;
    }

    // Fill P_{m+1}^m, then climb n for each m via the three-term recurrence:
    //   (n-m) P_n^m = x(2n-1) P_{n-1}^m - (n+m-1) P_{n-2}^m
    for (int m = 0; m <= n_max; ++m) {
        if (m + 1 <= n_max) {
            P[legendre_index(m + 1, m)] = x * (2 * m + 1) * P[legendre_index(m, m)];
        }
        for (int n = m + 2; n <= n_max; ++n) {
            double term1 = (2 * n - 1) * x * P[legendre_index(n - 1, m)];
            double term2 = (n + m - 1) * P[legendre_index(n - 2, m)];
            P[legendre_index(n, m)] = (term1 - term2) / (n - m);
        }
    }

    return P;
}

std::complex<double> spherical_harmonic(int n, int m, double theta, double phi) {
    int am = std::abs(m);
    auto P = associated_legendre(n, std::cos(theta));

    double p_n_am = P[legendre_index(n, am)];
    double norm = std::sqrt((2 * n + 1) / (4.0 * PI) * factorial_ratio(n, am));
    std::complex<double> phase(std::cos(am * phi), std::sin(am * phi));
    std::complex<double> base = norm * p_n_am * phase;

    if (m >= 0) return base;

    // Y_n^{-m} = (-1)^m * conj(Y_n^m), the standard relation under the
    // Condon-Shortley phase convention used in associated_legendre above.
    double sign = (am % 2 == 0) ? 1.0 : -1.0;
    return sign * std::conj(base);
}

std::vector<std::complex<double>> spherical_harmonics_all(int n_max, double theta, double phi) {
    std::vector<std::complex<double>> Y((n_max + 1) * (n_max + 1));
    auto P = associated_legendre(n_max, std::cos(theta));

    for (int n = 0; n <= n_max; ++n) {
        for (int m = -n; m <= n; ++m) {
            int am = std::abs(m);
            double p_n_am = P[legendre_index(n, am)];
            double norm = std::sqrt((2 * n + 1) / (4.0 * PI) * factorial_ratio(n, am));
            std::complex<double> phase(std::cos(am * phi), std::sin(am * phi));
            std::complex<double> base = norm * p_n_am * phase;

            std::complex<double> value;
            if (m >= 0) {
                value = base;
            } else {
                double sign = (am % 2 == 0) ? 1.0 : -1.0;
                value = sign * std::conj(base);
            }
            Y[sh_index(n, m)] = value;
        }
    }
    return Y;
}

std::complex<double> spherical_harmonic_gr(int n, int m, double theta, double phi) {
    int am = std::abs(m);
    auto P = associated_legendre(n, std::cos(theta));
    double norm = std::sqrt(factorial_ratio(n, am));
    // Phase uses SIGNED m (GR eq. 34); P_n^{|m|} is real, so negative
    // orders come out as conjugates of positive orders automatically.
    std::complex<double> phase(std::cos(m * phi), std::sin(m * phi));
    return norm * P[legendre_index(n, am)] * phase;
}

std::vector<std::complex<double>> spherical_harmonics_gr_all(int n_max, double theta, double phi) {
    std::vector<std::complex<double>> Y((n_max + 1) * (n_max + 1));
    auto P = associated_legendre(n_max, std::cos(theta));

    for (int n = 0; n <= n_max; ++n) {
        for (int m = -n; m <= n; ++m) {
            int am = std::abs(m);
            double norm = std::sqrt(factorial_ratio(n, am));
            std::complex<double> phase(std::cos(m * phi), std::sin(m * phi));
            Y[sh_index(n, m)] = norm * P[legendre_index(n, am)] * phase;
        }
    }
    return Y;
}

}  // namespace fmm
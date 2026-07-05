#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include "fmm/spherical_harmonics.hpp"

using namespace fmm;
using Catch::Matchers::WithinAbs;

namespace {
constexpr double PI = 3.14159265358979323846;
}

TEST_CASE("spherical_harmonic: Y_0^0 is the constant 1/sqrt(4*pi) everywhere", "[spherical_harmonics]") {
    double expected = 1.0 / std::sqrt(4.0 * PI);
    auto y1 = spherical_harmonic(0, 0, 0.3, 1.2);
    auto y2 = spherical_harmonic(0, 0, 2.1, 5.0);
    REQUIRE_THAT(y1.real(), WithinAbs(expected, 1e-10));
    REQUIRE_THAT(y1.imag(), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(y2.real(), WithinAbs(expected, 1e-10));
}

TEST_CASE("spherical_harmonic: Y_1^0 matches sqrt(3/4pi) * cos(theta)", "[spherical_harmonics]") {
    double theta = 0.7;
    double expected = std::sqrt(3.0 / (4.0 * PI)) * std::cos(theta);
    auto y = spherical_harmonic(1, 0, theta, 0.0);
    REQUIRE_THAT(y.real(), WithinAbs(expected, 1e-10));
    REQUIRE_THAT(y.imag(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("spherical_harmonic: negative-m relation Y_n^-m = (-1)^m conj(Y_n^m)", "[spherical_harmonics]") {
    double theta = 1.1, phi = 0.9;
    for (int n = 0; n <= 4; ++n) {
        for (int m = 1; m <= n; ++m) {
            auto y_pos = spherical_harmonic(n, m, theta, phi);
            auto y_neg = spherical_harmonic(n, -m, theta, phi);
            double sign = (m % 2 == 0) ? 1.0 : -1.0;
            auto expected = sign * std::conj(y_pos);
            REQUIRE_THAT(y_neg.real(), WithinAbs(expected.real(), 1e-9));
            REQUIRE_THAT(y_neg.imag(), WithinAbs(expected.imag(), 1e-9));
        }
    }
}

TEST_CASE("spherical_harmonics_all: matches individually-computed values", "[spherical_harmonics]") {
    double theta = 0.5, phi = 1.3;
    int n_max = 5;
    auto all = spherical_harmonics_all(n_max, theta, phi);

    for (int n = 0; n <= n_max; ++n) {
        for (int m = -n; m <= n; ++m) {
            auto individual = spherical_harmonic(n, m, theta, phi);
            auto from_all = all[sh_index(n, m)];
            REQUIRE_THAT(from_all.real(), WithinAbs(individual.real(), 1e-9));
            REQUIRE_THAT(from_all.imag(), WithinAbs(individual.imag(), 1e-9));
        }
    }
}

TEST_CASE("spherical_harmonics_all: orthonormality via numerical quadrature over the sphere", "[spherical_harmonics]") {
    // Coarse grid quadrature: sum Y_n1^m1 * conj(Y_n2^m2) * sin(theta) dtheta dphi
    // should be ~1 when (n1,m1)==(n2,m2), ~0 otherwise. Loose tolerance since
    // this is a simple midpoint-rule quadrature, not adaptive integration.
    const int n_theta = 60, n_phi = 60;
    const double dtheta = PI / n_theta;
    const double dphi = 2 * PI / n_phi;

    auto inner_product = [&](int n1, int m1, int n2, int m2) {
        std::complex<double> total = 0.0;
        for (int i = 0; i < n_theta; ++i) {
            double theta = (i + 0.5) * dtheta;
            for (int j = 0; j < n_phi; ++j) {
                double phi = (j + 0.5) * dphi;
                auto y1 = spherical_harmonic(n1, m1, theta, phi);
                auto y2 = spherical_harmonic(n2, m2, theta, phi);
                total += y1 * std::conj(y2) * std::sin(theta) * dtheta * dphi;
            }
        }
        return total;
    };

    auto same = inner_product(2, 1, 2, 1);
    REQUIRE_THAT(same.real(), WithinAbs(1.0, 5e-3));
    REQUIRE_THAT(same.imag(), WithinAbs(0.0, 5e-3));

    auto different_degree = inner_product(1, 0, 3, 0);
    REQUIRE_THAT(different_degree.real(), WithinAbs(0.0, 5e-3));

    auto different_order = inner_product(2, 1, 2, -1);
    REQUIRE_THAT(different_order.real(), WithinAbs(0.0, 5e-3));
}
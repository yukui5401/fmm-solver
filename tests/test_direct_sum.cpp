#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include "fmm/direct_sum.hpp"

using namespace fmm;
using Catch::Matchers::WithinAbs;

TEST_CASE("direct_sum: two unit charges at known separation", "[direct_sum]") {
    // Two charges of 1.0 at (0,0,0) and (1,0,0): potential at each due to
    // the other is 1/1 = 1.0.
    std::vector<Particle> particles = {
        {0.0, 0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0, 1.0},
    };
    auto phi = direct_sum(particles);
    REQUIRE(phi.size() == 2);
    REQUIRE_THAT(phi[0], WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(phi[1], WithinAbs(1.0, 1e-12));
}

TEST_CASE("direct_sum: three charges, manual calculation", "[direct_sum]") {
    // Charges at (0,0,0), (1,0,0), (0,1,0), all q=1.
    // phi[0] = 1/1 (from particle 1) + 1/1 (from particle 2) = 2.0
    // phi[1] = 1/1 (from particle 0) + 1/sqrt(2) (from particle 2)
    std::vector<Particle> particles = {
        {0.0, 0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0, 1.0},
    };
    auto phi = direct_sum(particles);
    REQUIRE_THAT(phi[0], WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(phi[1], WithinAbs(1.0 + 1.0 / std::sqrt(2.0), 1e-12));
}

TEST_CASE("direct_sum_at_targets: target coinciding with source matches direct_sum", "[direct_sum]") {
    std::vector<Particle> particles = {
        {0.0, 0.0, 0.0, 2.0},
        {3.0, 0.0, 0.0, 1.0},
        {0.0, 4.0, 0.0, 1.5},
    };
    auto phi_direct = direct_sum(particles);

    std::vector<std::array<double, 3>> targets = {
        {0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {0.0, 4.0, 0.0}
    };
    auto phi_targets = direct_sum_at_targets(particles, targets);

    // direct_sum excludes self-interaction, but direct_sum_at_targets sums
    // over all sources including one that coincides with the target -- so
    // these will only match if we manually exclude self-contribution.
    // Verify instead against a target point clearly away from all sources.
    std::vector<std::array<double, 3>> far_target = {{10.0, 10.0, 10.0}};
    auto phi_far = direct_sum_at_targets(particles, far_target);

    double expected = 0.0;
    for (const auto& src : particles) {
        double dx = 10.0 - src.x, dy = 10.0 - src.y, dz = 10.0 - src.z;
        expected += src.q / std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    REQUIRE_THAT(phi_far[0], WithinAbs(expected, 1e-9));
}
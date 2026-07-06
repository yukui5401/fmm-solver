#include <catch2/catch_test_macros.hpp>
#include <random>
#include "fmm/direct_sum.hpp"
#include "fmm/fmm_solver.hpp"

using namespace fmm;

namespace {
std::vector<Particle> random_particles(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos(-1.0, 1.0);
    std::uniform_real_distribution<double> charge(0.5, 2.0);
    std::vector<Particle> particles;
    for (int i = 0; i < n; ++i)
        particles.push_back({pos(rng), pos(rng), pos(rng), charge(rng)});
    return particles;
}

double max_rel_error(const std::vector<double>& approx,
                     const std::vector<double>& exact) {
    double worst = 0.0;
    for (std::size_t i = 0; i < exact.size(); ++i)
        worst = std::max(worst, std::abs(approx[i] - exact[i]) / std::abs(exact[i]));
    return worst;
}
}  // namespace

TEST_CASE("fmm_evaluate matches direct summation for uniform random particles", "[fmm][integration]") {
    auto particles = random_particles(300, 21);
    auto exact = direct_sum(particles);
    auto approx = fmm_evaluate(particles, /*order=*/12, /*levels=*/2);
    REQUIRE(max_rel_error(approx, exact) < 1e-4);
}

TEST_CASE("fmm_evaluate error decreases with expansion order", "[fmm]") {
    auto particles = random_particles(200, 5);
    auto exact = direct_sum(particles);

    double err_low = max_rel_error(fmm_evaluate(particles, 4, 2), exact);
    double err_high = max_rel_error(fmm_evaluate(particles, 14, 2), exact);

    REQUIRE(err_high < err_low);
    REQUIRE(err_high < 1e-5);
}

TEST_CASE("fmm_evaluate works at deeper refinement levels", "[fmm]") {
    auto particles = random_particles(500, 33);
    auto exact = direct_sum(particles);
    auto approx = fmm_evaluate(particles, /*order=*/12, /*levels=*/3);
    REQUIRE(max_rel_error(approx, exact) < 1e-4);
}

TEST_CASE("fmm_evaluate handles clustered (non-uniform) distributions", "[fmm]") {
    // Two tight clusters far apart inside the cube -- stresses empty-box
    // handling and the interaction-list far-field path.
    std::mt19937 rng(8);
    std::uniform_real_distribution<double> off(-0.05, 0.05);
    std::vector<Particle> particles;
    for (int i = 0; i < 60; ++i)
        particles.push_back({-0.8 + off(rng), -0.8 + off(rng), -0.8 + off(rng), 1.0});
    for (int i = 0; i < 60; ++i)
        particles.push_back({0.8 + off(rng), 0.8 + off(rng), 0.8 + off(rng), 1.5});

    auto exact = direct_sum(particles);
    auto approx = fmm_evaluate(particles, /*order=*/12, /*levels=*/3);
    REQUIRE(max_rel_error(approx, exact) < 1e-4);
}
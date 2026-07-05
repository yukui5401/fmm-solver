#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include "fmm/direct_sum.hpp"
#include "fmm/multipole.hpp"
#include "fmm/octree.hpp"

using namespace fmm;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("particles_to_multipole: single charge converges to direct sum as order increases", "[multipole]") {
    // Even with one source, the multipole series is an infinite expansion
    // of 1/r -- truncating at finite order leaves residual error that
    // should shrink as order increases (it does not vanish just because
    // there's a single source).
    std::vector<Particle> particles = {{1.0, 2.0, 3.0, 5.0}};
    std::vector<std::size_t> indices = {0};
    std::array<double, 3> target = {20.0, -15.0, 10.0};
    auto direct = direct_sum_at_targets(particles, {target});

    double err_low = std::abs(
        particles_to_multipole(particles, indices, 0.0, 0.0, 0.0, 2).evaluate(target[0], target[1], target[2])
        - direct[0]);
    double err_high = std::abs(
        particles_to_multipole(particles, indices, 0.0, 0.0, 0.0, 12).evaluate(target[0], target[1], target[2])
        - direct[0]);

    REQUIRE(err_high < err_low);
    REQUIRE(err_high < 1e-9);
}

TEST_CASE("particles_to_multipole: cluster of charges converges to direct sum as order increases", "[multipole]") {
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> pos(-1.0, 1.0);
    std::uniform_real_distribution<double> charge(0.5, 2.0);

    std::vector<Particle> particles;
    std::vector<std::size_t> indices;
    for (int i = 0; i < 20; ++i) {
        particles.push_back({pos(rng), pos(rng), pos(rng), charge(rng)});
        indices.push_back(i);
    }

    // Target well outside the source cluster (cluster fits in [-1,1]^3,
    // so a target at distance ~30 is comfortably in the far field).
    std::vector<std::array<double, 3>> target = {{20.0, 15.0, -10.0}};
    auto direct = direct_sum_at_targets(particles, target);

    double err_low_order = 0.0, err_high_order = 0.0;
    for (int order : {1, 8}) {
        auto expansion = particles_to_multipole(particles, indices, 0.0, 0.0, 0.0, order);
        double approx = expansion.evaluate(target[0][0], target[0][1], target[0][2]);
        double err = std::abs(approx - direct[0]);
        if (order == 1) err_low_order = err;
        if (order == 8) err_high_order = err;
    }

    // Higher-order expansion should be dramatically more accurate.
    REQUIRE(err_high_order < err_low_order);
    // At order 8 for a target this far away, error should be extremely small.
    REQUIRE(err_high_order < 1e-6);
}

TEST_CASE("particles_to_multipole: expansion centered off-origin still matches direct sum", "[multipole]") {
    // Regression check that the center-shifting logic in to_spherical is
    // actually being used correctly (a bug here would silently look fine
    // if center happened to be the origin in every other test).
    std::vector<Particle> particles = {
        {5.0, 5.1, 4.9, 1.0},
        {5.2, 4.8, 5.0, 2.0},
        {4.9, 5.0, 5.1, 1.5},
    };
    std::vector<std::size_t> indices = {0, 1, 2};

    auto expansion = particles_to_multipole(particles, indices, 5.0, 5.0, 5.0, /*order=*/10);

    std::vector<std::array<double, 3>> target = {{50.0, -40.0, 30.0}};
    auto direct = direct_sum_at_targets(particles, target);
    double approx = expansion.evaluate(target[0][0], target[0][1], target[0][2]);

    REQUIRE_THAT(approx, WithinRel(direct[0], 1e-8));
}

TEST_CASE("multipole_to_multipole: translated coefficients match direct P2M about new center", "[multipole][m2m]") {
    // GR Theorem 5.1 states the shift is EXACT for truncated expansions:
    // translating child coefficients must reproduce, to machine precision,
    // the expansion built directly from the particles about the new center.
    // This is the strongest available check -- coefficient-level equality,
    // not just agreement of evaluated potentials.
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> pos(-0.5, 0.5);

    std::vector<Particle> particles;
    std::vector<std::size_t> indices;
    for (int i = 0; i < 15; ++i) {
        particles.push_back({0.3 + pos(rng) * 0.2, -0.2 + pos(rng) * 0.2,
                             0.4 + pos(rng) * 0.2, 1.0 + 0.1 * i});
        indices.push_back(i);
    }

    const int order = 8;
    auto child = particles_to_multipole(particles, indices, 0.3, -0.2, 0.4, order);
    auto reference = particles_to_multipole(particles, indices, 0.0, 0.0, 0.0, order);
    auto translated = multipole_to_multipole(child, 0.0, 0.0, 0.0);

    for (std::size_t i = 0; i < reference.M.size(); ++i) {
        REQUIRE_THAT(translated.M[i].real(), WithinAbs(reference.M[i].real(), 1e-10));
        REQUIRE_THAT(translated.M[i].imag(), WithinAbs(reference.M[i].imag(), 1e-10));
    }
}

TEST_CASE("multipole_to_multipole: chained shifts still match direct P2M", "[multipole][m2m]") {
    // Shift child -> intermediate -> final and compare against direct P2M
    // about the final center; exactness should survive composition, which
    // is what the tree's multi-level upward pass relies on.
    std::vector<Particle> particles = {
        {1.05, 2.02, 2.97, 1.0}, {0.98, 1.96, 3.03, 2.0}, {1.01, 2.05, 3.01, 0.5},
    };
    std::vector<std::size_t> indices = {0, 1, 2};

    const int order = 6;
    auto e0 = particles_to_multipole(particles, indices, 1.0, 2.0, 3.0, order);
    auto e1 = multipole_to_multipole(e0, 0.5, 1.0, 1.5);
    auto e2 = multipole_to_multipole(e1, 0.0, 0.0, 0.0);
    auto reference = particles_to_multipole(particles, indices, 0.0, 0.0, 0.0, order);

    for (std::size_t i = 0; i < reference.M.size(); ++i) {
        REQUIRE_THAT(e2.M[i].real(), WithinAbs(reference.M[i].real(), 1e-9));
        REQUIRE_THAT(e2.M[i].imag(), WithinAbs(reference.M[i].imag(), 1e-9));
    }
}

TEST_CASE("build_upward_pass: root expansion matches direct sum for the whole particle set", "[multipole]") {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> pos(-2.0, 2.0);

    std::vector<Particle> particles;
    for (int i = 0; i < 100; ++i) {
        particles.push_back({pos(rng), pos(rng), pos(rng), 1.0});
    }

    Octree tree(particles, /*max_particles_per_leaf=*/8);
    auto expansions = build_upward_pass(tree, /*order=*/10);

    // Post-order traversal: the root's expansion is the LAST entry. Locate
    // it by node pointer rather than position to stay robust either way.
    const MultipoleExpansion* root_expansion = nullptr;
    for (const auto& ne : expansions) {
        if (ne.node == tree.root()) root_expansion = &ne.expansion;
    }
    REQUIRE(root_expansion != nullptr);

    std::vector<std::array<double, 3>> target = {{100.0, -80.0, 60.0}};
    auto direct = direct_sum_at_targets(particles, target);
    double approx = root_expansion->evaluate(target[0][0], target[0][1], target[0][2]);

    REQUIRE_THAT(approx, WithinRel(direct[0], 1e-6));
}

TEST_CASE("build_upward_pass: every leaf's expansion matches direct sum for its own particles", "[multipole]") {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> pos(-3.0, 3.0);

    std::vector<Particle> particles;
    for (int i = 0; i < 200; ++i) {
        particles.push_back({pos(rng), pos(rng), pos(rng), 1.0});
    }

    Octree tree(particles, /*max_particles_per_leaf=*/6);
    auto expansions = build_upward_pass(tree, /*order=*/8);

    int leaves_checked = 0;
    for (const auto& ne : expansions) {
        if (!ne.node->is_leaf || ne.node->particle_indices.empty()) continue;

        std::vector<Particle> leaf_particles;
        for (auto idx : ne.node->particle_indices) leaf_particles.push_back(particles[idx]);

        // Evaluate the leaf's own multipole expansion at a point far from
        // this leaf's box (using the box's half-width as a proxy for scale).
        double far = ne.node->box.half_width * 50.0 + 50.0;
        std::vector<std::array<double, 3>> target = {
            {ne.node->box.cx + far, ne.node->box.cy + far, ne.node->box.cz + far}
        };

        auto direct = direct_sum_at_targets(leaf_particles, target);
        double approx = ne.expansion.evaluate(target[0][0], target[0][1], target[0][2]);

        REQUIRE_THAT(approx, WithinRel(direct[0], 1e-6));
        ++leaves_checked;
    }
    REQUIRE(leaves_checked > 0);
}
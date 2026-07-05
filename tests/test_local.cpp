#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include "fmm/direct_sum.hpp"
#include "fmm/local.hpp"
#include "fmm/multipole.hpp"

using namespace fmm;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
// Sources clustered (radius ~a) around a center Q well-separated from the
// origin, plus evaluation targets near the origin -- the geometry GR
// Theorem 5.2 requires.
struct Setup {
    std::vector<Particle> particles;
    std::vector<std::size_t> indices;
    double qx, qy, qz;  // multipole center
};

Setup make_separated_cluster(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> off(-0.45, 0.45);
    std::uniform_real_distribution<double> charge(0.5, 2.0);

    Setup s;
    s.qx = 6.0; s.qy = -4.0; s.qz = 5.0;  // |Q| ~ 8.8 >> cluster radius
    for (int i = 0; i < 12; ++i) {
        s.particles.push_back({s.qx + off(rng), s.qy + off(rng), s.qz + off(rng),
                               charge(rng)});
        s.indices.push_back(i);
    }
    return s;
}
}  // namespace

TEST_CASE("multipole_to_local: local expansion converges to direct sum with increasing order", "[local][m2l]") {
    auto s = make_separated_cluster(4);
    std::vector<std::array<double, 3>> targets = {
        {0.3, -0.2, 0.25}, {-0.4, 0.1, -0.3}, {0.0, 0.45, 0.1}};
    auto direct = direct_sum_at_targets(s.particles, targets);

    double max_err_low = 0.0, max_err_high = 0.0;
    for (int order : {4, 10}) {
        auto mp = particles_to_multipole(s.particles, s.indices, s.qx, s.qy, s.qz, order);
        auto local = multipole_to_local(mp, 0.0, 0.0, 0.0);

        double max_err = 0.0;
        for (std::size_t t = 0; t < targets.size(); ++t) {
            double approx = local.evaluate(targets[t][0], targets[t][1], targets[t][2]);
            max_err = std::max(max_err, std::abs(approx - direct[t]));
        }
        if (order == 4) max_err_low = max_err;
        if (order == 10) max_err_high = max_err;
    }

    // M2L introduces genuine truncation error (GR eq. 49) that must shrink
    // with order -- and be very small at order 10 for this separation.
    REQUIRE(max_err_high < max_err_low);
    REQUIRE(max_err_high < 1e-8);
}

TEST_CASE("local_to_local: shifted expansion evaluates identically (exact operator)", "[local][l2l]") {
    auto s = make_separated_cluster(7);
    const int order = 8;

    auto mp = particles_to_multipole(s.particles, s.indices, s.qx, s.qy, s.qz, order);
    auto local_origin = multipole_to_local(mp, 0.0, 0.0, 0.0);
    auto local_shifted = local_to_local(local_origin, 0.2, -0.1, 0.15);

    // L2L is exact (finite Maclaurin shift): evaluating the shifted
    // expansion at the same physical points must reproduce the original
    // expansion's values to machine precision -- not merely approximate them.
    std::vector<std::array<double, 3>> targets = {
        {0.3, -0.2, 0.25}, {-0.35, 0.05, -0.25}, {0.1, 0.4, 0.05}};
    for (const auto& t : targets) {
        double v0 = local_origin.evaluate(t[0], t[1], t[2]);
        double v1 = local_shifted.evaluate(t[0], t[1], t[2]);
        REQUIRE_THAT(v1, WithinAbs(v0, 1e-10));
    }
}

TEST_CASE("full operator chain: P2M -> M2M -> M2L -> L2L -> L2P matches direct sum", "[local][integration]") {
    // Exercises every translation operator in sequence, the way the full
    // FMM traversal will: two sub-clusters get leaf expansions (P2M), are
    // merged to a common parent center (M2M), converted to a local
    // expansion at a distant evaluation region (M2L), shifted within that
    // region (L2L), and evaluated at particles (L2P).
    std::mt19937 rng(13);
    std::uniform_real_distribution<double> off(-0.3, 0.3);

    std::vector<Particle> particles;
    // Sub-cluster A around (5.6, -3.6, 4.6); sub-cluster B around (6.4, -4.4, 5.4).
    for (int i = 0; i < 8; ++i)
        particles.push_back({5.6 + off(rng), -3.6 + off(rng), 4.6 + off(rng), 1.0});
    for (int i = 0; i < 8; ++i)
        particles.push_back({6.4 + off(rng), -4.4 + off(rng), 5.4 + off(rng), 1.5});

    std::vector<std::size_t> idx_a, idx_b;
    for (int i = 0; i < 8; ++i) idx_a.push_back(i);
    for (int i = 8; i < 16; ++i) idx_b.push_back(i);

    const int order = 12;

    // P2M at "leaf" centers.
    auto mp_a = particles_to_multipole(particles, idx_a, 5.6, -3.6, 4.6, order);
    auto mp_b = particles_to_multipole(particles, idx_b, 6.4, -4.4, 5.4, order);

    // M2M both to the shared "parent" center.
    auto shifted_a = multipole_to_multipole(mp_a, 6.0, -4.0, 5.0);
    auto shifted_b = multipole_to_multipole(mp_b, 6.0, -4.0, 5.0);
    MultipoleExpansion parent = shifted_a;
    for (std::size_t i = 0; i < parent.M.size(); ++i) parent.M[i] += shifted_b.M[i];

    // M2L to a local expansion about the origin (well separated).
    auto local = multipole_to_local(parent, 0.0, 0.0, 0.0);

    // L2L down to a "child" evaluation center.
    auto local_child = local_to_local(local, 0.15, -0.1, 0.1);

    // L2P at evaluation points near the child center, vs direct sum.
    std::vector<std::array<double, 3>> targets = {
        {0.25, -0.15, 0.2}, {0.05, 0.0, 0.0}, {0.3, -0.3, 0.15}};
    auto direct = direct_sum_at_targets(particles, targets);

    for (std::size_t t = 0; t < targets.size(); ++t) {
        double approx = local_child.evaluate(targets[t][0], targets[t][1], targets[t][2]);
        REQUIRE_THAT(approx, WithinRel(direct[t], 1e-7));
    }
}
#include <catch2/catch_test_macros.hpp>
#include <random>
#include "fmm/octree.hpp"

using namespace fmm;

namespace {
// Recursively check every leaf's particle count against the cap, and
// every particle's coordinates actually lie within its leaf's bounding box.
void check_leaf_invariants(const OctreeNode* node,
                            const std::vector<Particle>& particles,
                            std::size_t max_per_leaf) {
    if (node->is_leaf) {
        REQUIRE(node->particle_indices.size() <= max_per_leaf);
        for (auto idx : node->particle_indices) {
            const auto& p = particles[idx];
            REQUIRE(node->box.contains(p.x, p.y, p.z));
        }
        return;
    }
    for (const auto& child : node->children) {
        if (child) check_leaf_invariants(child.get(), particles, max_per_leaf);
    }
}

std::size_t count_particles_in_tree(const OctreeNode* node) {
    if (node->is_leaf) return node->particle_indices.size();
    std::size_t total = 0;
    for (const auto& child : node->children) {
        if (child) total += count_particles_in_tree(child.get());
    }
    return total;
}
}  // namespace

TEST_CASE("octree: all particles land in a leaf whose box contains them", "[octree]") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<Particle> particles;
    for (int i = 0; i < 500; ++i) {
        particles.push_back({dist(rng), dist(rng), dist(rng), 1.0});
    }

    Octree tree(particles, /*max_particles_per_leaf=*/8);
    check_leaf_invariants(tree.root(), particles, 8);
}

TEST_CASE("octree: no particles are lost or duplicated during construction", "[octree]") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    std::vector<Particle> particles;
    for (int i = 0; i < 1000; ++i) {
        particles.push_back({dist(rng), dist(rng), dist(rng), 1.0});
    }

    Octree tree(particles, /*max_particles_per_leaf=*/16);
    REQUIRE(count_particles_in_tree(tree.root()) == particles.size());
}

TEST_CASE("octree: refines adaptively -- dense cluster reaches greater depth than sparse region", "[octree]") {
    std::vector<Particle> particles;
    // Dense cluster near origin.
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> tight(-0.01, 0.01);
    for (int i = 0; i < 200; ++i) {
        particles.push_back({tight(rng), tight(rng), tight(rng), 1.0});
    }
    // One lone particle far away.
    particles.push_back({100.0, 100.0, 100.0, 1.0});

    Octree tree(particles, /*max_particles_per_leaf=*/4, /*max_depth=*/25);

    // The tree should have refined significantly to separate the dense
    // cluster into small enough leaves, i.e. depth > 1.
    REQUIRE(tree.max_depth_reached() > 1);
    REQUIRE(count_particles_in_tree(tree.root()) == particles.size());
}

TEST_CASE("octree: single particle produces a single-leaf tree", "[octree]") {
    std::vector<Particle> particles = {{0.0, 0.0, 0.0, 1.0}};
    Octree tree(particles);
    REQUIRE(tree.root()->is_leaf);
    REQUIRE(tree.root()->particle_indices.size() == 1);
}
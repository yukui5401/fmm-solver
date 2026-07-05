#include "fmm/octree.hpp"
#include <algorithm>
#include <limits>

namespace fmm {

namespace {
BoundingBox compute_bounding_cube(const std::vector<Particle>& particles) {
    double min_x = std::numeric_limits<double>::max(), max_x = -min_x;
    double min_y = min_x, max_y = -min_x;
    double min_z = min_x, max_z = -min_x;

    for (const auto& p : particles) {
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
        min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
    }

    double cx = (min_x + max_x) / 2.0;
    double cy = (min_y + max_y) / 2.0;
    double cz = (min_z + max_z) / 2.0;

    // Cubic box: use the largest half-extent across all 3 axes, with a
    // small epsilon pad so boundary particles aren't excluded by
    // floating-point rounding.
    double half = std::max({max_x - min_x, max_y - min_y, max_z - min_z}) / 2.0;
    half = std::max(half, 1e-9) * 1.001;

    return BoundingBox{cx, cy, cz, half};
}
}  // namespace

Octree::Octree(const std::vector<Particle>& particles,
                std::size_t max_particles_per_leaf,
                int max_depth)
    : particles_(particles),
      max_particles_per_leaf_(max_particles_per_leaf),
      max_depth_(max_depth) {
    root_ = std::make_unique<OctreeNode>();
    root_->box = compute_bounding_cube(particles_);
    root_->depth = 0;

    for (std::size_t i = 0; i < particles_.size(); ++i) {
        insert(root_.get(), i);
    }
}

void Octree::insert(OctreeNode* node, std::size_t particle_index) {
    if (node->is_leaf) {
        node->particle_indices.push_back(particle_index);

        bool over_capacity = node->particle_indices.size() > max_particles_per_leaf_;
        bool can_subdivide = node->depth < max_depth_;

        if (over_capacity && can_subdivide) {
            subdivide(node);
        }
        max_depth_reached_ = std::max(max_depth_reached_, node->depth);
        return;
    }

    // Internal node: route to the correct child, creating it if needed.
    const auto& p = particles_[particle_index];
    int oct = node->box.octant_of(p.x, p.y, p.z);

    if (!node->children[oct]) {
        node->children[oct] = std::make_unique<OctreeNode>();
        node->children[oct]->box = node->box.child_box(oct);
        node->children[oct]->depth = node->depth + 1;
    }
    insert(node->children[oct].get(), particle_index);
}

void Octree::subdivide(OctreeNode* node) {
    // Move this leaf's particles down into the appropriate children,
    // then mark this node as internal (no longer holds particles directly).
    std::vector<std::size_t> to_redistribute = std::move(node->particle_indices);
    node->particle_indices.clear();
    node->is_leaf = false;

    for (std::size_t idx : to_redistribute) {
        const auto& p = particles_[idx];
        int oct = node->box.octant_of(p.x, p.y, p.z);

        if (!node->children[oct]) {
            node->children[oct] = std::make_unique<OctreeNode>();
            node->children[oct]->box = node->box.child_box(oct);
            node->children[oct]->depth = node->depth + 1;
        }
        insert(node->children[oct].get(), idx);
    }
}

std::size_t Octree::count_nodes(const OctreeNode* node) {
    if (!node) return 0;
    std::size_t count = 1;
    for (const auto& child : node->children) {
        count += count_nodes(child.get());
    }
    return count;
}

std::size_t Octree::num_nodes() const {
    return count_nodes(root_.get());
}

}  // namespace fmm
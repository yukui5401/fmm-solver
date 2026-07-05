#pragma once
#include <array>
#include <memory>
#include <vector>
#include "fmm/direct_sum.hpp"

namespace fmm {

// Axis-aligned bounding box: center + half-width (a cube, since octree
// nodes are always cubic regardless of the point distribution's shape).
struct BoundingBox {
    double cx, cy, cz;
    double half_width;

    bool contains(double x, double y, double z) const {
        return x >= cx - half_width && x <= cx + half_width &&
               y >= cy - half_width && y <= cy + half_width &&
               z >= cz - half_width && z <= cz + half_width;
    }

    // Which of the 8 octants (0-7) does this point belong to, relative
    // to this box's center? Bit 0 = x half, bit 1 = y half, bit 2 = z half.
    int octant_of(double x, double y, double z) const {
        int oct = 0;
        if (x >= cx) oct |= 1;
        if (y >= cy) oct |= 2;
        if (z >= cz) oct |= 4;
        return oct;
    }

    BoundingBox child_box(int octant) const {
        double hw = half_width / 2.0;
        double ox = (octant & 1) ? cx + hw : cx - hw;
        double oy = (octant & 2) ? cy + hw : cy - hw;
        double oz = (octant & 4) ? cz + hw : cz - hw;
        return BoundingBox{ox, oy, oz, hw};
    }
};

// A node in the adaptive octree. Leaves hold particle indices directly;
// internal nodes hold up to 8 children (nullptr where a child octant is
// empty). Refinement stops once a node holds <= max_particles_per_leaf
// particles, or a maximum depth is reached (safety valve against
// degenerate/duplicate-point inputs that would otherwise refine forever).
struct OctreeNode {
    BoundingBox box;
    std::vector<std::size_t> particle_indices;  // only populated at leaves
    std::array<std::unique_ptr<OctreeNode>, 8> children;
    bool is_leaf = true;
    int depth = 0;

    // Multipole/local expansion coefficients get attached here in later
    // stages (P2M/M2M/M2L/L2L) -- left absent for now since this module
    // only covers tree construction and geometry.
};

class Octree {
public:
    Octree(const std::vector<Particle>& particles,
           std::size_t max_particles_per_leaf = 8,
           int max_depth = 20);

    const OctreeNode* root() const { return root_.get(); }
    const std::vector<Particle>& particles() const { return particles_; }

    // Total number of nodes in the tree (for diagnostics/tests).
    std::size_t num_nodes() const;

    // Max depth actually reached (<= max_depth given at construction).
    int max_depth_reached() const { return max_depth_reached_; }

private:
    void insert(OctreeNode* node, std::size_t particle_index);
    void subdivide(OctreeNode* node);
    static std::size_t count_nodes(const OctreeNode* node);

    std::vector<Particle> particles_;
    std::unique_ptr<OctreeNode> root_;
    std::size_t max_particles_per_leaf_;
    int max_depth_;
    int max_depth_reached_ = 0;
};

}  // namespace fmm
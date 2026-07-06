#include "fmm/fmm_solver.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include "fmm/local.hpp"
#include "fmm/multipole.hpp"

namespace fmm {

namespace {

struct Grid {
    double min_x, min_y, min_z;  // corner of the bounding cube
    double width;                 // cube side length
};

Grid bounding_cube(const std::vector<Particle>& particles) {
    double lo = std::numeric_limits<double>::max(), hi = -lo;
    double min_x = lo, min_y = lo, min_z = lo, max_x = hi, max_y = hi, max_z = hi;
    for (const auto& p : particles) {
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
        min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
    }
    double width = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    width = std::max(width, 1e-9) * 1.000001;  // pad against boundary rounding
    return {min_x, min_y, min_z, width};
}

inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

// Linear index of box (ix, iy, iz) at a level with `n` boxes per side.
inline std::size_t lin(int ix, int iy, int iz, int n) {
    return (static_cast<std::size_t>(iz) * n + iy) * n + ix;
}

}  // namespace

std::vector<double> fmm_evaluate(const std::vector<Particle>& particles,
                                  int order, int levels) {
    const std::size_t N = particles.size();
    std::vector<double> phi(N, 0.0);
    if (N == 0) return phi;

    Grid g = bounding_cube(particles);
    const int nfine = 1 << levels;  // boxes per side at finest level

    // --- Assign particles to finest-level boxes ---
    std::vector<std::vector<std::size_t>> box_particles(
        static_cast<std::size_t>(nfine) * nfine * nfine);
    auto box_of = [&](const Particle& p) {
        double h = g.width / nfine;
        int ix = clampi(static_cast<int>((p.x - g.min_x) / h), 0, nfine - 1);
        int iy = clampi(static_cast<int>((p.y - g.min_y) / h), 0, nfine - 1);
        int iz = clampi(static_cast<int>((p.z - g.min_z) / h), 0, nfine - 1);
        return std::array<int, 3>{ix, iy, iz};
    };
    for (std::size_t i = 0; i < N; ++i) {
        auto b = box_of(particles[i]);
        box_particles[lin(b[0], b[1], b[2], nfine)].push_back(i);
    }

    auto box_center = [&](int level, int ix, int iy, int iz) {
        double h = g.width / (1 << level);
        return std::array<double, 3>{g.min_x + (ix + 0.5) * h,
                                      g.min_y + (iy + 0.5) * h,
                                      g.min_z + (iz + 0.5) * h};
    };

    // Per-level storage: multipole expansions, local expansions, occupancy.
    std::vector<std::vector<MultipoleExpansion>> Phi(levels + 1);
    std::vector<std::vector<LocalExpansion>> Psi(levels + 1);
    std::vector<std::vector<char>> occupied(levels + 1);
    const std::size_t ncoef = static_cast<std::size_t>(order + 1) * (order + 1);

    for (int l = 0; l <= levels; ++l) {
        int n = 1 << l;
        std::size_t nb = static_cast<std::size_t>(n) * n * n;
        Phi[l].resize(nb);
        Psi[l].resize(nb);
        occupied[l].assign(nb, 0);
        for (int iz = 0; iz < n; ++iz)
            for (int iy = 0; iy < n; ++iy)
                for (int ix = 0; ix < n; ++ix) {
                    auto c = box_center(l, ix, iy, iz);
                    auto& mp = Phi[l][lin(ix, iy, iz, n)];
                    mp.order = order;
                    mp.center_x = c[0]; mp.center_y = c[1]; mp.center_z = c[2];
                    mp.M.assign(ncoef, {0.0, 0.0});
                    auto& le = Psi[l][lin(ix, iy, iz, n)];
                    le.order = order;
                    le.center_x = c[0]; le.center_y = c[1]; le.center_z = c[2];
                    le.L.assign(ncoef, {0.0, 0.0});
                }
    }

    // --- Upward pass ---
    // Step 1 (GR): P2M at the finest level.
    for (int iz = 0; iz < nfine; ++iz)
        for (int iy = 0; iy < nfine; ++iy)
            for (int ix = 0; ix < nfine; ++ix) {
                std::size_t b = lin(ix, iy, iz, nfine);
                if (box_particles[b].empty()) continue;
                occupied[levels][b] = 1;
                auto c = box_center(levels, ix, iy, iz);
                Phi[levels][b] = particles_to_multipole(
                    particles, box_particles[b], c[0], c[1], c[2], order);
            }

    // Step 2: M2M up the hierarchy.
    for (int l = levels - 1; l >= 0; --l) {
        int n = 1 << l, nc = n * 2;
        for (int iz = 0; iz < n; ++iz)
            for (int iy = 0; iy < n; ++iy)
                for (int ix = 0; ix < n; ++ix) {
                    std::size_t b = lin(ix, iy, iz, n);
                    auto c = box_center(l, ix, iy, iz);
                    for (int dz = 0; dz < 2; ++dz)
                        for (int dy = 0; dy < 2; ++dy)
                            for (int dx = 0; dx < 2; ++dx) {
                                std::size_t cb = lin(2 * ix + dx, 2 * iy + dy,
                                                     2 * iz + dz, nc);
                                if (!occupied[l + 1][cb]) continue;
                                occupied[l][b] = 1;
                                auto shifted = multipole_to_multipole(
                                    Phi[l + 1][cb], c[0], c[1], c[2]);
                                for (std::size_t i = 0; i < ncoef; ++i)
                                    Phi[l][b].M[i] += shifted.M[i];
                            }
                }
    }

    // --- Downward pass ---
    // Psi at levels 0 and 1 stays zero (no well-separated boxes). For each
    // level l >= 2: L2L the parent's Psi down, then add M2L from the
    // interaction list (children of parent's near neighbors that are not
    // near neighbors of this box).
    for (int l = 2; l <= levels; ++l) {
        int n = 1 << l, np = n / 2;
        for (int iz = 0; iz < n; ++iz)
            for (int iy = 0; iy < n; ++iy)
                for (int ix = 0; ix < n; ++ix) {
                    std::size_t b = lin(ix, iy, iz, n);
                    auto c = box_center(l, ix, iy, iz);

                    // L2L from parent (skip at l == 2: parent Psi is zero).
                    if (l > 2) {
                        std::size_t pb = lin(ix / 2, iy / 2, iz / 2, np);
                        auto shifted = local_to_local(Psi[l - 1][pb],
                                                       c[0], c[1], c[2]);
                        for (std::size_t i = 0; i < ncoef; ++i)
                            Psi[l][b].L[i] += shifted.L[i];
                    }

                    // M2L over the interaction list.
                    int px = ix / 2, py = iy / 2, pz = iz / 2;
                    for (int qz = std::max(0, pz - 1); qz <= std::min(np - 1, pz + 1); ++qz)
                        for (int qy = std::max(0, py - 1); qy <= std::min(np - 1, py + 1); ++qy)
                            for (int qx = std::max(0, px - 1); qx <= std::min(np - 1, px + 1); ++qx)
                                for (int dz = 0; dz < 2; ++dz)
                                    for (int dy = 0; dy < 2; ++dy)
                                        for (int dx = 0; dx < 2; ++dx) {
                                            int jx = 2 * qx + dx, jy = 2 * qy + dy,
                                                jz = 2 * qz + dz;
                                            // Skip near neighbors of b (and b itself).
                                            if (std::abs(jx - ix) <= 1 &&
                                                std::abs(jy - iy) <= 1 &&
                                                std::abs(jz - iz) <= 1)
                                                continue;
                                            std::size_t jb = lin(jx, jy, jz, n);
                                            if (!occupied[l][jb]) continue;
                                            auto contrib = multipole_to_local(
                                                Phi[l][jb], c[0], c[1], c[2]);
                                            for (std::size_t i = 0; i < ncoef; ++i)
                                                Psi[l][b].L[i] += contrib.L[i];
                                        }
                }
    }

    // --- Finest level: L2P + near-neighbor P2P ---
    for (int iz = 0; iz < nfine; ++iz)
        for (int iy = 0; iy < nfine; ++iy)
            for (int ix = 0; ix < nfine; ++ix) {
                std::size_t b = lin(ix, iy, iz, nfine);
                const auto& mine = box_particles[b];
                if (mine.empty()) continue;

                // L2P: far field from the accumulated local expansion.
                for (std::size_t i : mine) {
                    phi[i] += Psi[levels][b].evaluate(
                        particles[i].x, particles[i].y, particles[i].z);
                }

                // P2P: direct sum over the 27 near-neighbor boxes.
                for (int jz = std::max(0, iz - 1); jz <= std::min(nfine - 1, iz + 1); ++jz)
                    for (int jy = std::max(0, iy - 1); jy <= std::min(nfine - 1, iy + 1); ++jy)
                        for (int jx = std::max(0, ix - 1); jx <= std::min(nfine - 1, ix + 1); ++jx) {
                            const auto& theirs =
                                box_particles[lin(jx, jy, jz, nfine)];
                            for (std::size_t i : mine)
                                for (std::size_t j : theirs) {
                                    if (i == j) continue;
                                    double dx = particles[i].x - particles[j].x;
                                    double dy = particles[i].y - particles[j].y;
                                    double dz = particles[i].z - particles[j].z;
                                    phi[i] += particles[j].q /
                                              std::sqrt(dx * dx + dy * dy + dz * dz);
                                }
                        }
            }

    return phi;
}

}  // namespace fmm
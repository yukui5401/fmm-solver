#include "fmm/direct_sum.hpp"
#include <cmath>

namespace fmm {

namespace {
inline double distance(double x1, double y1, double z1,
                        double x2, double y2, double z2) {
    double dx = x1 - x2, dy = y1 - y2, dz = z1 - z2;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}
}  // namespace

std::vector<double> direct_sum(const std::vector<Particle>& particles) {
    const std::size_t n = particles.size();
    std::vector<double> phi(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double r = distance(particles[i].x, particles[i].y, particles[i].z,
                                 particles[j].x, particles[j].y, particles[j].z);
            sum += particles[j].q / r;
        }
        phi[i] = sum;
    }
    return phi;
}

std::vector<double> direct_sum_at_targets(
    const std::vector<Particle>& sources,
    const std::vector<std::array<double, 3>>& targets) {
    std::vector<double> phi(targets.size(), 0.0);

    for (std::size_t t = 0; t < targets.size(); ++t) {
        double sum = 0.0;
        for (const auto& src : sources) {
            double r = distance(targets[t][0], targets[t][1], targets[t][2],
                                 src.x, src.y, src.z);
            sum += src.q / r;
        }
        phi[t] = sum;
    }
    return phi;
}

}  // namespace fmm
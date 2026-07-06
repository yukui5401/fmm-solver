// Benchmark driver: times fmm_evaluate vs direct_sum over a range of N and
// emits CSV to stdout. For each N, several candidate refinement levels are
// tried and the fastest is kept - a single rounded level-vs-N formula
// produces misleading plateaus/steps (see docs/theory.md), since M2L cost
// is fixed per box regardless of how many particles occupy it.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "fmm/direct_sum.hpp"
#include "fmm/fmm_solver.hpp"

using namespace fmm;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main() {
    const int order = 4;

    std::printf("N,levels,t_fmm_ms,t_direct_ms,max_rel_err\n");

    for (int N : {500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000}) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> pos(-1.0, 1.0);
        std::uniform_real_distribution<double> charge(0.5, 2.0);
        std::vector<Particle> particles;
        particles.reserve(N);
        for (int i = 0; i < N; ++i)
            particles.push_back({pos(rng), pos(rng), pos(rng), charge(rng)});

        // Candidate levels spanning coarse (few, large boxes -> more P2P
        // work) to fine (many, small boxes -> more M2L overhead); levels
        // 2 has no well-separated boxes so start at what's reasonable for
        // this N and always include a floor of 2.
        int rough = std::max(2, static_cast<int>(std::round(
            std::log(std::max(1.0, static_cast<double>(N) / 24.0)) / std::log(8.0))));
        std::vector<int> candidates;
        for (int lv = std::max(2, rough - 1); lv <= rough + 1; ++lv) candidates.push_back(lv);

        double best_t_fmm = -1.0;
        int best_levels = candidates.front();
        std::vector<double> best_approx;

        for (int lv : candidates) {
            auto t0 = Clock::now();
            auto approx = fmm_evaluate(particles, order, lv);
            double t = ms_since(t0);
            if (best_t_fmm < 0.0 || t < best_t_fmm) {
                best_t_fmm = t;
                best_levels = lv;
                best_approx = std::move(approx);
            }
        }

        auto t0 = Clock::now();
        auto exact = direct_sum(particles);
        double t_direct = ms_since(t0);

        double worst = 0.0;
        for (int i = 0; i < N; ++i)
            worst = std::max(worst,
                             std::abs(best_approx[i] - exact[i]) / std::abs(exact[i]));

        std::printf("%d,%d,%.3f,%.3f,%.6e\n", N, best_levels, best_t_fmm, t_direct, worst);
        std::fflush(stdout);
    }
    return 0;
}
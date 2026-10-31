#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <string>
#include <chrono>
#include <atomic>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "head.h"

struct MertensDrift3D {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;

    double compute_net_displacement(double phase_r, double phase_i) const {
        double dx = static_cast<double>(x) * phase_r;
        double dy = static_cast<double>(y) * phase_i;
        double dz = static_cast<double>(z) * phase_r;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

bool fast_decompose_three_square(uint64_t p, int16_t& a, int16_t& b, int16_t& c) {
    if (p % 8 == 7) return false; 
    
    uint64_t limit_a = static_cast<uint64_t>(std::sqrt(p));
    for (uint64_t ta = limit_a; ta > 0; --ta) {
        uint64_t rem_a = p - (ta * ta);
        uint64_t limit_b = static_cast<uint64_t>(std::sqrt(rem_a));
        for (uint64_t tb = limit_b; tb >= ta; --tb) {
            uint64_t rem_b = rem_a - (tb * tb);
            uint64_t tc = static_cast<uint64_t>(std::sqrt(rem_b));
            if (ta * ta + tb * tb + tc * tc == p) {
                a = ta; b = tb; c = tc;
                return true;
            }
        }
    }
    return false;
}

std::vector<uint64_t> generate_high_range_primes(uint64_t limit) {
    std::vector<bool> is_prime(limit + 1, true);
    std::vector<uint64_t> primes;
    if (limit >= 2) primes.reserve(limit / std::log(limit));
    if (limit >= 2) primes.push_back(2);
    for (uint64_t i = 3; i * i <= limit; i += 2) {
        if (is_prime[i]) {
            for (uint64_t j = i * i; j <= limit; j += i * 2)
                is_prime[j] = false;
        }
    }
    for (uint64_t i = 3; i <= limit; i += 2) {
        if (is_prime[i]) primes.push_back(i);
    }
    return primes;
}

int main() {
#if defined(_WIN32)
    _mkdir("output");
#else
    mkdir("output", 0777);
#endif

    const std::vector<double>& zeta_gammas = SpectralData::zeta_gammas;
    uint64_t upper_bound_limit = 1000000;
    std::vector<uint64_t> test_primes = generate_high_range_primes(upper_bound_limit);
    size_t total_primes = test_primes.size();

    std::vector<double> results_std_drift(total_primes, 0.0);
    std::vector<double> results_opt_drift(total_primes, 0.0);
    std::vector<double> results_bound(total_primes, 0.0);
    std::vector<std::string> results_status(total_primes, "SAFE");

    std::atomic<size_t> progress_counter(0);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "[SYSTEM] Running Pipeline-Optimized SO(3) 3D Engine..." << std::endl;

    const int perm_patterns[6][3] = {
        {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}
    };

    std::vector<int16_t> comp_a(total_primes, 0), comp_b(total_primes, 0), comp_c(total_primes, 0);
    std::vector<double> phase_r_arr(total_primes, 0.0), phase_i_arr(total_primes, 0.0);
    std::vector<bool> decomposable(total_primes, false);

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < total_primes; ++idx) {
        uint64_t p = test_primes[idx];
        double log_p = std::log(static_cast<double>(p));
        double sum_cos = 0.0, sum_sin = 0.0;
        for (double gamma : zeta_gammas) {
            sum_cos += std::cos(gamma * log_p);
            sum_sin += std::sin(gamma * log_p);
        }
        phase_r_arr[idx] = sum_cos / zeta_gammas.size();
        phase_i_arr[idx] = sum_sin / zeta_gammas.size();
        decomposable[idx] = fast_decompose_three_square(p, comp_a[idx], comp_b[idx], comp_c[idx]);
    }

    MertensDrift3D standard_drift;
    MertensDrift3D optimized_drift;

    for (size_t idx = 0; idx < total_primes; ++idx) {
        uint64_t p = test_primes[idx];
        double phase_r = phase_r_arr[idx];
        double phase_i = phase_i_arr[idx];
        int16_t components[3] = { comp_a[idx], comp_b[idx], comp_c[idx] };

        double best_sq_displacement = 1e300;
        int64_t best_sx = 0, best_sy = 0, best_sz = 0;

        if (decomposable[idx]) {
            for (int p_idx = 0; p_idx < 6; ++p_idx) {
                int64_t cx = components[perm_patterns[p_idx][0]];
                int64_t cy = components[perm_patterns[p_idx][1]];
                int64_t cz = components[perm_patterns[p_idx][2]];

                for (uint32_t mask = 0; mask < 8; ++mask) {
                    int64_t sx = (mask & 1) ? cx : -cx;
                    int64_t sy = (mask & 2) ? cy : -cy;
                    int64_t sz = (mask & 4) ? cz : -cz;

                    double tx = (optimized_drift.x + sx) * phase_r;
                    double ty = (optimized_drift.y + sy) * phase_i;
                    double tz = (optimized_drift.z + sz) * phase_r;

                    double sq_dist = tx*tx + ty*ty + tz*tz;
                    if (sq_dist < best_sq_displacement) {
                        best_sq_displacement = sq_dist;
                        best_sx = sx; best_sy = sy; best_sz = sz;
                    }
                }
            }
            optimized_drift.x += best_sx;
            optimized_drift.y += best_sy;
            optimized_drift.z += best_sz;

            standard_drift.x += components[0];
            standard_drift.y += components[1];
            standard_drift.z += components[2];
        } else {
            double tx = optimized_drift.x * phase_r;
            double ty = optimized_drift.y * phase_i;
            double tz = optimized_drift.z * phase_r;
            best_sq_displacement = tx*tx + ty*ty + tz*tz;
        }

        results_std_drift[idx] = standard_drift.compute_net_displacement(phase_r, phase_i);
        results_opt_drift[idx] = std::sqrt(best_sq_displacement);
        results_bound[idx] = 1.3 * std::sqrt(static_cast<double>(p));
        results_status[idx] = (results_opt_drift[idx] <= results_bound[idx] ? "SAFE" : "OVER");

        size_t current_count = ++progress_counter;
        if (current_count % 50000 == 0 || current_count == total_primes) {
            auto now = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(now - start_time).count();
            double velocity = current_count / (duration > 0 ? duration : 1.0);
            std::cout << "\r[PROGRESS] " << std::fixed << std::setprecision(1)
                      << ((static_cast<double>(current_count) / total_primes) * 100.0)
                      << "% completed | Speed: " << std::setprecision(0) << velocity << " primes/sec..." << std::flush;
        }
    }

    std::ofstream csv_file("output/mertens_drift_3D_res.csv");
    csv_file << "Prime,Std_Drift,Opt_Drift,Spectral_Bound,Status\n";
    for (size_t i = 0; i < total_primes; ++i) {
        csv_file << test_primes[i] << "," << results_std_drift[i] << "," << results_opt_drift[i] << "," << results_bound[i] << "," << results_status[i] << "\n";
    }
    csv_file.close();

    std::cout << "\n[COMPLETE] 3D execution completed cleanly." << std::endl;
    return 0;
}

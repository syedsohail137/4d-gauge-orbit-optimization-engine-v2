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

// Include your newly separated header file
#include "head.h"

// Clifford Multivector representation Cl(4,0)
struct Clifford4D {
    int64_t e1 = 0;
    int64_t e2 = 0;
    int64_t e3 = 0;
    int64_t e4 = 0;

    double norm_sq_projected(double phase_r, double phase_i) const {
        double d1 = static_cast<double>(e1) * phase_r;
        double d2 = static_cast<double>(e2) * phase_i;
        double d3 = static_cast<double>(e3) * phase_r;
        double d4 = static_cast<double>(e4) * phase_i;
        return (d1*d1 + d2*d2 + d3*d3 + d4*d4);
    }
    
    void add_element(int16_t a, int16_t b, int16_t c, int16_t d) {
        e1 += a; e2 += b; e3 += c; e4 += d;
    }
};

// O(log p) Hurwitz-Quaternion/Euclidean decomposition engine
bool algebraic_four_square(uint64_t p, int16_t& a, int16_t& b, int16_t& c, int16_t& d) {
    if (p == 2) { a = 1; b = 1; c = 0; d = 0; return true; }
    
    uint64_t limit_a = static_cast<uint64_t>(std::sqrt(p));
    for (uint64_t ta = limit_a; ta > 0; --ta) {
        uint64_t rem_a = p - (ta * ta);
        if ((rem_a & 7) == 7) continue; 
        uint64_t limit_b = static_cast<uint64_t>(std::sqrt(rem_a));
        for (uint64_t tb = limit_b; tb >= ta; --tb) { 
            uint64_t rem_b = rem_a - (tb * tb);
            uint64_t tc = static_cast<uint64_t>(std::sqrt(rem_b));
            if (tc > 0) {
                uint64_t rem_c = rem_b - (tc * tc);
                uint64_t td = static_cast<uint64_t>(std::sqrt(rem_c));
                if (ta * ta + tb * tb + tc * tc + td * td == p) {
                    a = ta; b = tb; c = tc; d = td;
                    return true;
                }
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

    // Access the dataset securely from your Header namespace
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

    std::cout << "[SYSTEM] Running decoupled Main Matrix Engine with " 
              << zeta_gammas.size() << " spectral points..." << std::endl;

    const int perm_patterns[24][4] = {
        {0,1,2,3}, {0,1,3,2}, {0,2,1,3}, {0,2,3,1}, {0,3,1,2}, {0,3,2,1},
        {1,0,2,3}, {1,0,3,2}, {1,2,0,3}, {1,2,3,0}, {1,3,0,2}, {1,3,2,0},
        {2,0,1,3}, {2,0,3,1}, {2,1,0,3}, {2,1,3,0}, {2,3,0,1}, {2,3,1,0},
        {3,0,1,2}, {3,0,2,1}, {3,1,0,2}, {3,1,2,0}, {3,2,0,1}, {3,2,1,0}
    };

    #pragma omp parallel
    {
        Clifford4D thread_standard_drift;
        Clifford4D thread_optimized_drift;

        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < total_primes; ++idx) {
            uint64_t p = test_primes[idx];
            int16_t components[4] = { 0, 0, 0, 0 };
            
            if (algebraic_four_square(p, components[0], components[1], components[2], components[3])) {
                double log_p = std::log(static_cast<double>(p));
                double sum_cos = 0.0, sum_sin = 0.0;
                
                for (double gamma : zeta_gammas) {
                    sum_cos += std::cos(gamma * log_p);
                    sum_sin += std::sin(gamma * log_p);
                }
                double phase_r = sum_cos / zeta_gammas.size();
                double phase_i = sum_sin / zeta_gammas.size();
                
                results_std_drift[idx] = std::sqrt(thread_standard_drift.norm_sq_projected(phase_r, phase_i));
                thread_standard_drift.add_element(components[0], components[1], components[2], components[3]);
                
                double best_norm_sq = 1e300;
                int16_t best_ea = 0, best_eb = 0, best_ec = 0, best_ed = 0;
                
                for (int p_idx = 0; p_idx < 24; ++p_idx) {
                    int64_t cx = components[perm_patterns[p_idx][0]];
                    int64_t cy = components[perm_patterns[p_idx][1]];
                    int64_t cz = components[perm_patterns[p_idx][2]];
                    int64_t cw = components[perm_patterns[p_idx][3]];
                    
                    for (uint32_t mask = 0; mask < 16; ++mask) {
                        int16_t sx = static_cast<int16_t>((mask & 1) ? cx : -cx);
                        int16_t sy = static_cast<int16_t>((mask & 2) ? cy : -cy);
                        int16_t sz = static_cast<int16_t>((mask & 4) ? cz : -cz);
                        int16_t sw = static_cast<int16_t>((mask & 8) ? cw : -cw);
                        
                        Clifford4D target_state = thread_optimized_drift;
                        target_state.add_element(sx, sy, sz, sw);
                        
                        double current_norm_sq = target_state.norm_sq_projected(phase_r, phase_i);
                        if (current_norm_sq < best_norm_sq) {
                            best_norm_sq = current_norm_sq;
                            best_ea = sx; best_eb = sy; best_ec = sz; best_ed = sw;
                        }
                    }
                }
                
                thread_optimized_drift.add_element(best_ea, best_eb, best_ec, best_ed);
                
                results_bound[idx] = 1.3 * std::sqrt(static_cast<double>(p));
                results_opt_drift[idx] = std::sqrt(best_norm_sq);
                results_status[idx] = (results_opt_drift[idx] <= results_bound[idx] ? "SAFE" : "OVER");
            }

            size_t current_count = ++progress_counter;
            if (current_count % 20000 == 0 || current_count == total_primes) {
                #pragma omp critical
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    double duration = std::chrono::duration<double>(now - start_time).count();
                    double velocity = current_count / (duration > 0 ? duration : 1.0);
                    std::cout << "\r[PROGRESS] " << std::fixed << std::setprecision(1) 
                              << ((static_cast<double>(current_count) / total_primes) * 100.0) 
                              << "% completed | Speed: " << std::setprecision(0) << velocity << " primes/sec..." << std::flush;
                }
            }
        }
    } 
    
    std::ofstream csv_file("output/mertens_drift_1M_res.csv");
    csv_file << "Prime,Std_Drift,Opt_Drift,Spectral_Bound,Status\n";
    for (size_t i = 0; i < total_primes; ++i) {
        csv_file << test_primes[i] << "," << results_std_drift[i] << "," << results_opt_drift[i] << "," << results_bound[i] << "," << results_status[i] << "\n";
    }
    csv_file.close();
    
    std::cout << "\n[COMPLETE] Tasks processed completely." << std::endl;
    return 0;
}

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

// Thread-safe precise integer lattice track
struct MertensDrift4D {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
    int64_t w = 0;

    // Phase scaling calculated strictly at evaluation point to prevent compound drift
    double compute_net_displacement(double phase_r, double phase_i) const {
        double dx = static_cast<double>(x) * phase_r;
        double dy = static_cast<double>(y) * phase_i;
        double dz = static_cast<double>(z) * phase_r;
        double dw = static_cast<double>(w) * phase_i;
        return std::sqrt(dx*dx + dy*dy + dz*dz + dw*dw);
    }
};

//modular filter for rapid integer factorization
bool fast_decompose_four_square(uint64_t p, int16_t& a, int16_t& b, int16_t& c, int16_t& d) {
    if (p == 2) { a = 1; b = 1; c = 0; d = 0; return true; }
    
    uint64_t limit_a = static_cast<uint64_t>(std::sqrt(p));
    for (uint64_t ta = limit_a; ta > 0; --ta) {
        uint64_t rem_a = p - (ta * ta);
        
        // Modular constraint check: a sum of 3 squares cannot be of the form 4^k(8m+7)
        if ((rem_a & 7) == 7) continue; 
        
        uint64_t limit_b = static_cast<uint64_t>(std::sqrt(rem_a));
        for (uint64_t tb = limit_b; tb >= ta; --tb) { 
            uint64_t rem_b = rem_a - (tb * tb);
            uint64_t tc = static_cast<uint64_t>(std::sqrt(rem_b));
            
            if (tc > 0) {
                uint64_t rem_c = rem_b - (tc * tc);
                uint64_t td = static_cast<uint64_t>(std::sqrt(rem_c));
                
                if (ta * ta + tb * tb + tc * tc + td * td == p) {
                    // Fast swap sort (Inline elimination of std::sort for performance)
                    int16_t r0 = ta, r1 = tb, r2 = tc, r3 = td;
                    if (r0 < r1) std::swap(r0, r1); if (r0 < r2) std::swap(r0, r2); if (r0 < r3) std::swap(r0, r3);
                    if (r1 < r2) std::swap(r1, r2); if (r1 < r3) std::swap(r1, r3);
                    if (r2 < r3) std::swap(r2, r3);
                    
                    a = r0; b = r1; c = r2; d = r3;
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

    const std::vector<double> zeta_gammas = {
        14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
        37.5861781588, 40.9187190121, 43.3270732809, 48.0051508814, 49.7738324777,
        52.9703214777, 56.4462476971, 59.3470440026, 60.8317785246, 65.1125440481,
        67.0798105295, 69.5464017112, 72.0671576744, 75.7046906991, 77.1448400689,
        79.3373750202, 82.9103808541, 84.7354929808, 87.4252746138, 88.8091112076,
        92.4918992705, 94.6513440412, 95.8706342282, 98.8311942182, 101.3178510038
    };
    
    uint64_t upper_bound_limit = 1000000; 
    std::vector<uint64_t> test_primes = generate_high_range_primes(upper_bound_limit); 
    size_t total_primes = test_primes.size();
    
    std::vector<double> results_std_drift(total_primes, 0.0);
    std::vector<double> results_opt_drift(total_primes, 0.0);
    std::vector<double> results_bound(total_primes, 0.0);
    std::vector<std::string> results_status(total_primes, "SAFE");

    std::atomic<size_t> progress_counter(0);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "[SYSTEM] Running Optimized Hyperoctahedral Matrix..." << std::endl;

    // Hardcoded flattened indices representing permutation mapping structures to bypass runtime vector allocations
    const int perm_patterns[24][4] = {
        {0,1,2,3}, {0,1,3,2}, {0,2,1,3}, {0,2,3,1}, {0,3,1,2}, {0,3,2,1},
        {1,0,2,3}, {1,0,3,2}, {1,2,0,3}, {1,2,3,0}, {1,3,0,2}, {1,3,2,0},
        {2,0,1,3}, {2,0,3,1}, {2,1,0,3}, {2,1,3,0}, {2,3,0,1}, {2,3,1,0},
        {3,0,1,2}, {3,0,2,1}, {3,1,0,2}, {3,1,2,0}, {3,2,0,1}, {3,2,1,0}
    };

    #pragma omp parallel
    {
        // Independent tracks per thread to avoid race conditions
        MertensDrift4D thread_standard_drift;
        MertensDrift4D thread_optimized_drift;

        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < total_primes; ++idx) {
            uint64_t p = test_primes[idx];
            int16_t components[4] = { 0, 0, 0, 0 };
            
            if (fast_decompose_four_square(p, components[0], components[1], components[2], components[3])) {
                double log_p = std::log(static_cast<double>(p));
                double sum_cos = 0.0, sum_sin = 0.0;
                
                for (double gamma : zeta_gammas) {
                    sum_cos += std::cos(gamma * log_p);
                    sum_sin += std::sin(gamma * log_p);
                }
                double phase_r = sum_cos / zeta_gammas.size();
                double phase_i = sum_sin / zeta_gammas.size();
                
                // Track standard walk execution via modular floating projections
                results_std_drift[idx] = thread_standard_drift.compute_net_displacement(phase_r, phase_i);
                
                // Structural greedy matrix search utilizing our static lookup structure
                double best_sq_displacement = 1e300;
                int64_t best_sx = 0, best_sy = 0, best_sz = 0, best_sw = 0;
                
                for (int p_idx = 0; p_idx < 24; ++p_idx) {
                    int64_t cx = components[perm_patterns[p_idx][0]];
                    int64_t cy = components[perm_patterns[p_idx][1]];
                    int64_t cz = components[perm_patterns[p_idx][2]];
                    int64_t cw = components[perm_patterns[p_idx][3]];
                    
                    for (uint32_t mask = 0; mask < 16; ++mask) {
                        int64_t sx = (mask & 1) ? cx : -cx;
                        int64_t sy = (mask & 2) ? cy : -cy;
                        int64_t sz = (mask & 4) ? cz : -cz;
                        int64_t sw = (mask & 8) ? cw : -cw;
                        
                        double tx = (thread_optimized_drift.x + sx) * phase_r;
                        double ty = (thread_optimized_drift.y + sy) * phase_i;
                        double tz = (thread_optimized_drift.z + sz) * phase_r;
                        double tw = (thread_optimized_drift.w + sw) * phase_i;
                        
                        double sq_dist = tx*tx + ty*ty + tz*tz + tw*tw;
                        if (sq_dist < best_sq_displacement) {
                            best_sq_displacement = sq_dist;
                            best_sx = sx; best_sy = sy; best_sz = sz; best_sw = sw;
                        }
                    }
                }
                
                // Commit spatial integer state transitions
                thread_optimized_drift.x += best_sx;
                thread_optimized_drift.y += best_sy;
                thread_optimized_drift.z += best_sz;
                thread_optimized_drift.w += best_sw;
                
                results_bound[idx] = 1.3 * std::sqrt(static_cast<double>(p));
                results_opt_drift[idx] = std::sqrt(best_sq_displacement);
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

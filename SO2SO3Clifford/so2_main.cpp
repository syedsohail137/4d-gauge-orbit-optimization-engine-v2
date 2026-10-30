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

// Thread-safe precise integer lattice track scaled to a 2D SO(2) projection plane
struct MertensDrift2D {
    int64_t x = 0;
    int64_t y = 0;

    // Phase scaling calculated strictly at evaluation point to prevent compound drift
    double compute_net_displacement(double phase_r, double phase_i) const {
        double dx = static_cast<double>(x) * phase_r;
        double dy = static_cast<double>(y) * phase_i;
        return std::sqrt(dx*dx + dy*dy);
    }
};

// Fermat-inspired modular filter for rapid two-square integer decomposition
// Returns true if p can be written as a^2 + b^2, false otherwise (e.g., if p = 3 mod 4)
bool fast_decompose_two_square(uint64_t p, int16_t& a, int16_t& b) {
    if (p == 2) { a = 1; b = 1; return true; }
    // Fermat's Christmas Theorem: Primes of the form 4k+3 cannot be a sum of two squares
    if ((p & 3) == 3) { a = 0; b = 0; return false; }
    
    uint64_t limit_a = static_cast<uint64_t>(std::sqrt(p));
    for (uint64_t ta = limit_a; ta > 0; --ta) {
        uint64_t rem_a = p - (ta * ta);
        uint64_t tb = static_cast<uint64_t>(std::sqrt(rem_a));
        
        if (ta * ta + tb * tb == p) {
            int16_t r0 = ta, r1 = tb;
            if (r0 < r1) std::swap(r0, r1); // Inline descending sort
            
            a = r0; b = r1;
            return true;
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

    std::cout << "[SYSTEM] Running Optimized SO(2) 2D Grid Matrix..." << std::endl;

    // Hardcoded patterns mapping permutations for 2 dimensions (2! = 2 combinations)
    const int perm_patterns[2][2] = {
        {0, 1}, {1, 0}
    };

    #pragma omp parallel
    {
        // Independent tracks per thread to avoid race conditions
        MertensDrift2D thread_standard_drift;
        MertensDrift2D thread_optimized_drift;

        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < total_primes; ++idx) {
            uint64_t p = test_primes[idx];
            int16_t components[2] = { 0, 0 };
            
            // Extract modular phase balances regardless of mathematical breakdown criteria
            double log_p = std::log(static_cast<double>(p));
            double sum_cos = 0.0, sum_sin = 0.0;
            
            for (double gamma : zeta_gammas) {
                sum_cos += std::cos(gamma * log_p);
                sum_sin += std::sin(gamma * log_p);
            }
            double phase_r = sum_cos / zeta_gammas.size();
            double phase_i = sum_sin / zeta_gammas.size();
            
            int64_t best_sx = 0, best_sy = 0;
            double best_sq_displacement = 1e300;

            if (fast_decompose_two_square(p, components[0], components[1])) {
                // Structural greedy matrix search looking through 8 discrete combinations (2 permutations * 4 sign shifts)
                for (int p_idx = 0; p_idx < 2; ++p_idx) {
                    int64_t cx = components[perm_patterns[p_idx][0]];
                    int64_t cy = components[perm_patterns[p_idx][1]];
                    
                    for (uint32_t mask = 0; mask < 4; ++mask) {
                        int64_t sx = (mask & 1) ? cx : -cx;
                        int64_t sy = (mask & 2) ? cy : -cy;
                        
                        double tx = (thread_optimized_drift.x + sx) * phase_r;
                        double ty = (thread_optimized_drift.y + sy) * phase_i;
                        
                        double sq_dist = tx*tx + ty*ty;
                        if (sq_dist < best_sq_displacement) {
                            best_sq_displacement = sq_dist;
                            best_sx = sx; best_sy = sy;
                        }
                    }
                }
                
                // Commit lattice step displacement jumps
                thread_optimized_drift.x += best_sx;
                thread_optimized_drift.y += best_sy;
                
                // Track standard walk baseline changes
                // In standard baseline step execution, we pick the default unshifted orientation path
                thread_standard_drift.x += components[0];
                thread_standard_drift.y += components[1];
            } else {
                // If the prime cannot be broken down into 2 squares (4k+3 primes), the lattice path stands still.
                // The current squared distance is updated based on its stationary position under new phase matrices.
                double tx = thread_optimized_drift.x * phase_r;
                double ty = thread_optimized_drift.y * phase_i;
                best_sq_displacement = tx*tx + ty*ty;
            }

            results_std_drift[idx] = thread_standard_drift.compute_net_displacement(phase_r, phase_i);
            results_opt_drift[idx] = std::sqrt(best_sq_displacement);
            results_bound[idx] = 1.3 * std::sqrt(static_cast<double>(p));
            results_status[idx] = (results_opt_drift[idx] <= results_bound[idx] ? "SAFE" : "OVER");

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

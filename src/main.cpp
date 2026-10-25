#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <string>

struct MertensDrift4D {
    double cumulative_x = 0.0;
    double cumulative_y = 0.0;
    double cumulative_z = 0.0;
    double cumulative_w = 0.0;

    double compute_net_displacement() const {
        return std::sqrt(cumulative_x * cumulative_x + 
                         cumulative_y * cumulative_y + 
                         cumulative_z * cumulative_z + 
                         cumulative_w * cumulative_w);
    }
};

bool fast_decompose_four_square(uint64_t p, int16_t& a, int16_t& b, int16_t& c, int16_t& d) {
    if (p == 2) { a = 1; b = 1; c = 0; d = 0; return true; }
    
    int64_t ta = static_cast<int64_t>(std::sqrt(p));
    for (; ta >= 0; --ta) {
        int64_t rem_a = p - (ta * ta);
        int64_t tb = static_cast<int64_t>(std::sqrt(rem_a));
        for (; tb >= 0; --tb) {
            int64_t rem_b = rem_a - (tb * tb);
            int64_t tc = static_cast<int64_t>(std::sqrt(rem_b));
            for (; tc >= 0; --tc) {
                int64_t rem_c = rem_b - (tc * tc);
                int64_t td = static_cast<int64_t>(std::round(std::sqrt(rem_c)));
                
                if (ta * ta + tb * tb + tc * tc + td * td == p) {
                    int16_t coords[4] = { static_cast<int16_t>(ta), static_cast<int16_t>(tb), 
                                          static_cast<int16_t>(tc), static_cast<int16_t>(td) };
                    std::sort(coords, coords + 4, std::greater<int16_t>());
                    a = coords[0]; b = coords[1]; c = coords[2]; d = coords[3];
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
    const std::vector<double> zeta_gammas = {
        14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
        37.5861781588, 40.9187190121, 43.3270732809, 48.0051508814, 49.7738324777,
        52.9703214777, 56.4462476971, 59.3470440026, 60.8317785246, 65.1125440481,
        67.0798105295, 69.5464017112, 72.0671576744, 75.7046906991, 77.1448400689,
        79.3373750202, 82.9103808541, 84.7354929808, 87.4252746138, 88.8091112076,
        92.4918992705, 94.6513440412, 95.8706342282, 98.8311942182, 101.3178510038
    };
    
    std::cout << "[SYSTEM] Initializing Sieve Array & Resolving Phase Fields..." << std::endl;
    std::vector<uint64_t> test_primes = generate_high_range_primes(2000); 
    
    MertensDrift4D standard_drift;
    MertensDrift4D optimized_drift;
    
    std::string csv_filename = "mertens_drift_high_res.csv";
    std::ofstream csv_file(csv_filename);
    csv_file << "Prime,Std_Drift,Opt_Drift,Spectral_Bound,Status\n";
    
    std::cout << "[HIGH-RES MATRIX RUNNING]" << std::endl;
    std::cout << std::setw(8) << "Prime" << " | " << std::setw(12) << "Std Drift" << " | " << std::setw(12) << "Opt Drift" << " | " << std::setw(12) << "Bound" << " | Status" << std::endl;
    
    for (uint64_t p : test_primes) {
        int16_t a = 0, b = 0, c = 0, d = 0;
        if (fast_decompose_four_square(p, a, b, c, d)) {
            double log_p = std::log(static_cast<double>(p));
            double sum_cos = 0.0, sum_sin = 0.0;
            
            for (double gamma : zeta_gammas) {
                sum_cos += std::cos(gamma * log_p);
                sum_sin += std::sin(gamma * log_p);
            }
            double phase_r = sum_cos / zeta_gammas.size();
            double phase_i = sum_sin / zeta_gammas.size();
            
            standard_drift.cumulative_x += a * phase_r;
            standard_drift.cumulative_y += b * phase_i;
            standard_drift.cumulative_z += c * phase_r;
            standard_drift.cumulative_w += d * phase_i;
            
            double best_step_drift = 1e100;
            int16_t best_a = a, best_b = b, best_c = c, best_d = d;
            int16_t signs[2] = {-1, 1};
            
            for (int16_t sa : signs) {
                for (int16_t sb : signs) {
                    for (int16_t sc : signs) {
                        for (int16_t sd : signs) {
                            double tx = optimized_drift.cumulative_x + (a * sa) * phase_r;
                            double ty = optimized_drift.cumulative_y + (b * sb) * phase_i;
                            double tz = optimized_drift.cumulative_z + (c * sc) * phase_r;
                            double tw = optimized_drift.cumulative_w + (d * sd) * phase_i;
                            double td = std::sqrt(tx*tx + ty*ty + tz*tz + tw*tw);
                            if (td < best_step_drift) {
                                best_step_drift = td;
                                best_a = a * sa; best_b = b * sb; best_c = c * sc; best_d = d * sd;
                            }
                        }
                    }
                }
            }
            
            optimized_drift.cumulative_x += best_a * phase_r;
            optimized_drift.cumulative_y += best_b * phase_i;
            optimized_drift.cumulative_z += best_c * phase_r;
            optimized_drift.cumulative_w += best_d * phase_i;
            
            double bound = std::sqrt(static_cast<double>(p));
            double opt_d = optimized_drift.compute_net_displacement();
            double std_d = standard_drift.compute_net_displacement();
            std::string status = (opt_d <= bound ? "SAFE" : "OVER");
            
            if (p % 11 == 0 || p < 20) {
                std::cout << std::setw(8) << p << " | "
                          << std::setw(12) << std::fixed << std::setprecision(4) << std_d << " | "
                          << std::setw(12) << opt_d << " | "
                          << std::setw(12) << bound << " | "
                          << status << std::endl;
            }
            csv_file << p << "," << std_d << "," << opt_d << "," << bound << "," << status << "\n";
        }
    }
    csv_file.close();
    return 0;
}

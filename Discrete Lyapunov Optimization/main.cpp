#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <string>
#include <chrono>

// Tracks 4D vector trajectories within the discrete dynamical system
struct LyapunovState4D {
    double state_x = 0.0;
    double state_y = 0.0;
    double state_z = 0.0;
    double state_w = 0.0;

    // Evaluates the current scalar Lyapunov energy candidate V(x) = ||W_n||_2
    double compute_lyapunov_energy() const {
        return std::sqrt(state_x * state_x + 
                         state_y * state_y + 
                         state_z * state_z + 
                         state_w * state_w);
    }
};

struct ControlStep4D {
    double dx, dy, dz, dw;
};

// Maps integers to 4D coordinate fields using four-square decomposition
bool fast_decompose_four_square(uint64_t p, int16_t& a, int16_t& b, int16_t& c, int16_t& d) {
    if (p == 2) { a = 1; b = 1; c = 0; d = 0; return true; }
    
    uint64_t limit_a = static_cast<uint64_t>(std::sqrt(p));
    for (uint64_t ta = limit_a; ta > 0; --ta) {
        uint64_t rem_a = p - (ta * ta);
        uint64_t limit_b = static_cast<uint64_t>(std::sqrt(rem_a));
        
        for (uint64_t tb = limit_b; tb >= ta; --tb) { 
            uint64_t rem_b = rem_a - (tb * tb);
            uint64_t tc = static_cast<uint64_t>(std::sqrt(rem_b));
            
            while (tc > 0) {
                uint64_t rem_c = rem_b - (tc * tc);
                uint64_t td = static_cast<uint64_t>(std::sqrt(rem_c));
                if (ta * ta + tb * tb + tc * tc + td * td == p) {
                    int16_t coords[4] = { static_cast<int16_t>(ta), static_cast<int16_t>(tb), 
                                          static_cast<int16_t>(tc), static_cast<int16_t>(td) };
                    std::sort(coords, coords + 4, std::greater<int16_t>());
                    a = coords[0]; b = coords[1]; c = coords[2]; d = coords[3];
                    return true;
                }
                --tc;
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
    const std::vector<double> modulation_frequencies = {
        14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
        37.5861781588, 40.9187190121, 43.3270732809, 48.0051508814, 49.7738324777,
        52.9703214777, 56.4462476971, 59.3470440026, 60.8317785246, 65.1125440481,
        67.0798105295, 69.5464017112, 72.0671576744, 75.7046906991, 77.1448400689,
        79.3373750202, 82.9103808541, 84.7354929808, 87.4252746138, 88.8091112076,
        92.4918992705, 94.6513440412, 95.8706342282, 98.8311942182, 101.3178510038
    };
    
    // Lowered limit to fit perfectly inside the online sandbox execution timeout window
    uint64_t upper_bound_limit = 5000; 
    std::cout << "[SYSTEM] Initializing Eratosthenes Sieve fields..." << std::endl;
    std::vector<uint64_t> test_primes = generate_high_range_primes(upper_bound_limit); 
    size_t total_primes = test_primes.size();
    
    std::cout << "[SYSTEM] Found " << total_primes << " vector seeds up to " << upper_bound_limit << "." << std::endl;
    std::cout << "\nLaunching 384-Hyperoctahedral Lyapunov Control Matrix...\n" << std::endl;
    std::cout << std::left << std::setw(12) << "VectorSeed" 
              << std::setw(22) << "Energy_Uncontrolled" 
              << std::setw(20) << "Energy_Controlled" 
              << std::setw(22) << "Stability_Threshold" 
              << "Status" << std::endl;
    std::cout << std::string(85, '-') << std::endl;

    LyapunovState4D uncontrolled_state;
    LyapunovState4D controlled_state;

    for (size_t idx = 0; idx < total_primes; ++idx) {
        uint64_t p = test_primes[idx];
        int16_t a = 0, b = 0, c = 0, d = 0;
        
        if (fast_decompose_four_square(p, a, b, c, d)) {
            double log_p = std::log(static_cast<double>(p));
            double sum_cos = 0.0, sum_sin = 0.0;
            
            for (double freq : modulation_frequencies) {
                sum_cos += std::cos(freq * log_p);
                sum_sin += std::sin(freq * log_p);
            }
            double phase_r = sum_cos / modulation_frequencies.size();
            double phase_i = sum_sin / modulation_frequencies.size();
            
            uncontrolled_state.state_x += a * phase_r;
            uncontrolled_state.state_y += b * phase_i;
            uncontrolled_state.state_z += c * phase_r;
            uncontrolled_state.state_w += d * phase_i;
            
            int16_t perm[4] = { a, b, c, d };
            std::sort(perm, perm + 4);
            
            std::vector<ControlStep4D> unique_perms;
            unique_perms.reserve(24);
            do {
                unique_perms.push_back({
                    static_cast<double>(perm[0]) * phase_r,
                    static_cast<double>(perm[1]) * phase_i,
                    static_cast<double>(perm[2]) * phase_r,
                    static_cast<double>(perm[3]) * phase_i
                });
            } while (std::next_permutation(perm, perm + 4));
            
            double minimum_energy_candidate = 1e300;
            ControlStep4D optimal_actuator_step = { 0.0, 0.0, 0.0, 0.0 };
            
            for (const auto& candidate : unique_perms) {
                for (uint32_t mask = 0; mask < 16; ++mask) {
                    double sx = (mask & 1) ? candidate.dx : -candidate.dx;
                    double sy = (mask & 2) ? candidate.dy : -candidate.dy;
                    double sz = (mask & 4) ? candidate.dz : -candidate.dz;
                    double sw = (mask & 8) ? candidate.dw : -candidate.dw;
                    
                    double tx = controlled_state.state_x + sx;
                    double ty = controlled_state.state_y + sy;
                    double tz = controlled_state.state_z + sz;
                    double tw = controlled_state.state_w + sw;
                    
                    double next_energy_sq = tx*tx + ty*ty + tz*tz + tw*tw;
                    if (next_energy_sq < minimum_energy_candidate) {
                        minimum_energy_candidate = next_energy_sq;
                        optimal_actuator_step = { sx, sy, sz, sw };
                    }
                }
            }
            
            controlled_state.state_x += optimal_actuator_step.dx;
            controlled_state.state_y += optimal_actuator_step.dy;
            controlled_state.state_z += optimal_actuator_step.dz;
            controlled_state.state_w += optimal_actuator_step.dw;
            
            double thresh = std::sqrt(static_cast<double>(p));
            double e_ctrl = controlled_state.compute_lyapunov_energy();
            double e_unctrl = uncontrolled_state.compute_lyapunov_energy();
            std::string status = (e_ctrl <= thresh ? "STABLE" : "UNSTABLE");

            // Print every 20th prime step to keep the console readable without interface lag
            if (idx % 20 == 0 || idx == total_primes - 1) {
                std::cout << std::left << std::setw(12) << p 
                          << std::setw(22) << std::fixed << std::setprecision(4) << e_unctrl 
                          << std::setw(20) << e_ctrl 
                          << std::setw(22) << thresh 
                          << status << std::setprecision(0) << std::endl;
            }
        }
    } 
    
    std::cout << std::string(85, '-') << std::endl;
    std::cout << "[COMPLETE] Stability evaluation stream printed to console successfully." << std::endl;
    return 0;
}

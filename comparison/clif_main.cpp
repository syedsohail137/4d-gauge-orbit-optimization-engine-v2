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

struct Clifford4D {
    int64_t e1 = 0; int64_t e2 = 0; int64_t e3 = 0; int64_t e4 = 0;

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

bool fast_decompose_four_square(uint64_t p, int16_t& a, int16_t& b, int16_t& c, int16_t& d) {
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
            for (uint64_t j = i * i; j <= limit; j += i * 2) is_prime[j] = false;
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
    
    std::vector<double> res_track_A(total_primes, 0.0);
    std::vector<double> res_track_B(total_primes, 0.0);
    std::vector<double> res_track_C(total_primes, 0.0);
    std::vector<double> results_bound(total_primes, 0.0);

    std::vector<int16_t> comp_a(total_primes, 0), comp_b(total_primes, 0), comp_c(total_primes, 0), comp_d(total_primes, 0);
    std::vector<int16_t> rand_a(total_primes, 0), rand_b(total_primes, 0), rand_c(total_primes, 0), rand_d(total_primes, 0);
    std::vector<double> phase_r_arr(total_primes, 0.0), phase_i_arr(total_primes, 0.0);
    std::vector<bool> decomposable(total_primes, false);

    std::cout << "[SYSTEM] Phase 1: Processing concurrent multi-track maps..." << std::endl;

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
        
        decomposable[idx] = fast_decompose_four_square(p, comp_a[idx], comp_b[idx], comp_c[idx], comp_d[idx]);

        uint64_t seed = p * 6364136223846793005ULL + 1442695040888963407ULL;
        rand_a[idx] = static_cast<int16_t>((seed % 120) + 1);
        rand_b[idx] = static_cast<int16_t>(((seed >> 4) % 120) + 1);
        rand_c[idx] = static_cast<int16_t>(((seed >> 8) % 120) + 1);
        rand_d[idx] = static_cast<int16_t>(((seed >> 12) % 120) + 1);
    }

    std::cout << "[SYSTEM] Phase 2: Executing deterministic tracking loop..." << std::endl;

    const int perm_patterns[24][4] = {
        {0,1,2,3}, {0,1,3,2}, {0,2,1,3}, {0,2,3,1}, {0,3,1,2}, {0,3,2,1},
        {1,0,2,3}, {1,0,3,2}, {1,2,0,3}, {1,2,3,0}, {1,3,0,2}, {1,3,2,0},
        {2,0,1,3}, {2,0,3,1}, {2,1,0,3}, {2,1,3,0}, {2,3,0,1}, {2,3,1,0},
        {3,0,1,2}, {3,0,2,1}, {3,1,0,2}, {3,1,2,0}, {3,2,0,1}, {3,2,1,0}
    };

    Clifford4D track_A; 
    Clifford4D track_B; 
    Clifford4D track_C; 

    for (size_t idx = 0; idx < total_primes; ++idx) {
        uint64_t p = test_primes[idx];
        double pr = phase_r_arr[idx];
        double pi = phase_i_arr[idx];

        if (decomposable[idx]) {
            track_A.add_element(comp_a[idx], comp_b[idx], comp_c[idx], comp_d[idx]);

            double b_best_norm = 1e300;
            int16_t b_ea = 0, b_eb = 0, b_ec = 0, b_ed = 0;
            int16_t b_comps[4] = { comp_a[idx], comp_b[idx], comp_c[idx], comp_d[idx] };

            for (int p_idx = 0; p_idx < 24; ++p_idx) {
                int64_t cx = b_comps[perm_patterns[p_idx][0]];
                int64_t cy = b_comps[perm_patterns[p_idx][1]];
                int64_t cz = b_comps[perm_patterns[p_idx][2]];
                int64_t cw = b_comps[perm_patterns[p_idx][3]];
                for (uint32_t mask = 0; mask < 16; ++mask) {
                    int16_t sx = static_cast<int16_t>((mask & 1) ? cx : -cx);
                    int16_t sy = static_cast<int16_t>((mask & 2) ? cy : -cy);
                    int16_t sz = static_cast<int16_t>((mask & 4) ? cz : -cz);
                    int16_t sw = static_cast<int16_t>((mask & 8) ? cw : -cw);
                    Clifford4D target = track_B; target.add_element(sx, sy, sz, sw);
                    double n_sq = target.norm_sq_projected(pr, pi);
                    if (n_sq < b_best_norm) { b_best_norm = n_sq; b_ea = sx; b_eb = sy; b_ec = sz; b_ed = sw; }
                }
            }
            track_B.add_element(b_ea, b_eb, b_ec, b_ed);
        }

        double c_best_norm = 1e300;
        int16_t c_ea = 0, c_eb = 0, c_ec = 0, c_ed = 0;
        int16_t c_comps[4] = { rand_a[idx], rand_b[idx], rand_c[idx], rand_d[idx] };

        for (int p_idx = 0; p_idx < 24; ++p_idx) {
            int64_t cx = c_comps[perm_patterns[p_idx][0]];
            int64_t cy = c_comps[perm_patterns[p_idx][1]];
            int64_t cz = c_comps[perm_patterns[p_idx][2]];
            int64_t cw = c_comps[perm_patterns[p_idx][3]];
            for (uint32_t mask = 0; mask < 16; ++mask) {
                int16_t sx = static_cast<int16_t>((mask & 1) ? cx : -cx);
                int16_t sy = static_cast<int16_t>((mask & 2) ? cy : -cy);
                int16_t sz = static_cast<int16_t>((mask & 4) ? cz : -cz);
                int16_t sw = static_cast<int16_t>((mask & 8) ? cw : -cw);
                Clifford4D target = track_C; target.add_element(sx, sy, sz, sw);
                double n_sq = target.norm_sq_projected(pr, pi);
                if (n_sq < c_best_norm) { c_best_norm = n_sq; c_ea = sx; c_eb = sy; c_ec = sz; c_ed = sw; }
            }
        }
        track_C.add_element(c_ea, c_eb, c_ec, c_ed);

        res_track_A[idx] = std::sqrt(track_A.norm_sq_projected(pr, pi));
        res_track_B[idx] = std::sqrt(track_B.norm_sq_projected(pr, pi));
        res_track_C[idx] = std::sqrt(track_C.norm_sq_projected(pr, pi));
        results_bound[idx] = 1.3 * std::sqrt(static_cast<double>(p));
    }

    std::ofstream csv_file("output/mertens_comparative_res.csv");
    csv_file << "Prime,Track_A_Raw,Track_B_Greedy,Track_C_Synthetic,Spectral_Bound\n";
    for (size_t i = 0; i < total_primes; ++i) {
        csv_file << test_primes[i] << "," << res_track_A[i] << "," << res_track_B[i] << "," << res_track_C[i] << "," << results_bound[i] << "\n";
    }
    csv_file.close();
    std::cout << "[COMPLETE] Metrics compiled to output/mertens_comparative_res.csv." << std::endl;
    return 0;
}

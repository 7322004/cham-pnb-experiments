// cham_idod_scan_rounds1to40_keysMedian.cpp
//
// CHAM1.cpp と同一仕様の IDOD スキャン（rounds=1..40, id_bit=0..63, od_bit=0..63）を行うが、
// 鍵方向の集計は「平均」ではなく「中央値 (median)」を採用する版。
//
// 出力: cham_idod_scan_rounds1to40_keysMedian.csv
//  列: rounds,samples_per_key,keys,id_bit,od_bit,p_same_median,epsilon_median
//
// Build:
//   g++ -O3 -std=c++17 -march=native cham_idod_scan_rounds1to40_keysMedian.cpp -o cham_scan_median
//   (OpenMP を使う場合)
//   g++ -O3 -std=c++17 -march=native -fopenmp cham_idod_scan_rounds1to40_keysMedian.cpp -o cham_scan_median
//
// Notes:
// - p_same_key = Pr[ΔOD(od_bit)=0] を各鍵ごとに NUM_SAMPLES サンプルで測定し、
//   その {p_same_key} の中央値を p_same_median として出力する。
// - NUM_KEYS が偶数のときの中央値は「中央2点の平均」を採用。

#include <cstdint>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <chrono>
#include <array>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// ===== 実験パラメータ（必要に応じて書き換え） =====
constexpr int ROUNDS_MIN = 1;
constexpr int ROUNDS_MAX = 40;

constexpr std::uint64_t NUM_SAMPLES = (1ULL << 20);

constexpr int ID_BIT_START = 0;
constexpr int ID_BIT_END = 63;

// 平文生成のベースシード（rounds, id, key で派生させる）
constexpr std::uint64_t RNG_SEED_BASE = 0x20251203ULL;

// 使う鍵の個数（鍵中央値）
constexpr std::uint32_t NUM_KEYS = 64;                 // 例: 2^6
constexpr std::uint64_t KEY_SEED_BASE = 0xC0FFEEULL;   // 鍵生成用seed（固定で再現性）

const char* const OUTPUT_CSV = "cham_idod_scan_rounds1to40_keysMedian.csv";

// ===== CHAM-64/128 実装部分 =====

inline std::uint16_t rotl16(std::uint16_t x, int n) {
    n &= 15;
    if (n == 0) return x;
    return static_cast<std::uint16_t>(
        (static_cast<std::uint32_t>(x) << n) |
        (static_cast<std::uint32_t>(x) >> (16 - n))
        );
}

void cham64_key_schedule(const std::uint16_t key[8], std::uint16_t round_keys[16]) {
    for (int i = 0; i < 8; ++i) {
        std::uint16_t k = key[i];
        std::uint16_t tmp1 = rotl16(k, 1);
        std::uint16_t tmp8 = rotl16(k, 8);
        std::uint16_t tmp11 = rotl16(k, 11);

        round_keys[i] = static_cast<std::uint16_t>(k ^ tmp1 ^ tmp8);

        int idx = ((i + 8) ^ 1) & 0xF;
        round_keys[idx] = static_cast<std::uint16_t>(k ^ tmp1 ^ tmp11);
    }
}

void cham64_encrypt_rounds(const std::uint16_t round_keys[16],
    const std::uint16_t plain[4],
    std::uint16_t out[4],
    int rounds) {
    std::uint16_t x = plain[0];
    std::uint16_t y = plain[1];
    std::uint16_t z = plain[2];
    std::uint16_t w = plain[3];

    for (int i = 0; i < rounds; ++i) {
        int alpha, beta;
        if ((i & 1) == 0) { alpha = 1; beta = 8; }
        else { alpha = 8; beta = 1; }

        std::uint16_t t = static_cast<std::uint16_t>(x ^ static_cast<std::uint16_t>(i));
        std::uint16_t u = static_cast<std::uint16_t>(rotl16(y, alpha) ^ round_keys[i & 0xF]);

        std::uint32_t sum = static_cast<std::uint32_t>(t) + static_cast<std::uint32_t>(u);
        std::uint16_t v = static_cast<std::uint16_t>(sum & 0xFFFFu);
        v = rotl16(v, beta);

        std::uint16_t nx = y;
        std::uint16_t ny = z;
        std::uint16_t nz = w;
        std::uint16_t nw = v;

        x = nx; y = ny; z = nz; w = nw;
    }

    out[0] = x; out[1] = y; out[2] = z; out[3] = w;
}

inline std::uint64_t pack_state64(const std::uint16_t s[4]) {
    std::uint64_t v = 0;
    v |= static_cast<std::uint64_t>(s[0]);
    v |= static_cast<std::uint64_t>(s[1]) << 16;
    v |= static_cast<std::uint64_t>(s[2]) << 32;
    v |= static_cast<std::uint64_t>(s[3]) << 48;
    return v;
}

inline void unpack_state64(std::uint64_t v, std::uint16_t s[4]) {
    s[0] = static_cast<std::uint16_t>(v & 0xFFFFu);
    s[1] = static_cast<std::uint16_t>((v >> 16) & 0xFFFFu);
    s[2] = static_cast<std::uint16_t>((v >> 32) & 0xFFFFu);
    s[3] = static_cast<std::uint16_t>((v >> 48) & 0xFFFFu);
}

// ===== 64-bit: ctz (set bit index) =====
static inline int ctz64(std::uint64_t x) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx);
#else
    return static_cast<int>(__builtin_ctzll(x));
#endif
}

// ===== median helper (counts -> p_same median) =====
static inline double median_from_counts(const std::vector<std::uint64_t>& counts_sorted,
    std::uint64_t samples_per_key) {
    const std::size_t n = counts_sorted.size();
    if (n == 0) return 0.0;
    if (n & 1U) {
        // odd
        return static_cast<double>(counts_sorted[n / 2]) / static_cast<double>(samples_per_key);
    }
    else {
        // even: average of two middle
        const std::uint64_t a = counts_sorted[n / 2 - 1];
        const std::uint64_t b = counts_sorted[n / 2];
        const double m = (static_cast<double>(a) + static_cast<double>(b)) * 0.5;
        return m / static_cast<double>(samples_per_key);
    }
}

// ===== main =====
int main() {
    using clock = std::chrono::steady_clock;
    auto t_start = clock::now();

    if (ID_BIT_START < 0 || ID_BIT_END > 63 || ID_BIT_START > ID_BIT_END) {
        std::cerr << "ID_BIT_START / ID_BIT_END の設定が不正です。\n";
        return 1;
    }
    if (NUM_KEYS == 0) {
        std::cerr << "NUM_KEYS must be >= 1\n";
        return 1;
    }

    std::ofstream ofs(OUTPUT_CSV);
    if (!ofs) {
        std::cerr << "出力ファイルを開けません: " << OUTPUT_CSV << "\n";
        return 1;
    }

    ofs << "rounds,samples_per_key,keys,id_bit,od_bit,p_same_median,epsilon_median\n";

    // ---- ランダム鍵をNUM_KEYS個生成し、round_keysも事前計算 ----
    std::mt19937_64 key_rng(KEY_SEED_BASE);
    std::uniform_int_distribution<std::uint16_t> dist16(0, 0xFFFFu);

    struct KeyPack {
        std::array<std::uint16_t, 8>  K{};
        std::array<std::uint16_t, 16> RK{};
    };
    std::vector<KeyPack> keys(NUM_KEYS);

    for (std::uint32_t k = 0; k < NUM_KEYS; ++k) {
        for (int i = 0; i < 8; ++i) keys[k].K[i] = dist16(key_rng);
        cham64_key_schedule(keys[k].K.data(), keys[k].RK.data());
    }

    // key_counts[k][od] = その鍵kで「diffのodビットが0だった回数」
    std::vector<std::array<std::uint64_t, 64>> key_counts(NUM_KEYS);

    // 作業バッファ（中央値用）
    std::vector<std::uint64_t> tmp_counts;
    tmp_counts.reserve(NUM_KEYS);

    for (int rounds = ROUNDS_MIN; rounds <= ROUNDS_MAX; ++rounds) {
        std::cout << "=== rounds = " << rounds << " ===\n";

        for (int id_bit = ID_BIT_START; id_bit <= ID_BIT_END; ++id_bit) {

            // 初期化
            for (std::uint32_t k = 0; k < NUM_KEYS; ++k) {
                key_counts[k].fill(0ULL);
            }

            // ---- OpenMP: keys loop を並列化（鍵ごとに独立に書き込み）----
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (std::int64_t k = 0; k < static_cast<std::int64_t>(NUM_KEYS); ++k) {
                std::array<std::uint64_t, 64> local_counts{};
                local_counts.fill(0ULL);

                const std::uint64_t seed =
                    RNG_SEED_BASE
                    ^ (static_cast<std::uint64_t>(rounds) * 0x9e3779b97f4a7c15ULL)
                    ^ (static_cast<std::uint64_t>(id_bit) * 0xbf58476d1ce4e5b9ULL)
                    ^ (static_cast<std::uint64_t>(k) * 0x94d049bb133111ebULL);

                std::mt19937_64 rng(seed);
                std::uniform_int_distribution<std::uint64_t> dist64_local(
                    0, std::numeric_limits<std::uint64_t>::max()
                );

                for (std::uint64_t s = 0; s < NUM_SAMPLES; ++s) {
                    const std::uint64_t P = dist64_local(rng);
                    const std::uint64_t P2 = P ^ (1ULL << id_bit);

                    std::uint16_t plain0[4], plain1[4];
                    unpack_state64(P, plain0);
                    unpack_state64(P2, plain1);

                    std::uint16_t c0[4], c1[4];
                    cham64_encrypt_rounds(keys[static_cast<std::size_t>(k)].RK.data(), plain0, c0, rounds);
                    cham64_encrypt_rounds(keys[static_cast<std::size_t>(k)].RK.data(), plain1, c1, rounds);

                    const std::uint64_t C0 = pack_state64(c0);
                    const std::uint64_t C1 = pack_state64(c1);
                    const std::uint64_t diff = C0 ^ C1;

                    // same_mask の 1 が「同じ（Δ=0）」ビット
                    std::uint64_t same_mask = ~diff;
                    while (same_mask) {
                        const int b = ctz64(same_mask);
                        local_counts[static_cast<std::size_t>(b)]++;
                        same_mask &= (same_mask - 1);
                    }
                }

                key_counts[static_cast<std::size_t>(k)] = local_counts;
            }

            // ---- od_bit ごとに鍵方向の中央値を計算して出力 ----
            for (int od_bit = 0; od_bit < 64; ++od_bit) {
                tmp_counts.clear();
                for (std::uint32_t k = 0; k < NUM_KEYS; ++k) {
                    tmp_counts.push_back(key_counts[k][static_cast<std::size_t>(od_bit)]);
                }
                std::sort(tmp_counts.begin(), tmp_counts.end());

                const double p_same_median = median_from_counts(tmp_counts, NUM_SAMPLES);
                const double epsilon_median = 2.0 * p_same_median - 1.0;

                ofs << rounds << ','
                    << NUM_SAMPLES << ','
                    << NUM_KEYS << ','
                    << id_bit << ','
                    << od_bit << ','
                    << std::setprecision(10) << p_same_median << ','
                    << std::setprecision(10) << epsilon_median << '\n';
            }

            std::cout << "Finished rounds=" << rounds
                << ", id_bit=" << id_bit << "\n";
        }
    }

    auto t_end = clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;

    std::cout << "Done. Output written to " << OUTPUT_CSV << "\n";
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    return 0;
}

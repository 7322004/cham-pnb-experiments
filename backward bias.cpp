// cham_backward_bias_multi_keys_rngsplit.cpp
// CHAM-64/128：後方バイアス εa を
//   p_agree = Pr[ Δ^(r)_{OD} == Γ^(r)_{OD} ] から
//   εa = 2*p_agree - 1
// として推定するコード（一致=1, 不一致=0）。
//
// 変更点（RNG分離）:
// - plaintext生成用RNG と PNB埋め用RNG を完全に分離
//   -> 「平文と近似鍵の独立性」を実験設計上より明確にする
//
// ★変更点（今回）:
// - 複数鍵の集約として eps_a の中央値（偶数本数は中央2点の平均）を出力
//   併せて mean も出力（p_agree も mean/median を併記）
//
// ID/OD の入力は 0..63 の linear index（LSB=0）
// PNB bit index は 0..127（K[word]のLSB=0）

#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <stdexcept>
#include <exception>
#include <cstdlib>
#include <cstddef>

// =====================
// 1) CHAM-64/128 定数
// =====================
static constexpr int W = 16;
static constexpr int STATE_WORDS = 4;
static constexpr int KEY_WORDS = 8;
static constexpr int RK_WORDS = 16;
static constexpr int MAX_ROUNDS_DEFAULT = 88;

// =====================
// 2) 16-bit rotation
// =====================
static inline uint16_t rotl16(uint16_t x, unsigned r) {
    r &= 15u;
    return static_cast<uint16_t>((x << r) | (x >> ((16u - r) & 15u)));
}
static inline uint16_t rotr16(uint16_t x, unsigned r) {
    r &= 15u;
    return static_cast<uint16_t>((x >> r) | (x << ((16u - r) & 15u)));
}

// =====================
// 3) Key schedule (CHAM-64/128)
// =====================
static std::array<uint16_t, RK_WORDS>
cham64_128_key_schedule(const std::array<uint16_t, KEY_WORDS>& K) {
    std::array<uint16_t, RK_WORDS> rk{};
    constexpr int m = KEY_WORDS; // 8
    for (int i = 0; i < m; i++) {
        const uint16_t ki = K[i];
        rk[i] = static_cast<uint16_t>(ki ^ rotl16(ki, 1) ^ rotl16(ki, 8));
        rk[(i + m) ^ 1] = static_cast<uint16_t>(ki ^ rotl16(ki, 1) ^ rotl16(ki, 11));
    }
    return rk;
}

// =====================
// 4) Round / Inverse round
// =====================
static inline void cham_round(std::array<uint16_t, STATE_WORDS>& s,
    uint16_t rki, uint32_t i) {
    const unsigned alpha = ((i & 1u) == 0u) ? 1u : 8u;
    const unsigned beta = ((i & 1u) == 0u) ? 8u : 1u;

    const uint16_t x = s[0], y = s[1], z = s[2], w = s[3];

    const uint32_t sum = static_cast<uint32_t>(x ^ static_cast<uint16_t>(i))
        + static_cast<uint32_t>(rotl16(y, alpha) ^ rki);
    const uint16_t t = rotl16(static_cast<uint16_t>(sum), beta);

    s[0] = y;
    s[1] = z;
    s[2] = w;
    s[3] = t;
}

static inline void cham_inv_round(std::array<uint16_t, STATE_WORDS>& s,
    uint16_t rki, uint32_t i) {
    const unsigned alpha = ((i & 1u) == 0u) ? 1u : 8u;
    const unsigned beta = ((i & 1u) == 0u) ? 8u : 1u;

    const uint16_t y = s[0];
    const uint16_t z = s[1];
    const uint16_t w = s[2];
    const uint16_t t = s[3];

    const uint16_t t0 = rotr16(t, beta);
    const uint16_t sub = static_cast<uint16_t>(rotl16(y, alpha) ^ rki);

    const uint16_t x_xor_i = static_cast<uint16_t>(t0 - sub);
    const uint16_t x = static_cast<uint16_t>(x_xor_i ^ static_cast<uint16_t>(i));

    s[0] = x;
    s[1] = y;
    s[2] = z;
    s[3] = w;
}

static inline void cham_forward(std::array<uint16_t, STATE_WORDS>& s,
    const std::array<uint16_t, RK_WORDS>& rk,
    uint32_t start_round, uint32_t end_round) {
    for (uint32_t i = start_round; i < end_round; i++) {
        cham_round(s, rk[i % RK_WORDS], i);
    }
}

static inline void cham_backward(std::array<uint16_t, STATE_WORDS>& s,
    const std::array<uint16_t, RK_WORDS>& rk,
    uint32_t start_round, uint32_t end_round) {
    for (int64_t i = static_cast<int64_t>(end_round) - 1;
        i >= static_cast<int64_t>(start_round); i--) {
        cham_inv_round(s, rk[static_cast<uint32_t>(i) % RK_WORDS], static_cast<uint32_t>(i));
    }
}

// =====================
// 5) Parsing helpers
// =====================
static std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(std::stoi(tok));
    }
    return out;
}

static std::vector<int> load_ints_file(const std::string& path) {
    std::vector<int> out;
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("failed to open file: " + path);
    std::string line;
    while (std::getline(ifs, line)) {
        auto pos = line.find('#');
        if (pos != std::string::npos) line = line.substr(0, pos);
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        size_t st = 0;
        while (st < line.size() && (line[st] == ' ' || line[st] == '\t')) st++;
        if (st >= line.size()) continue;
        out.push_back(std::stoi(line.substr(st)));
    }
    return out;
}

static inline uint8_t get_bit16(uint16_t x, int bit) {
    return static_cast<uint8_t>((x >> bit) & 1u);
}

static inline void linear_to_wordbit(int linear, int& word, int& bit) {
    if (linear < 0 || linear >= STATE_WORDS * W)
        throw std::runtime_error("ID/OD linear index out of range (0..63): " + std::to_string(linear));
    word = linear / W;
    bit = linear % W;
}

// =====================
// 6) Params
// =====================
struct Params {
    uint32_t R = 32;
    uint32_t r = 20;
    uint64_t N = (1ull << 18);

    uint32_t num_true_keys = 1;

    uint64_t seed = 1;

    uint64_t key_seed0 = 0;
    bool key_seed0_given = false;

    bool print_per_key = false;

    int id_pos = 0;
    int od_pos = 0;

    std::vector<int> pnb_bits;

    bool pnb_per_sample = true;
};

static void usage() {
    std::cout
        << "Usage: cham_backward_bias [options]\n"
        << "Options:\n"
        << "  --R <int>                output round R\n"
        << "  --r <int>                observation round r\n"
        << "  --N <int>                number of pairs per key\n"
        << "  --keys <int>             number of true keys to sample (default 1)\n"
        << "  --seed <int>             base RNG seed (samples & approx-key)\n"
        << "  --key-seed0 <int>        RNG seed for generating true keys (default=seed)\n"
        << "  --print-per-key <0|1>    print each key result (default 0)\n"
        << "  --id <0..63>             input difference position (linear bit index, LSB=0)\n"
        << "  --od <0..63>             OD position at round r (linear bit index, LSB=0)\n"
        << "  --pnb <csv>              PNB key bit indices (0..127), comma-separated\n"
        << "  --pnb-file <path>        PNB key bit indices (one per line)\n"
        << "  --pnb-per-sample <0|1>   randomize PNB per sample (default 1)\n";
}

static Params parse_args(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need = [&](const char* opt)->std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + opt);
            return argv[++i];
            };

        if (a == "--help" || a == "-h") {
            usage();
            std::exit(0);
        }
        else if (a == "--R") {
            p.R = static_cast<uint32_t>(std::stoul(need("--R")));
        }
        else if (a == "--r") {
            p.r = static_cast<uint32_t>(std::stoul(need("--r")));
        }
        else if (a == "--N") {
            p.N = static_cast<uint64_t>(std::stoull(need("--N")));
        }
        else if (a == "--keys") {
            p.num_true_keys = static_cast<uint32_t>(std::stoul(need("--keys")));
        }
        else if (a == "--seed") {
            p.seed = static_cast<uint64_t>(std::stoull(need("--seed")));
        }
        else if (a == "--key-seed0") {
            p.key_seed0 = static_cast<uint64_t>(std::stoull(need("--key-seed0")));
            p.key_seed0_given = true;
        }
        else if (a == "--print-per-key") {
            p.print_per_key = (std::stoi(need("--print-per-key")) != 0);
        }
        else if (a == "--id") {
            const std::string v = need("--id");
            if (v.find(',') != std::string::npos) {
                auto wb = parse_csv_ints(v);
                if (wb.size() != 2) throw std::runtime_error("--id expects 0..63 or \"word,bit\"");
                if (wb[0] < 0 || wb[0] >= STATE_WORDS || wb[1] < 0 || wb[1] >= W)
                    throw std::runtime_error("--id word/bit out of range (word=0..3, bit=0..15)");
                p.id_pos = wb[0] * W + wb[1];
            }
            else {
                p.id_pos = std::stoi(v);
            }
        }
        else if (a == "--od") {
            const std::string v = need("--od");
            if (v.find(',') != std::string::npos) {
                auto wb = parse_csv_ints(v);
                if (wb.size() != 2) throw std::runtime_error("--od expects 0..63 or \"word,bit\"");
                if (wb[0] < 0 || wb[0] >= STATE_WORDS || wb[1] < 0 || wb[1] >= W)
                    throw std::runtime_error("--od word/bit out of range (word=0..3, bit=0..15)");
                p.od_pos = wb[0] * W + wb[1];
            }
            else {
                p.od_pos = std::stoi(v);
            }
        }
        else if (a == "--pnb") {
            p.pnb_bits = parse_csv_ints(need("--pnb"));
        }
        else if (a == "--pnb-file") {
            auto more = load_ints_file(need("--pnb-file"));
            p.pnb_bits.insert(p.pnb_bits.end(), more.begin(), more.end());
        }
        else if (a == "--pnb-per-sample") {
            p.pnb_per_sample = (std::stoi(need("--pnb-per-sample")) != 0);
        }
        else {
            throw std::runtime_error("unknown option: " + a);
        }
    }

    if (p.id_pos < 0 || p.id_pos >= STATE_WORDS * W)
        throw std::runtime_error("--id out of range (0..63): " + std::to_string(p.id_pos));
    if (p.od_pos < 0 || p.od_pos >= STATE_WORDS * W)
        throw std::runtime_error("--od out of range (0..63): " + std::to_string(p.od_pos));
    if (p.num_true_keys == 0)
        throw std::runtime_error("--keys must be >= 1");
    if (!p.key_seed0_given) p.key_seed0 = p.seed;

    if (p.r > p.R) throw std::runtime_error("require r <= R");
    if (p.N == 0) throw std::runtime_error("--N must be >= 1");

    return p;
}

// =====================
// 7) RNG seed splitting helper (SplitMix64)
// =====================
static inline uint64_t splitmix64_next(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// =====================
// 7.5) mean / median helper
// =====================
static inline double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

static inline double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n & 1u) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// =====================
// 8) εa measurement core
// =====================
struct Result {
    uint64_t agree = 0;
    uint64_t trials = 0;
    double p_agree = 0.0;
    double eps_a = 0.0;
    double se_eps = 0.0; // binomial approx
};

static Result measure_eps_a_one_key(const Params& P,
    const std::array<uint16_t, KEY_WORDS>& trueK,
    uint64_t base_seed_for_this_key) {
    int id_word, id_bit, od_word, od_bit;
    linear_to_wordbit(P.id_pos, id_word, id_bit);
    linear_to_wordbit(P.od_pos, od_word, od_bit);

    // RNG split: plaintext RNG and PNB RNG are independent streams
    uint64_t sm = base_seed_for_this_key;
    const uint64_t seed_plain = splitmix64_next(sm);
    const uint64_t seed_pnb = splitmix64_next(sm);

    std::mt19937_64 rng_plain(seed_plain);
    std::mt19937_64 rng_pnb(seed_pnb);

    std::uniform_int_distribution<uint16_t> dist16_plain(0, 0xFFFFu);
    std::uniform_int_distribution<uint16_t> dist16_pnb(0, 0xFFFFu);

    // build PNB mask over master key words
    std::array<uint16_t, KEY_WORDS> pnb_mask{};
    for (int b : P.pnb_bits) {
        if (b < 0 || b >= KEY_WORDS * W)
            throw std::runtime_error("PNB bit index out of range (0..127): " + std::to_string(b));
        const int wi = b / W;
        const int bi = b % W;
        pnb_mask[wi] = static_cast<uint16_t>(pnb_mask[wi] | (static_cast<uint16_t>(1u) << bi));
    }

    auto make_approxK = [&]() -> std::array<uint16_t, KEY_WORDS> {
        std::array<uint16_t, KEY_WORDS> approxK{};
        for (int i = 0; i < KEY_WORDS; i++) {
            const uint16_t keep = static_cast<uint16_t>(trueK[i] & static_cast<uint16_t>(~pnb_mask[i]));
            const uint16_t rnd = dist16_pnb(rng_pnb);     // <-- PNB RNG only
            const uint16_t fill = static_cast<uint16_t>(rnd & pnb_mask[i]);
            approxK[i] = static_cast<uint16_t>(keep | fill);
        }
        return approxK;
        };

    const auto trueRK = cham64_128_key_schedule(trueK);

    std::array<uint16_t, RK_WORDS> fixedApproxRK{};
    bool has_fixed = false;
    if (!P.pnb_per_sample) {
        const auto fixedApproxK = make_approxK();       // consumes only rng_pnb
        fixedApproxRK = cham64_128_key_schedule(fixedApproxK);
        has_fixed = true;
    }

    uint64_t agree = 0;
    for (uint64_t t = 0; t < P.N; t++) {
        // sample X0 and X0' (plaintext RNG only)
        std::array<uint16_t, STATE_WORDS> X0{};
        for (int i = 0; i < STATE_WORDS; i++) X0[i] = dist16_plain(rng_plain);

        auto X0p = X0;
        X0p[id_word] = static_cast<uint16_t>(X0p[id_word] ^ (static_cast<uint16_t>(1u) << id_bit));

        // true Xr (0 -> r)
        auto Xr = X0;
        auto Xrp = X0p;
        cham_forward(Xr, trueRK, 0, P.r);
        cham_forward(Xrp, trueRK, 0, P.r);

        const uint8_t f = static_cast<uint8_t>(
            get_bit16(Xr[od_word], od_bit) ^ get_bit16(Xrp[od_word], od_bit));

        // true XR (0 -> R)
        auto XR = X0;
        auto XRp = X0p;
        cham_forward(XR, trueRK, 0, P.R);
        cham_forward(XRp, trueRK, 0, P.R);

        // approx rk for backward (PNB RNG only)
        std::array<uint16_t, RK_WORDS> approxRK{};
        if (has_fixed) {
            approxRK = fixedApproxRK;
        }
        else {
            const auto approxK = make_approxK();
            approxRK = cham64_128_key_schedule(approxK);
        }

        // backward XR -> Xr using approxRK
        auto Xr_t = XR;
        auto Xr_tp = XRp;
        cham_backward(Xr_t, approxRK, P.r, P.R);
        cham_backward(Xr_tp, approxRK, P.r, P.R);

        const uint8_t g = static_cast<uint8_t>(
            get_bit16(Xr_t[od_word], od_bit) ^ get_bit16(Xr_tp[od_word], od_bit));

        agree += (f == g) ? 1ull : 0ull;
    }

    const double p_agree = static_cast<double>(agree) / static_cast<double>(P.N);
    const double eps_a = 2.0 * p_agree - 1.0;

    const double se_p = std::sqrt(std::max(0.0, p_agree * (1.0 - p_agree)) / static_cast<double>(P.N));
    const double se_eps = 2.0 * se_p;

    return Result{ agree, P.N, p_agree, eps_a, se_eps };
}

// generate random true key (8x16-bit)
static std::array<uint16_t, KEY_WORDS> random_true_key(std::mt19937_64& key_rng) {
    std::uniform_int_distribution<uint16_t> dist16(0, 0xFFFFu);
    std::array<uint16_t, KEY_WORDS> K{};
    for (int i = 0; i < KEY_WORDS; i++) K[i] = dist16(key_rng);
    return K;
}

int main(int argc, char** argv) {
    try {
        const Params P = parse_args(argc, argv);

        if (P.R > static_cast<uint32_t>(MAX_ROUNDS_DEFAULT)) {
            std::cerr << "[warn] R=" << P.R << " > " << MAX_ROUNDS_DEFAULT
                << " (revised CHAM-64/128 rounds). Ensure intended.\n";
        }
        if (P.pnb_bits.empty()) {
            std::cerr << "[warn] pnb_bits is empty -> approxK==trueK (expect eps_a near 1).\n";
        }

        // key RNG
        std::mt19937_64 key_rng(P.key_seed0);

        uint64_t agree_all = 0;
        uint64_t trials_all = 0;

        std::vector<double> eps_each;
        std::vector<double> p_agree_each;
        eps_each.reserve(P.num_true_keys);
        p_agree_each.reserve(P.num_true_keys);

        // per-key base seed: separate stream per key, then split inside
        const uint64_t mix = 0x9e3779b97f4a7c15ULL;

        for (uint32_t kidx = 0; kidx < P.num_true_keys; kidx++) {
            const auto trueK = random_true_key(key_rng);
            const uint64_t base_seed_for_this_key = P.seed + mix * static_cast<uint64_t>(kidx);

            const Result res = measure_eps_a_one_key(P, trueK, base_seed_for_this_key);
            agree_all += res.agree;
            trials_all += res.trials;

            eps_each.push_back(res.eps_a);
            p_agree_each.push_back(res.p_agree);

            if (P.print_per_key) {
                std::cout << "key#" << kidx
                    << "  p_agree=" << res.p_agree
                    << "  eps_a=" << res.eps_a
                    << "  se(eps_a)=" << res.se_eps
                    << "\n";
            }
        }

        // aggregate by pooled counts (参考として残す)
        const double p_agree_all = (trials_all == 0) ? 0.5
            : (static_cast<double>(agree_all) / static_cast<double>(trials_all));
        const double eps_a_all = 2.0 * p_agree_all - 1.0;

        const double se_p_all = std::sqrt(std::max(0.0, p_agree_all * (1.0 - p_agree_all))
            / std::max<double>(1.0, static_cast<double>(trials_all)));
        const double se_eps_all = 2.0 * se_p_all;

        // mean/median across keys（★中央値は偶数時に中央2点の平均）
        const double eps_mean = mean_of(eps_each);
        const double eps_median = median_of(eps_each);

        const double pag_mean = mean_of(p_agree_each);
        const double pag_median = median_of(p_agree_each);

        std::cout << "---- aggregate ----\n"
            << "R=" << P.R << " r=" << P.r << " N=" << P.N
            << " keys=" << P.num_true_keys
            << " id=" << P.id_pos
            << " od=" << P.od_pos
            << " pnb_bits=" << P.pnb_bits.size()
            << " pnb_per_sample=" << (P.pnb_per_sample ? 1 : 0)
            << " seed=" << P.seed
            << " key_seed0=" << P.key_seed0
            << "\n"
            << "p_agree_all=" << p_agree_all << "\n"
            << "eps_a_all=" << eps_a_all << "\n"
            << "se(eps_a_all)=" << se_eps_all << "\n"
            << "p_agree_mean(keys)=" << pag_mean << "\n"
            << "p_agree_median(keys)=" << pag_median << "\n"
            << "eps_a_mean(keys)=" << eps_mean << "\n"
            << "eps_a_median(keys)=" << eps_median << "\n";

        return 0;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        usage();
        return 1;
    }
}

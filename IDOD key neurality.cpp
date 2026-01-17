// cham_key_neutrality_multi_idod_randomkeys_median.cpp
// CHAM-64/128 鍵ビット中立度測定（複数IDOD / IDODは0..63 / ランダム鍵を複数生成）
//
// CHAM3.cpp の仕様は維持しつつ、出力に
//   - 従来の「全キー・全シード・全サンプルを合算した平均（p_same_avg）」
//   - 「鍵ごとの p_same を作って、その中央値（p_same_median）」
// を併記する版。
//
// 追加機能（CHAM3同様）:
//   --args-file <path> : ファイルの各行を「1回分のコマンド引数」として順番に実行。
//                         空行は無視。'#' 以降はコメントとして無視。
//
// 出力列（平均＋中央値）:
// R,r,id_bit,od_bit,key_bit,samples_per_seed,num_seeds,samples_total,key_hex,
// p_same_avg,epsilon_avg,p_same_median,epsilon_median
//
// p_same_avg の定義（CHAM3と同じ）:
//   全キー・全シード・全サンプルで same/total を合算し、p_same_avg = same/total, epsilon_avg=2*p_same_avg-1
//
// p_same_median の定義（追加）:
//   鍵 k ごとに p_same_k = same_k / total_k を計算し、その中央値を p_same_median とする。
//   total_k==0 の場合は p_same_k=0.5 とする（CHAM3の used==0 の扱いに合わせる）
//
// Build:
//   g++ -O3 -std=c++17 -march=native cham_key_neutrality_multi_idod_randomkeys_median.cpp -o cham_neutrality_median
//
// Example:
//   ./cham_neutrality_median --R 32 --r 28 --samples-per-seed 262144 --num-seeds 4 --seed0 1 \
//     --od-filter 1 --od-expected 1 --num-keys 64 --key-seed0 999 --idod 12,7 --out-prefix neutrality_test
//
//   ./cham_neutrality_median --args-file runs.txt

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int W = 16;
static constexpr int STATE_WORDS = 4;   // 64-bit state = 4x16
static constexpr int KEY_WORDS = 8;   // 128-bit key  = 8x16
static constexpr int RK_WORDS = 16;  // CHAM-64/128 key schedule outputs 16 round keys

// ------------------ util ------------------
static inline std::uint16_t rotl16(std::uint16_t x, unsigned r) {
    r &= 15u;
    return static_cast<std::uint16_t>((x << r) | (x >> ((16u - r) & 15u)));
}
static inline std::uint16_t rotr16(std::uint16_t x, unsigned r) {
    r &= 15u;
    return static_cast<std::uint16_t>((x >> r) | (x << ((16u - r) & 15u)));
}

static inline void linear_to_wordbit(int linear, int& word, int& bit) {
    if (linear < 0 || linear >= STATE_WORDS * W)
        throw std::runtime_error("state bit index out of range (0..63): " + std::to_string(linear));
    word = linear / W;
    bit = linear % W;
}
static inline std::uint8_t get_bit16(std::uint16_t x, int bit) {
    return static_cast<std::uint8_t>((x >> bit) & 1u);
}

// 64-bit <-> 4x16-bit (LSB=bit0)
static inline void unpack_state64(std::uint64_t v, std::array<std::uint16_t, 4>& s) {
    s[0] = static_cast<std::uint16_t>(v & 0xFFFFu);
    s[1] = static_cast<std::uint16_t>((v >> 16) & 0xFFFFu);
    s[2] = static_cast<std::uint16_t>((v >> 32) & 0xFFFFu);
    s[3] = static_cast<std::uint16_t>((v >> 48) & 0xFFFFu);
}

// ------------------ CHAM-64/128 key schedule ------------------
static std::array<std::uint16_t, RK_WORDS>
cham64_128_key_schedule(const std::array<std::uint16_t, KEY_WORDS>& K) {
    std::array<std::uint16_t, RK_WORDS> rk{};
    constexpr int m = KEY_WORDS; // 8
    for (int i = 0; i < m; i++) {
        const std::uint16_t ki = K[i];
        rk[i] = static_cast<std::uint16_t>(ki ^ rotl16(ki, 1) ^ rotl16(ki, 8));
        rk[(i + m) ^ 1] = static_cast<std::uint16_t>(ki ^ rotl16(ki, 1) ^ rotl16(ki, 11));
    }
    return rk;
}

// ------------------ round / inverse round ------------------
static inline void cham_round(std::array<std::uint16_t, 4>& s,
    std::uint16_t rki, std::uint32_t i) {
    const unsigned alpha = ((i & 1u) == 0u) ? 1u : 8u;
    const unsigned beta = ((i & 1u) == 0u) ? 8u : 1u;

    const std::uint16_t x = s[0], y = s[1], z = s[2], w = s[3];
    const std::uint32_t sum = static_cast<std::uint32_t>(x ^ static_cast<std::uint16_t>(i))
        + static_cast<std::uint32_t>(rotl16(y, alpha) ^ rki);
    const std::uint16_t t = rotl16(static_cast<std::uint16_t>(sum), beta);

    s[0] = y; s[1] = z; s[2] = w; s[3] = t;
}

static inline void cham_inv_round(std::array<std::uint16_t, 4>& s,
    std::uint16_t rki, std::uint32_t i) {
    const unsigned alpha = ((i & 1u) == 0u) ? 1u : 8u;
    const unsigned beta = ((i & 1u) == 0u) ? 8u : 1u;

    // s = (y,z,w,t)
    const std::uint16_t y = s[0], z = s[1], w = s[2], t = s[3];

    const std::uint16_t t0 = rotr16(t, beta);
    const std::uint16_t sub = static_cast<std::uint16_t>(rotl16(y, alpha) ^ rki);

    const std::uint16_t x_xor_i = static_cast<std::uint16_t>(t0 - sub);
    const std::uint16_t x = static_cast<std::uint16_t>(x_xor_i ^ static_cast<std::uint16_t>(i));

    s[0] = x; s[1] = y; s[2] = z; s[3] = w;
}

static inline void cham_forward(std::array<std::uint16_t, 4>& s,
    const std::array<std::uint16_t, RK_WORDS>& rk,
    std::uint32_t start_round, std::uint32_t end_round) {
    for (std::uint32_t i = start_round; i < end_round; i++) {
        cham_round(s, rk[i % RK_WORDS], i);
    }
}

static inline void cham_backward(std::array<std::uint16_t, 4>& s,
    const std::array<std::uint16_t, RK_WORDS>& rk,
    std::uint32_t start_round, std::uint32_t end_round) {
    for (std::int64_t i = static_cast<std::int64_t>(end_round) - 1;
        i >= static_cast<std::int64_t>(start_round); i--) {
        cham_inv_round(s, rk[static_cast<std::uint32_t>(i) % RK_WORDS], static_cast<std::uint32_t>(i));
    }
}

// ------------------ key bit flip (0..127, LSB=0 in each word) ------------------
static std::array<std::uint16_t, KEY_WORDS>
flip_key_bit(const std::array<std::uint16_t, KEY_WORDS>& K, int bit_index) {
    if (bit_index < 0 || bit_index >= KEY_WORDS * W)
        throw std::runtime_error("key bit index out of range (0..127): " + std::to_string(bit_index));
    auto out = K;
    const int wi = bit_index / W;
    const int bi = bit_index % W;
    out[wi] = static_cast<std::uint16_t>(out[wi] ^ (static_cast<std::uint16_t>(1u) << bi));
    return out;
}

// ------------------ IDOD list load ------------------
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

static std::vector<std::pair<int, int>> load_idod_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("failed to open idod file: " + path);
    std::vector<std::pair<int, int>> v;
    std::string line;
    while (std::getline(ifs, line)) {
        auto pos = line.find('#');
        if (pos != std::string::npos) line = line.substr(0, pos);
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        std::size_t st = 0;
        while (st < line.size() && (line[st] == ' ' || line[st] == '\t')) st++;
        if (st >= line.size()) continue;

        auto items = parse_csv_ints(line.substr(st));
        if (items.size() != 2) throw std::runtime_error("idod file line must be: id_bit,od_bit");
        v.emplace_back(items[0], items[1]);
    }
    return v;
}

// ------------------ params ------------------
struct Params {
    std::uint32_t R = 32;
    std::uint32_t r = 28;

    std::uint64_t samples_per_seed = (1ull << 18);
    std::uint32_t num_seeds = 4;
    std::uint64_t seed0 = 1;

    bool use_od_filter = true;
    int  od_expected = 1;

    // random keys
    std::uint32_t num_keys = 5;
    std::uint64_t key_seed0 = 12345;

    std::vector<std::pair<int, int>> idod_pairs;
    std::string out_prefix = "neutrality";
};

static void usage() {
    std::cout
        << "Usage:\n"
        << "  cham_neutrality_median [options]\n"
        << "  cham_neutrality_median --args-file <path>\n"
        << "\n"
        << "Options:\n"
        << "  --R <int>\n"
        << "  --r <int>\n"
        << "  --samples-per-seed <int>\n"
        << "  --num-seeds <int>\n"
        << "  --seed0 <int>\n"
        << "  --od-filter <0|1>\n"
        << "  --od-expected <0|1>\n"
        << "  --num-keys <int>\n"
        << "  --key-seed0 <int>\n"
        << "  --idod <id,od>           (repeatable)\n"
        << "  --idod-file <path>\n"
        << "  --out-prefix <str>\n"
        << "  --args-file <path>\n";
}

// 1行を argv 風に分割（簡易：空白区切り、"..." / '...' 対応、\" と \\ の最低限対応）
static std::vector<std::string> tokenize_command_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quote = false;
    char quote_char = '"';

    auto flush = [&]() {
        if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        };

    for (std::size_t i = 0; i < line.size(); i++) {
        char c = line[i];

        if (c == '\\' && i + 1 < line.size()) {
            char n = line[i + 1];
            if (n == '"' || n == '\\') {
                cur.push_back(n);
                i++;
                continue;
            }
        }

        if (in_quote) {
            if (c == quote_char) {
                in_quote = false;
            }
            else {
                cur.push_back(c);
            }
            continue;
        }

        if (c == '"' || c == '\'') {
            in_quote = true;
            quote_char = c;
            continue;
        }

        if (c == ' ' || c == '\t') {
            flush();
            continue;
        }

        cur.push_back(c);
    }
    flush();
    return out;
}

static Params parse_args(const std::vector<std::string>& argv) {
    Params p;

    auto need = [&](std::size_t& i, const char* opt)->std::string {
        if (i + 1 >= argv.size()) throw std::runtime_error(std::string("missing value for ") + opt);
        return argv[++i];
        };

    for (std::size_t i = 1; i < argv.size(); i++) {
        const std::string& a = argv[i];

        if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else if (a == "--R") p.R = static_cast<std::uint32_t>(std::stoul(need(i, "--R")));
        else if (a == "--r") p.r = static_cast<std::uint32_t>(std::stoul(need(i, "--r")));
        else if (a == "--samples-per-seed") p.samples_per_seed = static_cast<std::uint64_t>(std::stoull(need(i, "--samples-per-seed")));
        else if (a == "--num-seeds") p.num_seeds = static_cast<std::uint32_t>(std::stoul(need(i, "--num-seeds")));
        else if (a == "--seed0") p.seed0 = static_cast<std::uint64_t>(std::stoull(need(i, "--seed0")));
        else if (a == "--od-filter") p.use_od_filter = (std::stoi(need(i, "--od-filter")) != 0);
        else if (a == "--od-expected") p.od_expected = std::stoi(need(i, "--od-expected"));
        else if (a == "--num-keys") p.num_keys = static_cast<std::uint32_t>(std::stoul(need(i, "--num-keys")));
        else if (a == "--key-seed0") p.key_seed0 = static_cast<std::uint64_t>(std::stoull(need(i, "--key-seed0")));
        else if (a == "--idod") {
            auto v = parse_csv_ints(need(i, "--idod"));
            if (v.size() != 2) throw std::runtime_error("--idod expects id,od");
            p.idod_pairs.emplace_back(v[0], v[1]);
        }
        else if (a == "--idod-file") {
            auto more = load_idod_file(need(i, "--idod-file"));
            p.idod_pairs.insert(p.idod_pairs.end(), more.begin(), more.end());
        }
        else if (a == "--out-prefix") {
            p.out_prefix = need(i, "--out-prefix");
        }
        else if (a == "--args-file") {
            throw std::runtime_error("--args-file must be used alone: program --args-file <path>");
        }
        else {
            throw std::runtime_error("unknown option: " + a);
        }
    }

    if (p.r > p.R) throw std::runtime_error("require r <= R");
    if (!(p.od_expected == 0 || p.od_expected == 1)) throw std::runtime_error("--od-expected must be 0 or 1");
    if (p.idod_pairs.empty()) throw std::runtime_error("no IDOD pairs. use --idod or --idod-file");
    if (p.num_keys == 0) throw std::runtime_error("--num-keys must be >= 1");

    for (auto [idb, odb] : p.idod_pairs) {
        if (idb < 0 || idb > 63) throw std::runtime_error("id_bit out of range (0..63): " + std::to_string(idb));
        if (odb < 0 || odb > 63) throw std::runtime_error("od_bit out of range (0..63): " + std::to_string(odb));
    }
    return p;
}

// ------------------ sample struct ------------------
struct Sample {
    std::array<std::uint16_t, 4> XR0;
    std::array<std::uint16_t, 4> XR1;
    std::uint8_t diff_r_bit; // original OD diff bit at round r (under the current master key)
};

// generate one random master key (8x16)
static std::array<std::uint16_t, KEY_WORDS> random_master_key(std::mt19937_64& rng) {
    std::uniform_int_distribution<std::uint16_t> dist16(0, 0xFFFFu);
    std::array<std::uint16_t, KEY_WORDS> K{};
    for (int i = 0; i < KEY_WORDS; i++) K[i] = dist16(rng);
    return K;
}

static double median_inplace_sorted(std::vector<double>& v) {
    if (v.empty()) return 0.5;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if (n & 1u) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void run_one_idod(const Params& P, int id_bit, int od_bit) {
    int id_word, id_b, od_word, od_b;
    linear_to_wordbit(id_bit, id_word, id_b);
    linear_to_wordbit(od_bit, od_word, od_b);

    // ---- global (CHAM3互換の平均) ----
    std::array<std::uint64_t, 128> total_all{};
    std::array<std::uint64_t, 128> same_all{};
    total_all.fill(0);
    same_all.fill(0);

    // ---- per-key probabilities for median ----
    std::vector<std::vector<double>> psame_per_kb(128);
    for (int kb = 0; kb < 128; kb++) psame_per_kb[kb].reserve(P.num_keys);

    // key RNG (separate from plaintext RNG)
    std::mt19937_64 key_rng(P.key_seed0);

    // loop over random keys
    for (std::uint32_t kidx = 0; kidx < P.num_keys; kidx++) {
        const auto master_key = random_master_key(key_rng);
        const auto rk_master = cham64_128_key_schedule(master_key);

        // precompute round keys for each flipped key bit for this master_key
        std::array<std::array<std::uint16_t, RK_WORDS>, 128> rk_flip_all{};
        for (int kb = 0; kb < 128; kb++) {
            auto Kf = flip_key_bit(master_key, kb);
            rk_flip_all[static_cast<std::size_t>(kb)] = cham64_128_key_schedule(Kf);
        }

        // per-key accumulators (for median)
        std::array<std::uint64_t, 128> total_k{};
        std::array<std::uint64_t, 128> same_k{};
        total_k.fill(0);
        same_k.fill(0);

        // per seed
        for (std::uint32_t si = 0; si < P.num_seeds; si++) {
            const std::uint64_t seed = P.seed0 + static_cast<std::uint64_t>(si)
                + (static_cast<std::uint64_t>(kidx) << 32); // separate streams per key
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<std::uint64_t> dist64(0, std::numeric_limits<std::uint64_t>::max());

            std::vector<Sample> samples;
            samples.reserve(static_cast<std::size_t>(P.samples_per_seed));

            // generate samples under current master_key
            for (std::uint64_t t = 0; t < P.samples_per_seed; t++) {
                const std::uint64_t P0 = dist64(rng);
                const std::uint64_t P1 = P0 ^ (1ull << id_bit);

                std::array<std::uint16_t, 4> X0{}, X0p{};
                unpack_state64(P0, X0);
                unpack_state64(P1, X0p);

                // 0 -> r
                auto Xr = X0;
                auto Xrp = X0p;
                cham_forward(Xr, rk_master, 0, P.r);
                cham_forward(Xrp, rk_master, 0, P.r);

                const std::uint8_t bit_orig =
                    static_cast<std::uint8_t>(get_bit16(Xr[od_word], od_b) ^ get_bit16(Xrp[od_word], od_b));

                // filter early (メモリ/時間削減)
                if (P.use_od_filter && bit_orig != static_cast<std::uint8_t>(P.od_expected))
                    continue;

                // r -> R
                auto XR = Xr;
                auto XRp = Xrp;
                cham_forward(XR, rk_master, P.r, P.R);
                cham_forward(XRp, rk_master, P.r, P.R);

                samples.push_back(Sample{ XR, XRp, bit_orig });
            }

            const std::uint64_t used_seed = static_cast<std::uint64_t>(samples.size());
            // total_k は kb ごとに同じ数だけ増える（filter後のサンプル数）
            for (int kb = 0; kb < 128; kb++) {
                total_k[static_cast<std::size_t>(kb)] += used_seed;
            }

            // evaluate each key bit
            for (int kb = 0; kb < 128; kb++) {
                const auto& rk_flip = rk_flip_all[static_cast<std::size_t>(kb)];

                for (const auto& sp : samples) {
                    auto Xr0 = sp.XR0;
                    auto Xr1 = sp.XR1;
                    cham_backward(Xr0, rk_flip, P.r, P.R);
                    cham_backward(Xr1, rk_flip, P.r, P.R);

                    const std::uint8_t bit_mod =
                        static_cast<std::uint8_t>(get_bit16(Xr0[od_word], od_b) ^ get_bit16(Xr1[od_word], od_b));

                    if (bit_mod == sp.diff_r_bit) {
                        same_k[static_cast<std::size_t>(kb)]++;
                    }
                }
            }
        } // seeds

        // ---- update global avg accumulators ----
        for (int kb = 0; kb < 128; kb++) {
            total_all[static_cast<std::size_t>(kb)] += total_k[static_cast<std::size_t>(kb)];
            same_all[static_cast<std::size_t>(kb)] += same_k[static_cast<std::size_t>(kb)];
        }

        // ---- store per-key p_same for median ----
        for (int kb = 0; kb < 128; kb++) {
            const std::uint64_t used = total_k[static_cast<std::size_t>(kb)];
            double p_same_k = 0.5;
            if (used != 0) {
                p_same_k = static_cast<double>(same_k[static_cast<std::size_t>(kb)]) / static_cast<double>(used);
            }
            psame_per_kb[static_cast<std::size_t>(kb)].push_back(p_same_k);
        }
    } // keys

    // output CSV (per IDOD)
    std::ostringstream fn;
    fn << P.out_prefix << "_R" << P.R << "_r" << P.r
        << "_id" << id_bit << "_od" << od_bit << ".csv";

    std::ofstream ofs(fn.str());
    if (!ofs) throw std::runtime_error("failed to open output: " + fn.str());

    ofs << "R,r,id_bit,od_bit,key_bit,samples_per_seed,num_seeds,samples_total,key_hex,"
        << "p_same_avg,epsilon_avg,p_same_median,epsilon_median\n";

    // key_hex: fixed key is not meaningful here, so store reproducibility info
    std::ostringstream key_meta;
    key_meta << "random_keys(num_keys=" << P.num_keys
        << ",key_seed0=" << P.key_seed0
        << ",seed0=" << P.seed0 << ")";

    for (int kb = 0; kb < 128; kb++) {
        const std::uint64_t used_all = total_all[static_cast<std::size_t>(kb)];

        // avg (CHAM3互換: global same/total)
        double p_avg = 0.5;
        double e_avg = 0.0;
        if (used_all != 0) {
            p_avg = static_cast<double>(same_all[static_cast<std::size_t>(kb)]) / static_cast<double>(used_all);
            e_avg = 2.0 * p_avg - 1.0;
        }

        // median across keys
        auto v = psame_per_kb[static_cast<std::size_t>(kb)]; // copy (ソートで破壊しない)
        double p_med = median_inplace_sorted(v);
        double e_med = 2.0 * p_med - 1.0;

        ofs << P.R << ','
            << P.r << ','
            << id_bit << ','
            << od_bit << ','
            << kb << ','
            << P.samples_per_seed << ','
            << P.num_seeds << ','
            << used_all << ','
            << '"' << key_meta.str() << '"' << ','
            << std::setprecision(12) << p_avg << ','
            << std::setprecision(12) << e_avg << ','
            << std::setprecision(12) << p_med << ','
            << std::setprecision(12) << e_med
            << "\n";
    }

    std::cout << "[OK] wrote " << fn.str() << "\n";
}

int main(int argc, char** argv) {
    try {
        std::vector<std::string> av;
        av.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; i++) av.push_back(argv[i]);

        // --args-file <path> があればバッチ実行（このモードは単独指定のみ許可）
        if (argc == 3 && std::string(argv[1]) == "--args-file") {
            const std::string path = argv[2];
            std::ifstream ifs(path);
            if (!ifs) throw std::runtime_error("failed to open args file: " + path);

            std::string line;
            std::size_t lineno = 0;
            while (std::getline(ifs, line)) {
                lineno++;

                // コメント除去
                if (auto pos = line.find('#'); pos != std::string::npos) line = line.substr(0, pos);

                // trim（簡易）
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t'))
                    line.pop_back();
                std::size_t st = 0;
                while (st < line.size() && (line[st] == ' ' || line[st] == '\t')) st++;
                if (st >= line.size()) continue;
                line = line.substr(st);

                // 1行をトークン化 → argv 風にする
                auto tokens = tokenize_command_line(line);
                std::vector<std::string> one;
                one.reserve(tokens.size() + 1);
                one.push_back(av[0]); // program name
                one.insert(one.end(), tokens.begin(), tokens.end());

                const Params P = parse_args(one);

                std::cout << "---- batch line " << lineno << " ----\n";
                std::cout << "R=" << P.R << " r=" << P.r
                    << " samples_per_seed=" << P.samples_per_seed
                    << " num_seeds=" << P.num_seeds
                    << " num_keys=" << P.num_keys
                    << " key_seed0=" << P.key_seed0
                    << " od_filter=" << (P.use_od_filter ? 1 : 0)
                    << " od_expected=" << P.od_expected
                    << " out_prefix=" << P.out_prefix
                    << "\n";

                for (auto [idb, odb] : P.idod_pairs) {
                    run_one_idod(P, idb, odb);
                }
            }
            return 0;
        }

        // 通常の単発実行
        const Params P = parse_args(av);

        std::cout << "R=" << P.R << " r=" << P.r
            << " samples_per_seed=" << P.samples_per_seed
            << " num_seeds=" << P.num_seeds
            << " num_keys=" << P.num_keys
            << " key_seed0=" << P.key_seed0
            << " od_filter=" << (P.use_od_filter ? 1 : 0)
            << " od_expected=" << P.od_expected
            << "\n";

        for (auto [idb, odb] : P.idod_pairs) {
            run_one_idod(P, idb, odb);
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        usage();
        return 1;
    }
}

// Card abstraction: k-means over equity features, one table per postflop street.
//
// Suit-isomorphic boards share a row, so flop, turn and river need only 1,755,
// 16,432 and 134,459 rows of 1,326 bytes, and a lookup is plain array indexing:
//     row = canonical board id; column = hole cards relabelled by its suit map.
//
// Features: the river uses EHS (equity against a uniformly random hand). The
// flop and turn use the CDF of that equity over every runout. L1 distance
// between CDFs is exactly the 1-D earth mover's distance (Johanson et al. 2013).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include "cards.hpp"

inline constexpr int BINS = 50;  // equity histogram resolution, flop and turn

#pragma pack(push, 1)
struct AbstractionHeader {
    char magic[8];
    u64 id;  // hash of the tables: ties a strategy to the abstraction it used
    u32 buckets[4];  // preflop, flop, turn, river
    u32 boards[3];   // canonical boards: flop, turn, river
};
#pragma pack(pop)

// File layout: header, preflop[1326] u8, perms[24][52] u8, then per street
// (k = 3, 4, 5): canon[C(52, k)] u32, buckets[boards * 1326] u8.
struct Abstraction {
    AbstractionHeader header{};
    std::array<u8, HOLES> preflop{};
    std::vector<u32> canon[3];  // by board_rank: canonical id << 5 | suit perm
    std::vector<u8> table[3];   // [id * 1326 + hole]

    int bucket(int street, u64 board, int a, int b) const {  // street 1..3
        u32 entry = canon[street - 1][board_rank(board)];
        const auto& perm = SUIT_PERMS[entry & 31];
        return table[street - 1][size_t(entry >> 5) * HOLES +
                                 hole_index(perm[a], perm[b])];
    }
};

// Fills equity[hole] for every hole not blocked by the 5-card board, in one
// sorted sweep instead of 990 comparisons per hole.
class EquitySweep {
    std::vector<u64> keys;

public:
    EquitySweep() { keys.reserve(HOLES); }

    void run(u64 board, float* equity) {
        u16 suits[4] = {};
        for (u64 m = board; m; m &= m - 1) add_card(suits, std::countr_zero(m));
        keys.clear();
        for (int h = 0; h < HOLES; h++) {
            int a = HOLE_CARDS[h][0], b = HOLE_CARDS[h][1];
            if (board >> a & 1 || board >> b & 1) continue;
            u16 s[4] = {suits[0], suits[1], suits[2], suits[3]};
            add_card(s, a);
            add_card(s, b);
            keys.push_back(u64(evaluate7(s)) << 16 | u64(h));
        }
        std::sort(keys.begin(), keys.end());
        int weaker = 0, weaker_with[52] = {};
        for (size_t i = 0, n = keys.size(); i < n;) {
            size_t j = i;
            int group_with[52] = {};
            for (; j < n && keys[j] >> 16 == keys[i] >> 16; j++) {
                const auto& c = HOLE_CARDS[keys[j] & 0xFFFF];
                group_with[c[0]]++;
                group_with[c[1]]++;
            }
            int group = int(j - i);
            for (size_t k = i; k < j; k++) {
                const auto& c = HOLE_CARDS[keys[k] & 0xFFFF];
                // A hand sharing a card is not a possible opponent; a hand
                // sharing both is this hand, which the +1 puts back.
                int wins = weaker - weaker_with[c[0]] - weaker_with[c[1]];
                int ties = group - group_with[c[0]] - group_with[c[1]] + 1;
                equity[keys[k] & 0xFFFF] = (wins + 0.5f * ties) / 990.0f;
            }
            for (size_t k = i; k < j; k++) {
                const auto& c = HOLE_CARDS[keys[k] & 0xFFFF];
                weaker_with[c[0]]++;
                weaker_with[c[1]]++;
            }
            weaker += group;
            i = j;
        }
    }
};

inline int feature_dim(int street) { return street == 3 ? 1 : BINS; }

// out[hole * dim..] gets each unblocked hole's feature, in which a smaller
// value means a stronger hand: 1 - EHS on the river, the equity CDF elsewhere.
class Features {
    EquitySweep sweep;
    std::vector<float> equity;
    std::vector<u32> hist;

public:
    Features() : equity(HOLES), hist(HOLES * BINS) {}

    void compute(int street, u64 board, float* out) {
        if (street == 3) {
            sweep.run(board, equity.data());
            for (int h = 0; h < HOLES; h++) out[h] = 1.0f - equity[h];
            return;
        }
        std::fill(hist.begin(), hist.end(), 0u);
        auto add_runout = [&](u64 full) {
            sweep.run(full, equity.data());
            for (int h = 0; h < HOLES; h++) {
                const auto& c = HOLE_CARDS[h];
                if (full >> c[0] & 1 || full >> c[1] & 1) continue;
                hist[h * BINS + std::min(int(equity[h] * BINS), BINS - 1)]++;
            }
        };
        for (int t = 0; t < 52; t++) {
            if (board >> t & 1) continue;
            if (street == 2) {
                add_runout(board | 1ULL << t);
            } else {
                for (int r = 0; r < t; r++)
                    if (!(board >> r & 1)) add_runout(board | 1ULL << t | 1ULL << r);
            }
        }
        for (int h = 0; h < HOLES; h++) {
            const auto& c = HOLE_CARDS[h];
            if (board >> c[0] & 1 || board >> c[1] & 1) continue;
            u32 total = 0, running = 0;
            for (int i = 0; i < BINS; i++) total += hist[h * BINS + i];
            for (int i = 0; i < BINS; i++) {
                running += hist[h * BINS + i];
                out[h * BINS + i] = float(running) / float(total);
            }
        }
    }
};

inline float emd(const float* a, const float* b, int dim) {
    float sum = 0;
#pragma omp simd reduction(+ : sum)
    for (int i = 0; i < dim; i++) sum += std::fabs(a[i] - b[i]);
    return sum;
}

inline int nearest(const float* point, const float* centroids, int k, int dim) {
    int best = 0;
    float best_distance = 1e30f;
    for (int c = 0; c < k; c++) {
        float distance = emd(point, centroids + size_t(c) * dim, dim);
        if (distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }
    return best;
}

// Lloyd's algorithm under EMD with mean centroids, as in Johanson et al.
// Returns k centroids ordered weakest first, so buckets rank by hand strength.
inline std::vector<float> kmeans(const std::vector<float>& points, int dim, int k,
                                 int threads, u64 seed) {
    const size_t n = points.size() / dim;
    if (n < size_t(k)) die("too few points to cluster");
    const size_t chunk = 4096, chunks = (n + chunk - 1) / chunk;
    Rng rng(seed);
    std::vector<float> centroids(size_t(k) * dim);

    // k-means++ seeding: spreads the centroids over the distribution.
    std::vector<float> distance(n, 1e30f);
    size_t chosen = rng.below(u32(n));
    for (int c = 0; c < k; c++) {
        std::memcpy(&centroids[size_t(c) * dim], &points[chosen * dim],
                    sizeof(float) * dim);
        parallel_for(chunks, threads, [&](size_t ch) {
            for (size_t i = ch * chunk; i < std::min(n, (ch + 1) * chunk); i++)
                distance[i] = std::min(
                    distance[i],
                    emd(&points[i * dim], &centroids[size_t(c) * dim], dim));
        });
        double total = 0;
        for (float x : distance) total += x;
        double target = rng.unit() * total;
        for (size_t i = 0; i < n; i++) {
            target -= distance[i];
            if (target <= 0) {
                chosen = i;
                break;
            }
        }
    }

    std::vector<int> assigned(n, -1);
    for (int iteration = 0; iteration < 30; iteration++) {
        std::atomic<size_t> changed{0};
        std::vector<std::vector<double>> sums(threads);
        std::vector<std::vector<u32>> counts(threads);
        std::atomic<size_t> cursor{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++)
            pool.emplace_back([&, t] {
                auto& sum = sums[t];
                auto& count = counts[t];
                sum.assign(size_t(k) * dim, 0.0);
                count.assign(k, 0);
                for (size_t ch; (ch = cursor.fetch_add(1)) < chunks;)
                    for (size_t i = ch * chunk; i < std::min(n, (ch + 1) * chunk); i++) {
                        const float* p = &points[i * dim];
                        int c = nearest(p, centroids.data(), k, dim);
                        if (c != assigned[i]) {
                            assigned[i] = c;
                            changed.fetch_add(1, std::memory_order_relaxed);
                        }
                        count[c]++;
                        double* s = &sum[size_t(c) * dim];
                        for (int d = 0; d < dim; d++) s[d] += p[d];
                    }
            });
        for (auto& t : pool) t.join();
        for (int c = 0; c < k; c++) {
            u64 count = 0;
            for (int t = 0; t < threads; t++) count += counts[t][c];
            if (!count) continue;  // an empty cluster keeps its old centroid
            for (int d = 0; d < dim; d++) {
                double sum = 0;
                for (int t = 0; t < threads; t++) sum += sums[t][size_t(c) * dim + d];
                centroids[size_t(c) * dim + d] = float(sum / double(count));
            }
        }
        if (changed.load() < n / 1000) break;
    }

    std::vector<int> order(k);
    std::vector<double> weakness(k);
    for (int c = 0; c < k; c++) {
        order[c] = c;
        for (int d = 0; d < dim; d++) weakness[c] += centroids[size_t(c) * dim + d];
    }
    std::sort(order.begin(), order.end(),
              [&](int x, int y) { return weakness[x] > weakness[y]; });
    std::vector<float> sorted(centroids.size());
    for (int c = 0; c < k; c++)
        std::memcpy(&sorted[size_t(c) * dim], &centroids[size_t(order[c]) * dim],
                    sizeof(float) * dim);
    return sorted;
}

inline u64 random_board(Rng& rng, int cards) {
    u64 mask = 0;
    while (std::popcount(mask) < cards) mask |= 1ULL << rng.below(52);
    return mask;
}

inline Abstraction build_abstraction(int flop_k, int turn_k, int river_k, int threads,
                                     u64 seed) {
    for (int k : {flop_k, turn_k, river_k})
        if (k < 2 || k > 256) die("buckets per street must be 2..256 (stored as u8)");
    Abstraction abs;
    std::memcpy(abs.header.magic, "CFRABS1", 8);
    abs.header.buckets[0] = 169;
    int ks[3] = {flop_k, turn_k, river_k};
    for (int s = 0; s < 3; s++) abs.header.buckets[s + 1] = u32(ks[s]);
    for (int h = 0; h < HOLES; h++)
        abs.preflop[h] = u8(preflop_class(HOLE_CARDS[h][0], HOLE_CARDS[h][1]));

    // Fit on a sample of real deals, so frequent boards count for more.
    // {boards, holes per board}: flop features cost 1,081 sweeps a board.
    const size_t sample[3][2] = {{300, 1000}, {3000, 100}, {20000, 15}};
    for (int s = 0; s < 3; s++) {
        int street = s + 1, cards = s + 3, dim = feature_dim(street);

        // Index the suit-isomorphic boards, walking sets in colex order.
        const u32 sets = BINOMIAL[52][cards];
        abs.canon[s].resize(sets);
        std::vector<u32> id_of_rank(sets);
        std::vector<u64> canonical;
        u64 mask = (1ULL << cards) - 1;
        for (u32 rank = 0; rank < sets; rank++) {
            u64 smallest = ~0ULL;
            int perm = 0;
            for (int p = 0; p < 24; p++) {
                u64 relabelled = permute_mask(mask, SUIT_PERMS[p]);
                if (relabelled < smallest) {
                    smallest = relabelled;
                    perm = p;
                }
            }
            if (smallest == mask) {  // the smallest of its class comes first
                id_of_rank[rank] = u32(canonical.size());
                canonical.push_back(mask);
            }
            abs.canon[s][rank] = id_of_rank[board_rank(smallest)] << 5 | u32(perm);
            u64 low = mask & -mask, high = mask + low;  // Gosper's next set
            mask = high | ((mask ^ high) >> 2) / low;
        }
        abs.header.boards[s] = u32(canonical.size());
        std::printf("street %d: %zu canonical boards; fitting %d buckets...\n", street,
                    canonical.size(), ks[s]);
        std::fflush(stdout);

        // Fit.
        const size_t fit_boards = sample[s][0], holes_per = sample[s][1];
        std::vector<float> points(fit_boards * holes_per * dim);
        std::atomic<size_t> cursor{0};
        {
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; t++)
                pool.emplace_back([&, t] {
                    Features features;
                    std::vector<float> row(size_t(HOLES) * dim);
                    for (size_t i; (i = cursor.fetch_add(1)) < fit_boards;) {
                        Rng rng(seed * 1000003 + s * 7919 + i);
                        u64 board = random_board(rng, cards);
                        features.compute(street, board, row.data());
                        for (size_t j = 0; j < holes_per; j++) {
                            int h;
                            do h = rng.below(HOLES);
                            while (board >> HOLE_CARDS[h][0] & 1 ||
                                   board >> HOLE_CARDS[h][1] & 1);
                            std::memcpy(&points[(i * holes_per + j) * dim],
                                        &row[size_t(h) * dim], sizeof(float) * dim);
                        }
                    }
                });
            for (auto& t : pool) t.join();
        }
        auto centroids = kmeans(points, dim, ks[s], threads, seed + s);
        std::vector<float>().swap(points);

        // Assign every canonical board and hole to its nearest centroid.
        std::printf("street %d: assigning...\n", street);
        std::fflush(stdout);
        abs.table[s].assign(canonical.size() * HOLES, 0);
        cursor = 0;
        {
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; t++)
                pool.emplace_back([&] {
                    Features features;
                    std::vector<float> row(size_t(HOLES) * dim);
                    for (size_t i; (i = cursor.fetch_add(1)) < canonical.size();) {
                        features.compute(street, canonical[i], row.data());
                        for (int h = 0; h < HOLES; h++) {
                            if (canonical[i] >> HOLE_CARDS[h][0] & 1 ||
                                canonical[i] >> HOLE_CARDS[h][1] & 1)
                                continue;
                            abs.table[s][i * HOLES + h] = u8(nearest(
                                &row[size_t(h) * dim], centroids.data(), ks[s], dim));
                        }
                    }
                });
            for (auto& t : pool) t.join();
        }
    }

    u64 hash = 0xCBF29CE484222325ULL;  // FNV-1a
    for (const auto& table : abs.table)
        for (u8 byte : table) hash = (hash ^ byte) * 0x100000001B3ULL;
    abs.header.id = hash;
    return abs;
}

inline void save_abstraction(const Abstraction& abs, const char* path) {
    FILE* f = std::fopen(path, "wb");
    if (!f) die("cannot write", path);
    bool ok = std::fwrite(&abs.header, sizeof abs.header, 1, f) == 1 &&
              std::fwrite(abs.preflop.data(), 1, HOLES, f) == HOLES;
    for (const auto& perm : SUIT_PERMS)
        ok = ok && std::fwrite(perm.data(), 1, 52, f) == 52;
    for (int s = 0; s < 3; s++) {
        ok = ok && std::fwrite(abs.canon[s].data(), 4, abs.canon[s].size(), f) ==
                       abs.canon[s].size();
        ok = ok && std::fwrite(abs.table[s].data(), 1, abs.table[s].size(), f) ==
                       abs.table[s].size();
    }
    if (std::fclose(f) != 0 || !ok) die("failed writing", path);
}

inline Abstraction load_abstraction(const char* path) {
    Abstraction abs;
    FILE* f = std::fopen(path, "rb");
    if (!f) die("cannot read", path);
    bool ok = std::fread(&abs.header, sizeof abs.header, 1, f) == 1 &&
              std::memcmp(abs.header.magic, "CFRABS1", 8) == 0 &&
              std::fread(abs.preflop.data(), 1, HOLES, f) == HOLES;
    if (!ok) die("not an abstraction file:", path);
    std::fseek(f, 24 * 52, SEEK_CUR);  // perms: compiled in, stored for the reader
    for (int s = 0; s < 3; s++) {
        abs.canon[s].resize(BINOMIAL[52][s + 3]);
        abs.table[s].resize(size_t(abs.header.boards[s]) * HOLES);
        ok = ok && std::fread(abs.canon[s].data(), 4, abs.canon[s].size(), f) ==
                       abs.canon[s].size();
        ok = ok && std::fread(abs.table[s].data(), 1, abs.table[s].size(), f) ==
                       abs.table[s].size();
    }
    std::fclose(f);
    if (!ok) die("truncated abstraction file:", path);
    return abs;
}

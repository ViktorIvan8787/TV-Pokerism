// Cards, the 7-card evaluator, suit-isomorphic board indexing, threads, RNG.
// A card is rank * 4 + suit: the order of card_utils.full_deck().
#pragma once
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

inline constexpr int HOLES = 1326;  // C(52, 2)

[[noreturn]] inline void die(const char* message, const char* detail = "") {
    std::fprintf(stderr, "error: %s %s\n", message, detail);
    std::exit(1);
}

struct Rng {  // xoshiro256++: thread-local and far faster than std::mt19937
    u64 s[4];
    explicit Rng(u64 seed) {
        for (auto& x : s) {  // splitmix64
            seed += 0x9E3779B97F4A7C15ULL;
            u64 z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            x = z ^ (z >> 31);
        }
    }
    u64 next() {
        u64 r = std::rotl(s[0] + s[3], 23) + s[0], t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = std::rotl(s[3], 45);
        return r;
    }
    u32 below(u32 n) { return u32(((next() >> 32) * n) >> 32); }
    double unit() { return double(next() >> 11) * 0x1p-53; }
};

// Calls fn(i) for i in 0..n-1 across threads, handing out indices dynamically.
template <class Fn>
void parallel_for(size_t n, int threads, Fn fn) {
    std::atomic<size_t> cursor{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&] {
            for (size_t i; (i = cursor.fetch_add(1)) < n;) fn(i);
        });
    for (auto& t : pool) t.join();
}

// -- hole cards ---------------------------------------------------------------

inline int hole_index(int a, int b) {
    if (a < b) std::swap(a, b);
    return a * (a - 1) / 2 + b;
}

inline const auto HOLE_CARDS = [] {
    std::array<std::array<u8, 2>, HOLES> cards{};
    for (int a = 1; a < 52; a++)
        for (int b = 0; b < a; b++) cards[hole_index(a, b)] = {u8(a), u8(b)};
    return cards;
}();

// The 169 strategically distinct starting hands: lossless, so no clustering.
inline int preflop_class(int a, int b) {
    int hi = a >> 2, lo = b >> 2;
    if (hi < lo) std::swap(hi, lo);
    if (hi == lo) return hi;
    int pair = hi * (hi - 1) / 2 + lo;  // 0..77
    return 13 + pair + ((a & 3) == (b & 3) ? 78 : 0);
}

// -- 7-card evaluator ---------------------------------------------------------

// STRAIGHT[rank mask] is the high card of the best straight, plus 1; 0 if none.
inline const auto STRAIGHT = [] {
    std::array<u8, 8192> table{};
    for (u32 mask = 0; mask < 8192; mask++) {
        for (int high = 12; high >= 4 && !table[mask]; high--)
            if (((mask >> (high - 4)) & 31) == 31) table[mask] = u8(high + 1);
        if (!table[mask] && (mask & 0x100F) == 0x100F) table[mask] = 4;  // wheel
    }
    return table;
}();

inline int top_bit(u32 mask) { return 31 - std::countl_zero(mask); }

inline u32 top_bits(u32 mask, int n) {
    u32 kept = 0;
    for (; n > 0 && mask; n--) {
        u32 bit = 1u << top_bit(mask);
        kept |= bit;
        mask ^= bit;
    }
    return kept;
}

// suits[s] is the mask of ranks held in suit s. Equal scores split the pot.
// The order matches card_utils.evaluate; tests/test_solver.py checks it.
inline u32 evaluate7(const u16 suits[4]) {
    auto score = [](u32 category, u32 major, u32 minor) {
        return category << 26 | major << 13 | minor;
    };
    u32 a = suits[0], b = suits[1], c = suits[2], d = suits[3];
    u32 any = a | b | c | d;
    u32 flush = 0;
    for (int s = 0; s < 4; s++)
        if (std::popcount(u32(suits[s])) >= 5) flush = suits[s];
    if (flush && STRAIGHT[flush]) return score(8, STRAIGHT[flush], 0);
    u32 two = (a & b) | (c & d) | ((a | b) & (c | d));  // held in 2+ suits
    u32 three = (a & b & (c | d)) | (c & d & (a | b));
    if (u32 four = a & b & c & d) {
        int rank = top_bit(four);
        return score(7, rank, top_bit(any ^ (1u << rank)));
    }
    if (three) {
        int rank = top_bit(three);
        if (u32 rest = two ^ (1u << rank)) return score(6, rank, top_bit(rest));
    }
    if (flush) return score(5, 0, top_bits(flush, 5));
    if (STRAIGHT[any]) return score(4, STRAIGHT[any], 0);
    if (three) {
        int rank = top_bit(three);
        return score(3, rank, top_bits(any ^ (1u << rank), 2));
    }
    if (std::popcount(two) >= 2) {
        u32 pairs = top_bits(two, 2);
        return score(2, pairs, top_bit(any ^ pairs));
    }
    if (two) return score(1, top_bit(two), top_bits(any ^ two, 3));
    return score(0, 0, top_bits(any, 5));
}

inline void add_card(u16 suits[4], int card) {
    suits[card & 3] |= u16(1u << (card >> 2));
}

// -- board indexing -----------------------------------------------------------

inline const auto BINOMIAL = [] {
    std::array<std::array<u32, 6>, 53> c{};
    for (int n = 0; n <= 52; n++) {
        c[n][0] = 1;
        for (int k = 1; k <= 5 && k <= n; k++)
            c[n][k] = c[n - 1][k - 1] + c[n - 1][k];
    }
    return c;
}();

// A k-card set's position among all k-card sets (colexicographic).
inline u32 board_rank(u64 mask) {
    u32 rank = 0;
    for (int k = 1; mask; k++, mask &= mask - 1)
        rank += BINOMIAL[std::countr_zero(mask)][k];
    return rank;
}

// The 24 ways to relabel suits, as a card -> card table.
inline const auto SUIT_PERMS = [] {
    std::array<std::array<u8, 52>, 24> perms{};
    int count = 0;
    for (int w = 0; w < 4; w++)
        for (int x = 0; x < 4; x++)
            for (int y = 0; y < 4; y++)
                for (int z = 0; z < 4; z++) {
                    if (w == x || w == y || w == z || x == y || x == z || y == z)
                        continue;
                    int to[4] = {w, x, y, z};
                    for (int c = 0; c < 52; c++)
                        perms[count][c] = u8((c & ~3) | to[c & 3]);
                    count++;
                }
    return perms;
}();

inline u64 permute_mask(u64 mask, const std::array<u8, 52>& perm) {
    u64 out = 0;
    for (; mask; mask &= mask - 1) out |= 1ULL << perm[std::countr_zero(mask)];
    return out;
}

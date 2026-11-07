// Real-time subgame solver for the turn and river: exact card equities, no
// abstraction, Discounted CFR (Brown & Sandholm 2019) in vector form.
//
// One row of floats per hand at each node, so the updates are SIMD loops, and a
// showdown is an O(n) sweep over hands sorted by strength. On the turn each
// chance node fans out over the 48 river cards on a spin-waiting thread pool; a
// river solve has no such fan-out and runs on one thread.
//
// Everything is allocated in ss_solve before the first iteration, in one block
// laid out street by street. The loop allocates nothing: ss_result reports the
// count of operator new calls made during it, and the tests check that it is 0.
//
// Built as a shared library with a C interface, for ctypes: see solver/README.md.
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <new>

#include "cards.hpp"

#ifdef _WIN32
#define SS_EXPORT __declspec(dllexport)
#else
#define SS_EXPORT __attribute__((visibility("default")))
#endif

namespace {

std::atomic<u64> g_news{0};

using Clock = std::chrono::steady_clock;

enum Kind : u8 { PLAYER, FOLD, SHOWDOWN, CHANCE };
enum Mode { CFR, BEST_RESPONSE, EVALUATE };

struct alignas(64) SNode {
    int child[5];
    int to[5];        // -1 fold, 0 call or check, else the raise-to total
    u64 offset;       // first float of this node's tables
    int amount;       // FOLD: what the folder loses. SHOWDOWN: what each risks
    u8 kind, player, nact, below_chance;
};

struct Sizes {
    float turn[2], river[2];  // pot fractions; 0 is unused. All-in is implied
    int turn_cap, river_cap;  // raises per street after the root
};

// Tried in order until the tables fit the limit: real states vary too much for
// one tree to suit them all, and a smaller tree converges further in the budget.
constexpr Sizes LADDER[] = {
    {{0.5f, 1.0f}, {0.5f, 1.0f}, 3, 3},
    {{0.5f, 1.0f}, {1.0f, 0.0f}, 2, 2},
    {{1.0f, 0.0f}, {1.0f, 0.0f}, 2, 1},
    // Always keeps a pot-sized bet: with all-in as the only bet, a limped pot
    // of 40 with 480 behind has nothing but check or a 12x pot shove.
    {{1.0f, 0.0f}, {1.0f, 0.0f}, 1, 1},
};

}  // namespace

extern "C" {

struct SsInput {
    int board[5];
    int board_count;  // 4 or 5
    int invested[2];  // chips each seat has put in this hand; seat 0 is the button
    int stacks[2];    // chips behind
    int bets[2];      // chips bet this street
    int acted[2];     // acted this street since the last full raise
    int last_raise;
    int big_blind;
    int to_act;
    int can_raise;    // at the root: the engine's own answer
    int budget_us;
    int max_table_mb;
    float prune;      // drop hands below this fraction of their range's largest
};

struct SsResult {
    int iterations;
    int ladder_level;
    int nodes;
    int hands[2];
    double exploitability;  // chips per hand; 0 is a Nash equilibrium
    double value[2];        // whole-hand net chips per hand for each seat
    double elapsed_ms;
    double table_mb;
    double setup_ms;
    long long loop_news;  // operator new calls during the iterations
};

}  // extern "C"

namespace {

class Solver {
    struct Order {
        u32 rank, hand;
    };
    struct Frame {
        float* top;
        float* alloc(size_t count) {
            float* at = top;
            top += (count + 15) & ~size_t(15);  // whole cache lines, for SIMD
            return at;
        }
    };

    int threads_;
    std::vector<SNode> nodes_;
    std::vector<std::thread> pool_;

    // Set once per solve, then read-only during the iterations.
    SsInput in_{};
    Sizes sizes_{};
    float* memory_ = nullptr;
    size_t capacity_ = 0;
    int n_[2] = {};
    u8* card_a_[2] = {};
    u8* card_b_[2] = {};
    int* same_[2] = {};
    float* range_[2] = {};
    float* mask_[2] = {};
    Order* order_[2] = {};
    int* order_count_[2] = {};
    float* river_result_ = nullptr;
    float* frames_ = nullptr;
    size_t frame_floats_ = 0;
    float* tables_ = nullptr;
    int river_cards_[48] = {};
    int rivers_ = 1;
    int depth_ = 0;
    bool on_turn_ = false;
    int hand_of_[2][HOLES] = {};

    float positive_discount_ = 1, negative_discount_ = 1, average_discount_ = 1;

    // The pool. Fields a worker polls get their own cache lines.
    alignas(64) std::atomic<u32> generation_{0};
    alignas(64) std::atomic<int> next_river_{0};
    alignas(64) std::atomic<int> finished_{0};
    alignas(64) std::atomic<bool> stop_{false};
    int task_child_ = 0, task_player_ = 0;
    Mode task_mode_ = CFR;
    const float* task_reach_ = nullptr;

    // The cursor: where the hand is in the solved tree.
    int cursor_ = 0, cursor_river_ = 0;

    // -- tree -----------------------------------------------------------------

    int add_terminal(Kind kind, int player, int amount, bool below) {
        SNode n{};
        n.kind = kind;
        n.player = u8(player);
        n.amount = amount;
        n.below_chance = below;
        nodes_.push_back(n);
        return int(nodes_.size()) - 1;
    }

    int build(bool river, int player, int b0, int b1, int s0, int s1, int i0, int i1, int last,
              int raises, bool acted0, bool acted1, bool root, int depth) {
        depth_ = std::max(depth_, depth);
        const int bet[2] = {b0, b1}, stack[2] = {s0, s1}, invested[2] = {i0, i1};
        const bool acted[2] = {acted0, acted1};
        const bool below = river && on_turn_;
        const int current = std::max(b0, b1), mine = bet[player], other = 1 - player;
        const int owed = std::min(current - mine, stack[player]);
        const int pot = i0 + i1;
        const int all_in = mine + stack[player];
        const int capped = std::min(all_in, bet[other] + stack[other]);
        const int cap = river ? sizes_.river_cap : sizes_.turn_cap;

        int to[5], count = 0;
        if (current > mine) to[count++] = -1;
        to[count++] = 0;
        bool may_raise = all_in > current && stack[other] > 0 && raises < cap;
        if (root) may_raise = in_.can_raise != 0;
        if (may_raise) {
            const int min_to = std::min(current + last, all_in);
            const float* fractions = river ? sizes_.river : sizes_.turn;
            for (int f = 0; f < 2; f++) {
                if (fractions[f] <= 0) continue;
                const int total = std::max(current + int(fractions[f] * float(pot + owed)), min_to);
                if (total < capped && (count == 0 || to[count - 1] != total) && total > current)
                    to[count++] = total;
            }
            // Capped by a shorter opponent, a total under the minimum raise is
            // only legal when it is really our whole stack.
            if (capped > current && (capped >= min_to || capped == all_in)) to[count++] = capped;
            else if (count == (current > mine ? 2 : 1) && all_in > current) to[count++] = min_to;
        }

        const int index = int(nodes_.size());
        nodes_.emplace_back();
        int child[5] = {};
        for (int a = 0; a < count; a++) {
            if (to[a] == -1) {
                child[a] = add_terminal(FOLD, player, invested[player], below);
                continue;
            }
            int nb0 = b0, nb1 = b1, ns0 = s0, ns1 = s1, ni0 = i0, ni1 = i1;
            const int pay = to[a] == 0 ? owed : to[a] - mine;
            (player == 0 ? nb0 : nb1) += pay;
            (player == 0 ? ns0 : ns1) -= pay;
            (player == 0 ? ni0 : ni1) += pay;
            if (to[a] > 0) {
                const int next_last = std::max(to[a] - current, last);
                child[a] = build(river, other, nb0, nb1, ns0, ns1, ni0, ni1, next_last, raises + 1,
                                 player == 0, player == 1, false, depth + 1);
            } else if (!acted[other]) {
                child[a] = build(river, other, nb0, nb1, ns0, ns1, ni0, ni1, last, raises,
                                 player == 0 || acted0, player == 1 || acted1, false, depth + 1);
            } else if (river) {
                child[a] = add_terminal(SHOWDOWN, 0, std::min(ni0, ni1), below);
            } else {
                const int chance = int(nodes_.size());
                nodes_.emplace_back();
                int below_child;
                if (ns0 == 0 || ns1 == 0) {
                    below_child = add_terminal(SHOWDOWN, 0, std::min(ni0, ni1), true);
                } else {
                    below_child = build(true, 1, 0, 0, ns0, ns1, ni0, ni1, in_.big_blind, 0, false,
                                        false, false, depth + 1);
                }
                SNode& c = nodes_[chance];
                c = SNode{};
                c.kind = CHANCE;
                c.nact = 1;
                c.child[0] = below_child;
                child[a] = chance;
            }
        }
        SNode& n = nodes_[index];
        n = SNode{};
        for (int a = 0; a < count; a++) {
            n.child[a] = child[a];
            n.to[a] = to[a];
        }
        n.kind = PLAYER;
        n.player = u8(player);
        n.nact = u8(count);
        n.below_chance = below;
        return index;
    }

    // -- values at the leaves -------------------------------------------------

    // A fold: every opponent hand that shares no card with ours pays.
    void fold_value(const SNode& n, int p, const float* opp, float* out) const {
        const int q = 1 - p, np = n_[p], nq = n_[q];
        const float amount = float(n.player == p ? -n.amount : n.amount);
        float with_card[52] = {}, total = 0;
        const u8* qa = card_a_[q];
        const u8* qb = card_b_[q];
        for (int j = 0; j < nq; j++) {
            with_card[qa[j]] += opp[j];
            with_card[qb[j]] += opp[j];
            total += opp[j];
        }
        const u8* pa = card_a_[p];
        const u8* pb = card_b_[p];
        const int* same = same_[p];
        for (int i = 0; i < np; i++) {
            const float both = same[i] >= 0 ? opp[same[i]] : 0.0f;
            out[i] = amount * (total - with_card[pa[i]] - with_card[pb[i]] + both);
        }
    }

    // A showdown: two sweeps over the hands in strength order. A hand sharing
    // both our cards has our strength, so it is never counted as win or loss.
    void showdown_value(const SNode& n, int r, int p, const float* opp, float* out) const {
        const int q = 1 - p;
        const Order* mine = order_[p] + size_t(r) * n_[p];
        const Order* theirs = order_[q] + size_t(r) * n_[q];
        const int mine_count = order_count_[p][r], their_count = order_count_[q][r];
        const u8* pa = card_a_[p];
        const u8* pb = card_b_[p];
        const u8* qa = card_a_[q];
        const u8* qb = card_b_[q];
        const float amount = float(n.amount);
        std::memset(out, 0, sizeof(float) * n_[p]);

        float with_card[52] = {}, total = 0;
        for (int i = 0, j = 0; i < mine_count; i++) {
            for (; j < their_count && theirs[j].rank < mine[i].rank; j++) {
                const u32 h = theirs[j].hand;
                total += opp[h];
                with_card[qa[h]] += opp[h];
                with_card[qb[h]] += opp[h];
            }
            const u32 h = mine[i].hand;
            out[h] = amount * (total - with_card[pa[h]] - with_card[pb[h]]);
        }
        std::memset(with_card, 0, sizeof with_card);
        total = 0;
        for (int i = mine_count - 1, j = their_count - 1; i >= 0; i--) {
            for (; j >= 0 && theirs[j].rank > mine[i].rank; j--) {
                const u32 h = theirs[j].hand;
                total += opp[h];
                with_card[qa[h]] += opp[h];
                with_card[qb[h]] += opp[h];
            }
            const u32 h = mine[i].hand;
            out[h] -= amount * (total - with_card[pa[h]] - with_card[pb[h]]);
        }
    }

    // -- strategies -----------------------------------------------------------

    static void regret_match(const float* regret, int nact, int n, float* sigma, float* sums) {
        std::memset(sums, 0, sizeof(float) * n);
        for (int a = 0; a < nact; a++) {
            const float* r = regret + size_t(a) * n;
#pragma omp simd
            for (int i = 0; i < n; i++) sums[i] += r[i] > 0 ? r[i] : 0.0f;
        }
        const float uniform = 1.0f / float(nact);
        for (int a = 0; a < nact; a++) {
            const float* r = regret + size_t(a) * n;
            float* s = sigma + size_t(a) * n;
#pragma omp simd
            for (int i = 0; i < n; i++) {
                const float positive = r[i] > 0 ? r[i] : 0.0f;
                s[i] = sums[i] > 0 ? positive / (sums[i] > 0 ? sums[i] : 1.0f) : uniform;
            }
        }
    }

    static void normalise(const float* average, int nact, int n, float* sigma, float* sums) {
        std::memset(sums, 0, sizeof(float) * n);
        for (int a = 0; a < nact; a++) {
            const float* x = average + size_t(a) * n;
#pragma omp simd
            for (int i = 0; i < n; i++) sums[i] += x[i];
        }
        const float uniform = 1.0f / float(nact);
        for (int a = 0; a < nact; a++) {
            const float* x = average + size_t(a) * n;
            float* s = sigma + size_t(a) * n;
#pragma omp simd
            for (int i = 0; i < n; i++)
                s[i] = sums[i] > 0 ? x[i] / (sums[i] > 0 ? sums[i] : 1.0f) : uniform;
        }
    }

    float* table(const SNode& n, int r) const {
        const size_t per_river = size_t(2) * n.nact * n_[n.player];
        return tables_ + n.offset + (n.below_chance ? size_t(r) * per_river : 0);
    }

    // -- traversal ------------------------------------------------------------

    template <Mode M>
    void visit(int index, int r, int p, const float* opp, float* out, Frame& frame) {
        const SNode& n = nodes_[index];
        switch (n.kind) {
            case FOLD: fold_value(n, p, opp, out); return;
            case SHOWDOWN: showdown_value(n, r, p, opp, out); return;
            case CHANCE: chance<M>(n, p, opp, out, frame); return;
            default: break;
        }
        float* const saved = frame.top;
        const int nact = n.nact, np = n_[p];
        float* const regret = table(n, r);
        if (n.player == p) {
            float* values = frame.alloc(size_t(nact) * np);
            for (int a = 0; a < nact; a++) visit<M>(n.child[a], r, p, opp, values + size_t(a) * np, frame);
            if constexpr (M == BEST_RESPONSE) {
                std::memcpy(out, values, sizeof(float) * np);
                for (int a = 1; a < nact; a++) {
                    const float* v = values + size_t(a) * np;
#pragma omp simd
                    for (int i = 0; i < np; i++) out[i] = out[i] > v[i] ? out[i] : v[i];
                }
            } else {
                float* sigma = frame.alloc(size_t(nact) * np);
                float* sums = frame.alloc(np);
                if constexpr (M == CFR) regret_match(regret, nact, np, sigma, sums);
                else normalise(regret + size_t(nact) * np, nact, np, sigma, sums);
                std::memset(out, 0, sizeof(float) * np);
                for (int a = 0; a < nact; a++) {
                    const float* v = values + size_t(a) * np;
                    const float* s = sigma + size_t(a) * np;
#pragma omp simd
                    for (int i = 0; i < np; i++) out[i] += s[i] * v[i];
                }
                if constexpr (M == CFR) {
                    const float keep_positive = positive_discount_, keep_negative = negative_discount_;
                    for (int a = 0; a < nact; a++) {
                        float* x = regret + size_t(a) * np;
                        const float* v = values + size_t(a) * np;
#pragma omp simd
                        for (int i = 0; i < np; i++)
                            x[i] = x[i] * (x[i] > 0 ? keep_positive : keep_negative) + v[i] - out[i];
                    }
                }
            }
        } else {
            const int nq = n_[n.player];
            float* sigma = frame.alloc(size_t(nact) * nq);
            float* sums = frame.alloc(nq);
            float* reach = frame.alloc(nq);
            float* value = frame.alloc(np);
            if constexpr (M == CFR) regret_match(regret, nact, nq, sigma, sums);
            else normalise(regret + size_t(nact) * nq, nact, nq, sigma, sums);
            std::memset(out, 0, sizeof(float) * np);
            for (int a = 0; a < nact; a++) {
                const float* s = sigma + size_t(a) * nq;
#pragma omp simd
                for (int j = 0; j < nq; j++) reach[j] = opp[j] * s[j];
                if constexpr (M == CFR) {
                    // Weighted by the acting player's reach: that is opp here.
                    float* average = regret + size_t(nact + a) * nq;
                    const float keep = average_discount_;
#pragma omp simd
                    for (int j = 0; j < nq; j++) average[j] = average[j] * keep + reach[j];
                }
                visit<M>(n.child[a], r, p, reach, value, frame);
#pragma omp simd
                for (int i = 0; i < np; i++) out[i] += value[i];
            }
        }
        frame.top = saved;
    }

    template <Mode M>
    void run_river(int r, const float* opp, int child, int p, Frame& frame) {
        float* const saved = frame.top;
        const int q = 1 - p, nq = n_[q], np = n_[p];
        float* reach = frame.alloc(nq);
        const float* q_mask = mask_[q] + size_t(r) * nq;
#pragma omp simd
        for (int j = 0; j < nq; j++) reach[j] = opp[j] * q_mask[j];
        float* result = river_result_ + size_t(r) * np;
        visit<M>(child, r, p, reach, result, frame);
        const float* p_mask = mask_[p] + size_t(r) * np;
#pragma omp simd
        for (int i = 0; i < np; i++) result[i] *= p_mask[i];
        frame.top = saved;
    }

    template <Mode M>
    void work_through_rivers(Frame& frame) {
        for (int r; (r = next_river_.fetch_add(1, std::memory_order_relaxed)) < rivers_;)
            run_river<M>(r, task_reach_, task_child_, task_player_, frame);
    }

    template <Mode M>
    void chance(const SNode& n, int p, const float* opp, float* out, Frame& frame) {
        task_child_ = n.child[0];
        task_player_ = p;
        task_mode_ = M;
        task_reach_ = opp;
        next_river_.store(0, std::memory_order_relaxed);
        finished_.store(0, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        work_through_rivers<M>(frame);
        const int helpers = int(pool_.size());
        while (finished_.load(std::memory_order_acquire) < helpers) _mm_pause();
        // Any two hands leave 44 of the 48 cards possible on the river.
        const int np = n_[p];
        std::memset(out, 0, sizeof(float) * np);
        for (int r = 0; r < rivers_; r++) {
            const float* result = river_result_ + size_t(r) * np;
#pragma omp simd
            for (int i = 0; i < np; i++) out[i] += result[i];
        }
#pragma omp simd
        for (int i = 0; i < np; i++) out[i] /= 44.0f;
    }

    void helper(int index) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);  // denormals are ~100x slower
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        Frame frame{frames_ + size_t(index + 1) * frame_floats_};
        u32 seen = 0;
        for (;;) {
            u32 now;
            while ((now = generation_.load(std::memory_order_acquire)) == seen) {
                if (stop_.load(std::memory_order_relaxed)) return;
                _mm_pause();
            }
            seen = now;
            switch (task_mode_) {
                case CFR: work_through_rivers<CFR>(frame); break;
                case BEST_RESPONSE: work_through_rivers<BEST_RESPONSE>(frame); break;
                case EVALUATE: work_through_rivers<EVALUATE>(frame); break;
            }
            finished_.fetch_add(1, std::memory_order_release);
        }
    }

    // -- setup ----------------------------------------------------------------

    template <class T>
    T* carve(char*& at, size_t count) {
        T* out = reinterpret_cast<T*>(at);
        at += (count * sizeof(T) + 63) & ~size_t(63);
        return out;
    }

    // Returns false if no tree on the ladder fits.
    bool build_tree(int max_floats_mb, SsResult& result) {
        const bool river_street = in_.board_count == 5;
        for (int level = 0; level < int(std::size(LADDER)); level++) {
            sizes_ = LADDER[level];
            nodes_.clear();
            depth_ = 0;
            on_turn_ = !river_street;
            build(river_street, in_.to_act, in_.bets[0], in_.bets[1], in_.stacks[0], in_.stacks[1],
                  in_.invested[0], in_.invested[1], in_.last_raise, 0, in_.acted[0] != 0,
                  in_.acted[1] != 0, true, 0);
            u64 floats = 0;
            for (SNode& n : nodes_) {
                if (n.kind != PLAYER) continue;
                n.offset = floats;
                floats += u64(2) * n.nact * n_[n.player] * (n.below_chance ? u64(rivers_) : u64(1));
            }
            const double mb = double(floats) * 4.0 / 1e6;
            if (mb <= max_floats_mb || level + 1 == int(std::size(LADDER))) {
                result.ladder_level = level;
                result.table_mb = mb;
                result.nodes = int(nodes_.size());
                return true;
            }
        }
        return false;
    }

    int setup(const SsInput& input, const float* range0, const float* range1, SsResult& result) {
        in_ = input;
        const float* ranges[2] = {range0, range1};
        u64 board = 0;
        for (int i = 0; i < in_.board_count; i++) board |= 1ULL << in_.board[i];
        on_turn_ = in_.board_count == 4;
        rivers_ = 1;
        if (on_turn_) {
            rivers_ = 0;
            for (int c = 0; c < 52; c++)
                if (!(board >> c & 1)) river_cards_[rivers_++] = c;
        }
        // Which hands are worth a row: in the range, not blocked by the board.
        int hands[2][HOLES];
        for (int p = 0; p < 2; p++) {
            float largest = 0;
            for (int h = 0; h < HOLES; h++) largest = std::max(largest, ranges[p][h]);
            const float floor = std::max(largest * in_.prune, 1e-30f);
            n_[p] = 0;
            for (int h = 0; h < HOLES; h++) {
                hand_of_[p][h] = -1;
                const auto& c = HOLE_CARDS[h];
                if (ranges[p][h] >= floor && !(board >> c[0] & 1) && !(board >> c[1] & 1)) {
                    hand_of_[p][h] = n_[p];
                    hands[p][n_[p]++] = h;
                }
            }
            if (n_[p] == 0) return -3;
            result.hands[p] = n_[p];
        }
        if (!build_tree(in_.max_table_mb, result)) return -4;
        tables_ = nullptr;

        const int nmax = std::max(n_[0], n_[1]);
        const int workers = on_turn_ ? std::max(1, threads_) : 1;
        // Deep enough for any path: each level needs values, sigma, sums, reach.
        frame_floats_ = size_t(depth_ + 3) * ((size_t(5) * 2 + 4) * nmax + 64);
        const u64 table_floats = u64(result.table_mb * 1e6 / 4.0) + 16;
        const size_t needed =
            table_floats * 4 + size_t(workers) * frame_floats_ * 4 +
            size_t(rivers_) * nmax * (4 + 2 * 4 + 2 * sizeof(Order)) +
            size_t(2) * nmax * (2 + 4 + 4) + size_t(2) * rivers_ * 4 + 4096;
        if (!reserve(needed)) return -5;
        std::memset(static_cast<void*>(memory_), 0, table_floats * 4);
        char* at = reinterpret_cast<char*>(memory_);
        tables_ = carve<float>(at, table_floats);
        frames_ = carve<float>(at, size_t(workers) * frame_floats_);
        river_result_ = carve<float>(at, size_t(rivers_) * nmax);
        for (int p = 0; p < 2; p++) {
            const int n = n_[p];
            card_a_[p] = carve<u8>(at, n);
            card_b_[p] = carve<u8>(at, n);
            same_[p] = carve<int>(at, n);
            range_[p] = carve<float>(at, n);
            mask_[p] = carve<float>(at, size_t(rivers_) * n);
            order_[p] = carve<Order>(at, size_t(rivers_) * n);
            order_count_[p] = carve<int>(at, rivers_);
            for (int i = 0; i < n; i++) {
                const auto& c = HOLE_CARDS[hands[p][i]];
                card_a_[p][i] = c[0];
                card_b_[p][i] = c[1];
                range_[p][i] = ranges[p][hands[p][i]];
            }
        }
        for (int p = 0; p < 2; p++)
            for (int i = 0; i < n_[p]; i++) same_[p][i] = hand_of_[1 - p][hands[p][i]];

        // Hand strengths for every river, sorted once.
        u16 base[4] = {};
        for (int i = 0; i < in_.board_count; i++) add_card(base, in_.board[i]);
        for (int p = 0; p < 2; p++) {
            for (int r = 0; r < rivers_; r++) {
                u16 suits[4] = {base[0], base[1], base[2], base[3]};
                const int river = on_turn_ ? river_cards_[r] : -1;
                if (on_turn_) add_card(suits, river);
                Order* order = order_[p] + size_t(r) * n_[p];
                float* mask = mask_[p] + size_t(r) * n_[p];
                int count = 0;
                for (int i = 0; i < n_[p]; i++) {
                    const int a = card_a_[p][i], b = card_b_[p][i];
                    const bool blocked = a == river || b == river;
                    mask[i] = blocked ? 0.0f : 1.0f;
                    if (blocked) continue;
                    u16 s[4] = {suits[0], suits[1], suits[2], suits[3]};
                    add_card(s, a);
                    add_card(s, b);
                    order[count++] = {evaluate7(s), u32(i)};
                }
                std::sort(order, order + count, [](const Order& x, const Order& y) { return x.rank < y.rank; });
                order_count_[p][r] = count;
            }
        }
        cursor_ = 0;
        cursor_river_ = 0;
        return 0;
    }

    double pair_weight() const {
        // Both ranges over every pair of hands that share no card.
        double total0 = 0, sum = 0;
        double with_card[52] = {};
        for (int j = 0; j < n_[1]; j++) {
            total0 += range_[1][j];
            with_card[card_a_[1][j]] += range_[1][j];
            with_card[card_b_[1][j]] += range_[1][j];
        }
        for (int i = 0; i < n_[0]; i++) {
            const double both = same_[0][i] >= 0 ? range_[1][same_[0][i]] : 0.0;
            sum += range_[0][i] * (total0 - with_card[card_a_[0][i]] - with_card[card_b_[0][i]] + both);
        }
        return sum;
    }

    template <Mode M>
    double root_value(int p, Frame& frame) {
        float* const saved = frame.top;
        float* out = frame.alloc(n_[p]);
        visit<M>(0, 0, p, range_[1 - p], out, frame);
        double total = 0;
        for (int i = 0; i < n_[p]; i++) total += double(out[i]) * range_[p][i];
        frame.top = saved;
        return total;
    }

public:
    explicit Solver(int threads) : threads_(threads) { nodes_.reserve(1 << 14); }

    // Touches the memory now: a first touch during a solve costs a page fault
    // per 4 KB, which took a 50 ms solve to 390 ms.
    bool reserve(size_t bytes) {
        if (bytes <= capacity_) return true;
        if (memory_) _mm_free(memory_);
        memory_ = static_cast<float*>(_mm_malloc(bytes, 64));
        capacity_ = memory_ ? bytes : 0;
        if (!memory_) return false;
        std::memset(static_cast<void*>(memory_), 0, bytes);
        return true;
    }
    ~Solver() {
        if (memory_) _mm_free(memory_);
    }

    int solve(const SsInput& input, const float* range0, const float* range1, SsResult& result) {
        const auto start = Clock::now();
        auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };
        result = SsResult{};
        const int status = setup(input, range0, range1, result);
        if (status != 0) return status;
        const double weight = pair_weight();
        if (weight <= 0) return -3;
        const double budget = input.budget_us / 1e6;
        result.setup_ms = seconds() * 1000;

        const unsigned int saved_csr = _mm_getcsr();
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        generation_.store(0);
        stop_.store(false);
        pool_.clear();
        if (on_turn_)
            for (int t = 0; t < threads_ - 1; t++) pool_.emplace_back([this, t] { helper(t); });

        Frame frame{frames_};
        const u64 news_before = g_news.load();
        double recent = 0;
        int t = 1;
        for (;; t++) {
            const double before = seconds();
            // Stop while there is still time for one more iteration and the
            // measurement, which costs about two more. Predict from the latest
            // iteration, and never from the first: it is slow while the threads
            // start up (50+ ms here), and one descheduled thread must not end
            // the solve.
            if (t > 2 && before + 3.2 * recent > budget) break;
            if (t == 2 && before > 0.8 * budget) break;
            const double power = std::pow(double(t), 1.5);
            positive_discount_ = float(power / (power + 1));  // DCFR: alpha 1.5
            negative_discount_ = 0.5f;                        // beta 0
            const double ratio = double(t) / double(t + 1);
            average_discount_ = float(ratio * ratio);         // gamma 2
            for (int p = 0; p < 2; p++) {
                float* out = frame.alloc(n_[p]);
                visit<CFR>(0, 0, p, range_[1 - p], out, frame);
                frame.top = frames_;
            }
            recent = seconds() - before;
        }
        result.loop_news = (long long)(g_news.load() - news_before);
        result.iterations = t - 1;

        const double best0 = root_value<BEST_RESPONSE>(0, frame) / weight;
        const double best1 = root_value<BEST_RESPONSE>(1, frame) / weight;
        result.exploitability = std::max(0.0, (best0 + best1) / 2);
        result.value[0] = root_value<EVALUATE>(0, frame) / weight;
        result.value[1] = root_value<EVALUATE>(1, frame) / weight;

        stop_.store(true);
        for (auto& worker : pool_) worker.join();
        pool_.clear();
        _mm_setcsr(saved_csr);
        result.elapsed_ms = seconds() * 1000;
        return 0;
    }

    // -- reading the solution -------------------------------------------------

    int node_info(int* kind, int* player, int* nact, int* to) const {
        const SNode& n = nodes_[cursor_];
        *kind = n.kind;
        *player = n.player;
        *nact = n.nact;
        for (int a = 0; a < 5; a++) to[a] = n.to[a];
        return 0;
    }

    int advance(int action) {
        const SNode& n = nodes_[cursor_];
        if (n.kind != PLAYER || action < 0 || action >= n.nact) return -1;
        cursor_ = n.child[action];
        return 0;
    }

    int deal(int card) {
        const SNode& n = nodes_[cursor_];
        if (n.kind != CHANCE) return -1;
        for (int r = 0; r < rivers_; r++)
            if (river_cards_[r] == card) {
                cursor_river_ = r;
                cursor_ = n.child[0];
                return 0;
            }
        return -1;
    }

    // The average strategy at the cursor, as out[action * 1326 + hole]. Zero for
    // hands that were dropped from the range or are blocked by the board.
    int strategy(float* out) const {
        const SNode& n = nodes_[cursor_];
        if (n.kind != PLAYER) return -1;
        const int p = n.player, count = n_[p], nact = n.nact;
        const float* average = table(n, cursor_river_) + size_t(nact) * count;
        std::memset(out, 0, sizeof(float) * nact * HOLES);
        for (int h = 0; h < HOLES; h++) {
            const int i = hand_of_[p][h];
            if (i < 0) continue;
            float total = 0;
            for (int a = 0; a < nact; a++) total += average[size_t(a) * count + i];
            for (int a = 0; a < nact; a++)
                out[size_t(a) * HOLES + h] = total > 0 ? average[size_t(a) * count + i] / total : 1.0f / float(nact);
        }
        return 0;
    }
};

}  // namespace

void* operator new(size_t size) {
    g_news.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

extern "C" {

SS_EXPORT void* ss_create(int threads) {
    return new (std::nothrow) Solver(std::max(1, threads));
}

SS_EXPORT int ss_reserve(void* solver, int megabytes) {
    return static_cast<Solver*>(solver)->reserve(size_t(megabytes) << 20) ? 0 : -5;
}

SS_EXPORT void ss_destroy(void* solver) { delete static_cast<Solver*>(solver); }

// 0 on success. -1 a bad state, -2 a bad range, -3 no possible pair of hands,
// -4 no tree fits, -5 out of memory, -6 an unexpected C++ exception.
SS_EXPORT int ss_solve(void* solver, const SsInput* in, const float* range0,
                                   const float* range1, SsResult* out) {
    if (!solver || !in || !range0 || !range1 || !out) return -1;
    if (in->board_count != 4 && in->board_count != 5) return -1;
    u64 seen = 0;
    for (int i = 0; i < in->board_count; i++) {
        if (in->board[i] < 0 || in->board[i] > 51 || (seen >> in->board[i] & 1)) return -1;
        seen |= 1ULL << in->board[i];
    }
    if (in->to_act != 0 && in->to_act != 1) return -1;
    if (in->budget_us <= 0 || in->big_blind <= 0 || in->last_raise <= 0 || in->max_table_mb <= 0) return -1;
    if (!(in->prune >= 0 && in->prune < 1)) return -1;
    for (int p = 0; p < 2; p++)
        if (in->stacks[p] < 0 || in->bets[p] < 0 || in->invested[p] < in->bets[p]) return -1;
    for (int h = 0; h < HOLES; h++)
        if (!(range0[h] >= 0 && range1[h] >= 0 && std::isfinite(range0[h]) && std::isfinite(range1[h])))
            return -2;
    try {
        return static_cast<Solver*>(solver)->solve(*in, range0, range1, *out);
    } catch (...) {
        return -6;
    }
}

SS_EXPORT int ss_node(void* solver, int* kind, int* player, int* nact, int* to) {
    return static_cast<Solver*>(solver)->node_info(kind, player, nact, to);
}
SS_EXPORT int ss_advance(void* solver, int action) { return static_cast<Solver*>(solver)->advance(action); }
SS_EXPORT int ss_deal(void* solver, int card) { return static_cast<Solver*>(solver)->deal(card); }
SS_EXPORT int ss_strategy(void* solver, float* out) { return static_cast<Solver*>(solver)->strategy(out); }
}

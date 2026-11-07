// External-sampling MCCFR (Lanctot et al. 2009) with linear weighting (Brown &
// Sandholm 2019), run as asynchronous self-play over shared flat tables.
//
// Worker threads read and write the tables with no locks, as Pluribus does. The
// relaxed atomics compile to plain moves on x86-64 and keep this defined
// behaviour; a racing add can lose an update, which sampling noise dwarfs.
#pragma once
#include <atomic>
#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>

#include "abstraction.hpp"
#include "tree.hpp"

struct Deal {
    u8 bucket[4][2];  // [street][player]
    int result;       // showdown: +1 player 0 wins, -1 player 1 wins, 0 split
};

// Samples a full deal up front: external sampling draws chance once per
// traversal, and in hold'em the cards do not depend on the betting.
class HoldemDealer {
    const Abstraction* abs;
    u8 deck[52];

public:
    explicit HoldemDealer(const Abstraction& abstraction) : abs(&abstraction) {
        for (int c = 0; c < 52; c++) deck[c] = u8(c);
    }

    void operator()(Rng& rng, Deal& d) {
        for (int i = 0; i < 9; i++) std::swap(deck[i], deck[i + rng.below(52 - i)]);
        const int a0 = deck[0], b0 = deck[1], a1 = deck[2], b1 = deck[3];
        u64 board = 0;
        u16 s0[4] = {}, s1[4] = {};
        add_card(s0, a0);
        add_card(s0, b0);
        add_card(s1, a1);
        add_card(s1, b1);
        d.bucket[0][0] = abs->preflop[hole_index(a0, b0)];
        d.bucket[0][1] = abs->preflop[hole_index(a1, b1)];
        for (int i = 4; i < 9; i++) {
            board |= 1ULL << deck[i];
            add_card(s0, deck[i]);
            add_card(s1, deck[i]);
            if (i >= 6) {
                int street = i - 5;
                d.bucket[street][0] = u8(abs->bucket(street, board, a0, b0));
                d.bucket[street][1] = u8(abs->bucket(street, board, a1, b1));
            }
        }
        const u32 r0 = evaluate7(s0), r1 = evaluate7(s1);
        d.result = r0 > r1 ? 1 : r0 < r1 ? -1 : 0;
    }
};

// Regret matching: play in proportion to positive regret, uniform if there is none.
inline void regret_match(const double* regret, int n, double* out) {
    double total = 0;
    for (int a = 0; a < n; a++) {
        double r = std::atomic_ref<double>(const_cast<double&>(regret[a])).load(std::memory_order_relaxed);
        out[a] = r > 0 ? r : 0;
        total += out[a];
    }
    if (total > 0)
        for (int a = 0; a < n; a++) out[a] /= total;
    else
        for (int a = 0; a < n; a++) out[a] = 1.0 / n;
}

inline void normalise(const double* sums, int n, double* out) {
    double total = 0;
    for (int a = 0; a < n; a++) total += sums[a];
    for (int a = 0; a < n; a++) out[a] = total > 0 ? sums[a] / total : 1.0 / n;
}

inline void relaxed_add(double& slot, double delta) {
    std::atomic_ref<double> ref(slot);
    ref.store(ref.load(std::memory_order_relaxed) + delta, std::memory_order_relaxed);
}

inline int sample_action(Rng& rng, const double* probs, int n) {
    double u = rng.unit();
    for (int a = 0; a < n - 1; a++)
        if ((u -= probs[a]) < 0) return a;
    return n - 1;
}

inline double terminal_value(const Node& n, const Deal& d, int player) {
    if (n.kind == FOLD) return n.player == player ? -n.to[0] : n.to[0];
    double value = double(n.to[0] * d.result);
    return player == 0 ? value : -value;
}

// Best == false: self-play. Best == true: learn a best response in `regret`
// while the opponent plays the fixed average strategy `fixed`.
template <bool Best>
struct Traversal {
    const Node* nodes;
    double* regret;
    double* average;  // accumulated in self-play, read-only when Best
    Rng rng;
    Deal deal{};

    double visit(int index, int traverser) {
        const Node& n = nodes[index];
        if (n.kind != DECISION) return terminal_value(n, deal, traverser);
        const int count = n.nact;
        const size_t slot = n.offset + size_t(deal.bucket[n.street][n.player]) * count;
        double probs[MAX_ACTIONS];
        if (n.player == traverser) {
            regret_match(regret + slot, count, probs);
            double values[MAX_ACTIONS], expected = 0;
            for (int a = 0; a < count; a++) {
                values[a] = visit(n.child[a], traverser);
                expected += probs[a] * values[a];
            }
            for (int a = 0; a < count; a++)
                relaxed_add(regret[slot + a], values[a] - expected);
            return expected;
        }
        if constexpr (Best) {
            normalise(average + slot, count, probs);
        } else {
            regret_match(regret + slot, count, probs);
            // Sampling the opponent's actions already weights this by its reach.
            for (int a = 0; a < count; a++) relaxed_add(average[slot + a], probs[a]);
        }
        return visit(n.child[sample_action(rng, probs, count)], traverser);
    }
};

// Runs `iterations` traversals per seat across threads, each with its own dealer.
template <bool Best, class Dealer>
void run_traversals(const Tree& tree, std::vector<double>& regret, std::vector<double>& average,
                    const Dealer& dealer, u64 iterations, int threads, u64 seed, int only_seat = -1) {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&, t] {
            Dealer my_dealer = dealer;
            Traversal<Best> run{tree.nodes.data(), regret.data(), average.data(),
                                Rng(seed * 1000003 + u64(t))};
            u64 mine = iterations / threads + (u64(t) < iterations % threads ? 1 : 0);
            for (u64 i = 0; i < mine; i++) {
                my_dealer(run.rng, run.deal);
                if (only_seat >= 0) {
                    run.visit(0, only_seat);
                } else {
                    run.visit(0, 0);
                    run.visit(0, 1);
                }
            }
        });
    for (auto& t : pool) t.join();
}

// Linear CFR: multiplying everything by t / (t + 1) after epoch t weights each
// epoch in proportion to t, which drops the poor early iterations far faster.
inline void discount(std::vector<double>& table, double factor) {
    double* p = table.data();
    const size_t n = table.size();
#pragma omp simd
    for (size_t i = 0; i < n; i++) p[i] *= factor;
}

// -- files --------------------------------------------------------------------

#pragma pack(push, 1)
struct StrategyHeader {
    char magic[8];
    u64 abstraction_id;
    u64 iterations;
    u64 epoch;
    u32 stack, small_blind, big_blind, node_count, entries;
    u32 buckets[4];
};
#pragma pack(pop)

// Write then rename: a crash mid-save cannot destroy the previous good file.
template <class Write>
void save_atomic(const std::string& path, Write write) {
    const std::string temp = path + ".tmp";
    FILE* f = std::fopen(temp.c_str(), "wb");
    if (!f) die("cannot write", temp.c_str());
    const bool ok = write(f);
    if (std::fclose(f) != 0 || !ok) die("failed writing", temp.c_str());
    // Windows refuses to replace a file a reader has mapped. Training must not
    // die for that: the data is safe in the .tmp, and the next save tries again.
    std::error_code error;
    std::filesystem::rename(temp, path, error);
    if (error)
        std::fprintf(stderr, "warning: could not replace %s (%s); latest data is in %s\n",
                     path.c_str(), error.message().c_str(), temp.c_str());
}

inline void save_checkpoint(const std::string& path, const StrategyHeader& header,
                            const std::vector<double>& regret, const std::vector<double>& average) {
    save_atomic(path, [&](FILE* f) {
        return std::fwrite(&header, sizeof header, 1, f) == 1 &&
               std::fwrite(regret.data(), 8, regret.size(), f) == regret.size() &&
               std::fwrite(average.data(), 8, average.size(), f) == average.size();
    });
}

inline StrategyHeader load_checkpoint(const std::string& path, std::vector<double>& regret,
                                      std::vector<double>& average) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot read", path.c_str());
    StrategyHeader header;
    if (std::fread(&header, sizeof header, 1, f) != 1 || std::memcmp(header.magic, "CFRCKP1", 8))
        die("not a checkpoint:", path.c_str());
    regret.resize(header.entries);
    average.resize(header.entries);
    const bool ok = std::fread(regret.data(), 8, regret.size(), f) == regret.size() &&
                    std::fread(average.data(), 8, average.size(), f) == average.size();
    std::fclose(f);
    if (!ok) die("truncated checkpoint:", path.c_str());
    return header;
}

// The average strategy as u16 probabilities, in the regret tables' layout, so
// the reader finds one infoset with a single multiply and add.
inline std::vector<u16> quantise(const Tree& tree, const u32* buckets, const std::vector<double>& average) {
    std::vector<u16> out(average.size());
    for (const Node& n : tree.nodes) {
        if (n.kind != DECISION) continue;
        for (u32 b = 0; b < buckets[n.street]; b++) {
            const size_t slot = n.offset + size_t(b) * n.nact;
            double probs[MAX_ACTIONS];
            normalise(&average[slot], n.nact, probs);
            for (int a = 0; a < n.nact; a++) out[slot + a] = u16(std::lround(probs[a] * 65535.0));
        }
    }
    return out;
}

inline void save_strategy(const std::string& path, StrategyHeader header, const Tree& tree,
                          const std::vector<double>& average) {
    std::memcpy(header.magic, "CFRSTR1", 8);
    const auto strategy = quantise(tree, header.buckets, average);
    save_atomic(path, [&](FILE* f) {
        return std::fwrite(&header, sizeof header, 1, f) == 1 &&
               std::fwrite(tree.nodes.data(), sizeof(Node), tree.nodes.size(), f) == tree.nodes.size() &&
               std::fwrite(strategy.data(), 2, strategy.size(), f) == strategy.size();
    });
}

// -- measuring ----------------------------------------------------------------

// One hand: `seat` plays its action with the highest cumulative regret, the
// other seat samples its average strategy.
inline double play_best_response(const Node* nodes, const double* best, const double* average,
                                 const Deal& d, int seat, Rng& rng) {
    int index = 0;
    while (nodes[index].kind == DECISION) {
        const Node& n = nodes[index];
        const size_t slot = n.offset + size_t(d.bucket[n.street][n.player]) * n.nact;
        int action = 0;
        if (n.player == seat) {
            for (int a = 1; a < n.nact; a++)
                if (best[slot + a] > best[slot + action]) action = a;
        } else {
            double probs[MAX_ACTIONS];
            normalise(average + slot, n.nact, probs);
            action = sample_action(rng, probs, n.nact);
        }
        index = n.child[action];
    }
    return terminal_value(nodes[index], d, seat);
}

// What seat wins per hand by best-responding to the average strategy, trained
// by regret matching against it. A lower bound: the responder is limited to the
// abstraction and to a finite number of iterations.
template <class Dealer>
double best_response_value(const Tree& tree, std::vector<double>& average, const Dealer& dealer,
                           int seat, u64 iterations, u64 samples, int threads, u64 seed) {
    std::vector<double> best(tree.entries, 0.0);
    run_traversals<true>(tree, best, average, dealer, iterations, threads, seed, seat);
    std::vector<double> totals(threads);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&, t] {
            Dealer my_dealer = dealer;
            Rng rng(seed * 7919 + u64(t) + 1);
            Deal d;
            double total = 0;
            for (u64 i = 0; i < samples / threads; i++) {
                my_dealer(rng, d);
                total += play_best_response(tree.nodes.data(), best.data(), average.data(), d, seat, rng);
            }
            totals[t] = total;
        });
    for (auto& t : pool) t.join();
    double total = 0;
    for (double x : totals) total += x;
    return total / double((samples / threads) * threads);
}

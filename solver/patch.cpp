// Background refinement of one blueprint spot: external-sampling MCCFR on the
// sub-tree after an opponent's exact bet, with finer sizes on that street.
//
// Threading contract: a job belongs to one thread at a time and shares nothing
// mutable. It reads two memory-mapped files and writes only its own tables. The
// caller copies the result out with ps_job_export, so nothing here is ever seen
// by the game loop while it is being written.
//
// Deals are drawn from the spot's real distribution by rejection: a uniform deal
// is kept with probability proportional to how likely both hands were to take
// this betting line under the blueprint.
//
// Built as a shared library with a C interface, for ctypes: see solver/README.md.
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define PS_EXPORT __declspec(dllexport)
#else
#define PS_EXPORT __attribute__((visibility("default")))
#endif

#include <cstring>
#include <iterator>
#include <memory>

#include "mccfr.hpp"

namespace {

constexpr int MAX_TRIES = 2000000;  // per deal: a line this unlikely is unusable
constexpr double REACH_FLOOR = 0.01;  // estimated ranges must not rule a hand out

// Reading the two files in place. Layouts: abstraction.hpp, mccfr.hpp.
struct MappedAbstraction {
    const AbstractionHeader* header;
    const u8* preflop;
    const u8* canon[3];  // u32s, not aligned: the file's sections are packed
    const u8* table[3];

    explicit MappedAbstraction(const u8* base) {
        header = reinterpret_cast<const AbstractionHeader*>(base);
        const u8* at = base + sizeof(AbstractionHeader);
        preflop = at;
        at += HOLES + 24 * 52;
        for (int s = 0; s < 3; s++) {
            canon[s] = at;
            at += size_t(BINOMIAL[52][s + 3]) * 4;
            table[s] = at;
            at += size_t(header->boards[s]) * HOLES;
        }
    }

    int bucket(int street, u64 board, int a, int b) const {
        if (street == 0) return preflop[hole_index(a, b)];
        u32 entry;
        std::memcpy(&entry, canon[street - 1] + size_t(board_rank(board)) * 4, 4);
        const auto& perm = SUIT_PERMS[entry & 31];
        return table[street - 1][size_t(entry >> 5) * HOLES + hole_index(perm[a], perm[b])];
    }

};

struct MappedStrategy {
    const StrategyHeader* header;
    const u8* nodes;  // Node[], packed
    const u8* probs;  // u16[], packed

    explicit MappedStrategy(const u8* base) {
        header = reinterpret_cast<const StrategyHeader*>(base);
        nodes = base + sizeof(StrategyHeader);
        probs = nodes + size_t(header->node_count) * sizeof(Node);
    }

    Node node(int index) const {
        Node n;
        std::memcpy(&n, nodes + size_t(index) * sizeof(Node), sizeof(Node));
        return n;
    }

    double prob(const Node& n, int bucket, int action) const {
        u16 raw;
        std::memcpy(&raw, probs + 2 * (n.offset + size_t(bucket) * n.nact + action), 2);
        return raw / 65535.0;
    }
};

}  // namespace

extern "C" {

struct PsStep {
    int node, action;
    int action2;    // -1, or a second action this step blends with action
    float weight;   // of action, when blending
};

struct PsRoot {
    int bets[2];
    int behind;       // each seat's chips from earlier streets
    int last_raise;
    int responder;    // the seat that faces the opponent's bet
    int street;
};

}  // extern "C"

namespace {

// Finer on the street of the spot, as the blueprint elsewhere. Tried in order
// until the tables fit: a preflop spot has far more tree under it than a turn one.
struct Resolution {
    float open[3], facing[2];
    int cap;
};
constexpr Resolution LADDER[] = {
    {{0.33f, 0.66f, 1.0f}, {0.5f, 1.0f}, 3},
    {{0.5f, 1.0f, 0.0f}, {1.0f, 0.0f}, 2},
    {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, 1},
};
constexpr float LATER_OPEN[3] = {0.5f, 1.0f, 0.0f};
constexpr float LATER_FACING[2] = {1.0f, 0.0f};
constexpr u64 MAX_ENTRIES = 4'000'000;  // 64 MB of tables and an 8 MB patch at most

class PatchBuilder {
    int stack_, big_blind_, root_street_;
    const u32* buckets_;
    const Resolution& res_;
    Tree& tree_;

    int terminal(Kind kind, int player, int amount) {
        Node n{};
        n.kind = kind;
        n.player = u8(player);
        n.to[0] = amount;
        tree_.nodes.push_back(n);
        return int(tree_.nodes.size()) - 1;
    }

    int build(int street, int player, int b0, int b1, int behind, int last, bool a0, bool a1, int raises) {
        const int bet[2] = {b0, b1};
        const bool acted[2] = {a0, a1};
        const int current = std::max(b0, b1), mine = bet[player];
        const int owed = current - mine, pot = 2 * behind + b0 + b1;
        const int all_in = stack_ - behind;
        const bool fine = street == root_street_;
        const float* fractions = owed > 0 ? (fine ? res_.facing : LATER_FACING) : (fine ? res_.open : LATER_OPEN);
        const int fraction_count = owed > 0 ? 2 : 3;

        int to[MAX_ACTIONS], count = 0;
        if (owed > 0) to[count++] = -1;
        to[count++] = 0;
        if (all_in > current) {
            const int min_to = std::min(current + last, all_in);
            for (int f = 0; f < fraction_count && raises < (fine ? res_.cap : 2); f++) {
                if (fractions[f] <= 0) continue;
                const int total = std::max(current + int(fractions[f] * float(pot + owed)), min_to);
                if (total < all_in && to[count - 1] != total) to[count++] = total;
            }
            to[count++] = all_in;  // past the raise cap too: a shove must have a node
        }

        const int index = int(tree_.nodes.size());
        tree_.nodes.emplace_back();
        const u32 offset = tree_.entries;
        const u64 entries = u64(tree_.entries) + u64(count) * buckets_[street];
        tree_.entries = entries > MAX_ENTRIES * 2 ? u32(MAX_ENTRIES * 2) : u32(entries);
        int child[MAX_ACTIONS] = {};
        for (int a = 0; a < count && tree_.entries < MAX_ENTRIES * 2; a++) {
            if (to[a] == -1) {
                child[a] = terminal(FOLD, player, behind + mine);
            } else if (to[a] == 0) {
                if (!acted[1 - player]) {
                    child[a] = player == 0 ? build(street, 1, current, b1, behind, last, true, a1, raises)
                                           : build(street, 0, b0, current, behind, last, a0, true, raises);
                } else if (behind + current >= stack_ || street == 3) {
                    child[a] = terminal(SHOWDOWN, 0, behind + current);
                } else {
                    child[a] = build(street + 1, 1, 0, 0, behind + current, big_blind_, false, false, 0);
                }
            } else {
                const int next_last = std::max(to[a] - current, last);
                child[a] = player == 0 ? build(street, 1, to[a], b1, behind, next_last, true, false, raises + 1)
                                       : build(street, 0, b0, to[a], behind, next_last, false, true, raises + 1);
            }
        }
        Node& n = tree_.nodes[index];
        n = Node{};
        for (int a = 0; a < count; a++) {
            n.child[a] = child[a];
            n.to[a] = to[a];
        }
        n.offset = offset;
        n.street = u8(street);
        n.player = u8(player);
        n.nact = u8(count);
        n.kind = DECISION;
        return index;
    }

public:
    PatchBuilder(const PsRoot& root, int stack, int big_blind, const u32* buckets, const Resolution& res, Tree& tree)
        : stack_(stack), big_blind_(big_blind), root_street_(root.street), buckets_(buckets), res_(res), tree_(tree) {
        // The opponent has just bet; the responder owes a decision and may raise.
        build(root.street, root.responder, root.bets[0], root.bets[1], root.behind, root.last_raise,
              root.responder == 1, root.responder == 0, 0);
    }
};

// Draws deals that could have reached the spot, by rejection on the betting line.
class LineDealer {
    MappedAbstraction abs_;
    MappedStrategy book_;
    struct Step {
        Node node;
        int action, action2;
        double weight, scale;  // scale: this step's largest reach over all buckets
    };
    std::vector<Step> steps_;
    u8 deck_[52];

    double reach(const Step& s, int bucket) const {
        double p = book_.prob(s.node, bucket, s.action);
        if (s.action2 >= 0) p = s.weight * p + (1 - s.weight) * book_.prob(s.node, bucket, s.action2);
        return std::max(p, REACH_FLOOR);
    }

public:
    bool exhausted = false;

    LineDealer(const u8* abstraction, const u8* strategy, const PsStep* path, int length)
        : abs_(abstraction), book_(strategy) {
        for (int c = 0; c < 52; c++) deck_[c] = u8(c);
        for (int i = 0; i < length; i++) {
            Step s{book_.node(path[i].node), path[i].action, path[i].action2, path[i].weight, 0};
            // Dividing by the best case keeps far more deals than plain rejection,
            // and the kept deals have exactly the same distribution.
            for (u32 b = 0; b < book_.header->buckets[s.node.street]; b++)
                s.scale = std::max(s.scale, reach(s, int(b)));
            steps_.push_back(s);
        }
    }

    void operator()(Rng& rng, Deal& d) {
        for (int attempt = 0; attempt < MAX_TRIES; attempt++) {
            for (int i = 0; i < 9; i++) std::swap(deck_[i], deck_[i + rng.below(52 - i)]);
            u64 boards[4] = {0, 0, 0, 0};
            u64 board = 0;
            for (int i = 4; i < 9; i++) {
                board |= 1ULL << deck_[i];
                if (i >= 6) boards[i - 5] = board;
            }
            int known = -1;
            auto reveal = [&](int street) {
                for (; known < street;) {
                    known++;
                    d.bucket[known][0] = u8(abs_.bucket(known, boards[known], deck_[0], deck_[1]));
                    d.bucket[known][1] = u8(abs_.bucket(known, boards[known], deck_[2], deck_[3]));
                }
            };
            bool keep = true;
            for (const Step& s : steps_) {
                reveal(s.node.street);
                if (rng.unit() * s.scale >= reach(s, d.bucket[s.node.street][s.node.player])) {
                    keep = false;
                    break;
                }
            }
            if (!keep) continue;
            reveal(3);
            u16 s0[4] = {}, s1[4] = {};
            add_card(s0, deck_[0]);
            add_card(s0, deck_[1]);
            add_card(s1, deck_[2]);
            add_card(s1, deck_[3]);
            for (int i = 4; i < 9; i++) {
                add_card(s0, deck_[i]);
                add_card(s1, deck_[i]);
            }
            const u32 r0 = evaluate7(s0), r1 = evaluate7(s1);
            d.result = r0 > r1 ? 1 : r0 < r1 ? -1 : 0;
            return;
        }
        exhausted = true;
    }
};

struct Job {
    Tree tree;
    std::vector<double> regret, average;
    LineDealer dealer;
    Traversal<false> run;
    u64 iterations = 0;
    u32 buckets[4];
    int level = 0;

    Job(const u8* abstraction, const u8* strategy, const PsStep* path, int length, u64 seed)
        : dealer(abstraction, strategy, path, length),
          run{nullptr, nullptr, nullptr, Rng(seed)} {}
};

// Lowers the priority only while the GIL is released: a low-priority thread that
// holds the GIL would make the game loop wait for it.
class BackgroundPriority {
#ifdef _WIN32
    HANDLE thread_ = GetCurrentThread();
    int saved_ = GetThreadPriority(thread_);

public:
    BackgroundPriority() { SetThreadPriority(thread_, THREAD_PRIORITY_BELOW_NORMAL); }
    ~BackgroundPriority() { SetThreadPriority(thread_, saved_); }
#endif
};

}  // namespace

extern "C" {

// A new job, or null if the spot cannot be solved: a tree too big even at the
// coarsest resolution, or a bad file or path.
PS_EXPORT void* ps_job_create(const void* abstraction, const void* strategy, const PsStep* path, int length,
                              const PsRoot* root, unsigned long long seed) {
    try {
        const auto* base_a = static_cast<const u8*>(abstraction);
        const auto* base_s = static_cast<const u8*>(strategy);
        if (std::memcmp(base_a, "CFRABS1", 8) != 0 || std::memcmp(base_s, "CFRSTR1", 8) != 0) return nullptr;
        const MappedStrategy book(base_s);
        if (root->responder != 0 && root->responder != 1) return nullptr;
        if (root->street < 0 || root->street > 3) return nullptr;
        for (int i = 0; i < length; i++) {
            if (path[i].node < 0 || u32(path[i].node) >= book.header->node_count) return nullptr;
            const Node n = book.node(path[i].node);
            if (n.kind != DECISION || path[i].action < 0 || path[i].action >= n.nact) return nullptr;
            if (path[i].action2 >= n.nact) return nullptr;
        }
        auto job = std::make_unique<Job>(base_a, base_s, path, length, seed);
        std::memcpy(job->buckets, book.header->buckets, sizeof job->buckets);
        bool fits = false;
        for (int level = 0; level < int(std::size(LADDER)) && !fits; level++) {
            job->tree = Tree{};
            job->level = level;
            PatchBuilder(*root, int(book.header->stack), int(book.header->big_blind), job->buckets, LADDER[level],
                         job->tree);
            fits = job->tree.entries <= MAX_ENTRIES;
        }
        if (!fits) return nullptr;
        job->regret.assign(job->tree.entries, 0.0);
        job->average.assign(job->tree.entries, 0.0);
        job->run.nodes = job->tree.nodes.data();
        job->run.regret = job->regret.data();
        job->run.average = job->average.data();
        return job.release();
    } catch (...) {
        return nullptr;
    }
}

PS_EXPORT void ps_job_destroy(void* job) { delete static_cast<Job*>(job); }

// Runs more iterations: 0, or -1 if deals for this line have become impossible.
PS_EXPORT int ps_job_run(void* opaque, unsigned long long iterations) {
    auto& job = *static_cast<Job*>(opaque);
    BackgroundPriority background;
    for (unsigned long long i = 0; i < iterations; i++) {
        job.dealer(job.run.rng, job.run.deal);
        if (job.dealer.exhausted) return -1;
        job.run.visit(0, 0);
        job.run.visit(0, 1);
    }
    job.iterations += iterations;
    return 0;
}

// Average regret per iteration summed over infosets, for both seats averaged, in
// chips. For a perfect-recall game this bounds exploitability from above.
PS_EXPORT double ps_job_regret(void* opaque) {
    const auto& job = *static_cast<const Job*>(opaque);
    if (job.iterations == 0) return 1e30;
    double total = 0;
    for (const Node& n : job.tree.nodes) {
        if (n.kind != DECISION) continue;
        for (u32 b = 0; b < job.buckets[n.street]; b++) {
            const double* r = &job.regret[n.offset + size_t(b) * n.nact];
            double best = 0;
            for (int a = 0; a < n.nact; a++) best = std::max(best, r[a]);
            total += best;
        }
    }
    return total / double(job.iterations) / 2;
}

PS_EXPORT int ps_job_level(void* job) { return static_cast<Job*>(job)->level; }
PS_EXPORT int ps_job_nodes(void* job) { return int(static_cast<Job*>(job)->tree.nodes.size()); }
PS_EXPORT int ps_job_entries(void* job) { return int(static_cast<Job*>(job)->tree.entries); }

// The finished patch: Node[nodes] then u16[entries], as in the strategy file.
PS_EXPORT void ps_job_export(void* opaque, void* out) {
    const auto& job = *static_cast<const Job*>(opaque);
    auto* at = static_cast<u8*>(out);
    std::memcpy(at, job.tree.nodes.data(), job.tree.nodes.size() * sizeof(Node));
    at += job.tree.nodes.size() * sizeof(Node);
    const auto probabilities = quantise(job.tree, job.buckets, job.average);
    std::memcpy(at, probabilities.data(), probabilities.size() * 2);
}

// For tests: counts of the preflop classes of accepted deals, per seat.
PS_EXPORT int ps_job_sample(void* opaque, int deals, int* counts0, int* counts1) {
    auto& job = *static_cast<Job*>(opaque);
    for (int i = 0; i < deals; i++) {
        job.dealer(job.run.rng, job.run.deal);
        if (job.dealer.exhausted) return -1;
        counts0[job.run.deal.bucket[0][0]]++;
        counts1[job.run.deal.bucket[0][1]]++;
    }
    return 0;
}
}

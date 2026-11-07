// Command line for the solver: see solver/README.md.
#include <chrono>
#include <cstring>
#include <map>
#include <string>

#include "mccfr.hpp"

namespace {

using Args = std::map<std::string, std::string>;

Args parse(int argc, char** argv) {
    Args args;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || std::strncmp(argv[i], "--", 2) != 0) die("expected --key value, got", argv[i]);
        args[argv[i] + 2] = argv[i + 1];
    }
    return args;
}

std::string text(const Args& args, const char* key, const char* fallback = nullptr) {
    auto it = args.find(key);
    if (it != args.end()) return it->second;
    if (!fallback) die("missing required option --", key);
    return fallback;
}

u64 number(const Args& args, const char* key, u64 fallback) {
    auto it = args.find(key);
    if (it == args.end()) return fallback;
    char* end;
    u64 value = std::strtoull(it->second.c_str(), &end, 10);
    if (*end || it->second[0] == '-') die("not a non-negative integer: --", key);
    return value;
}

int thread_count(const Args& args) {
    return int(std::max<u64>(1, number(args, "threads", std::thread::hardware_concurrency())));
}

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int cmd_abstract(const Args& args) {
    const double start = now();
    const auto abs = build_abstraction(int(number(args, "flop", 200)), int(number(args, "turn", 200)),
                                       int(number(args, "river", 200)), thread_count(args), number(args, "seed", 1));
    save_abstraction(abs, text(args, "out").c_str());
    std::printf("abstraction written in %.0f s\n", now() - start);
    return 0;
}

int cmd_train(const Args& args) {
    const auto abs = load_abstraction(text(args, "abs").c_str());
    const std::string out = text(args, "out");
    const std::string checkpoint = out + ".ckpt";
    const u64 target = number(args, "iters", 100'000'000);
    const u64 epoch_size = std::max<u64>(1, number(args, "epoch", std::max<u64>(target / 200, 1)));
    const int threads = thread_count(args);
    u64 save_seconds = number(args, "save-seconds", 600);

    Tree tree;
    TreeBuilder(int(number(args, "stack", 500)), int(number(args, "sb", 10)), int(number(args, "bb", 20)),
                abs.header.buckets, tree);
    StrategyHeader header{};
    std::memcpy(header.magic, "CFRCKP1", 8);
    header.abstraction_id = abs.header.id;
    header.stack = u32(number(args, "stack", 500));
    header.small_blind = u32(number(args, "sb", 10));
    header.big_blind = u32(number(args, "bb", 20));
    header.node_count = u32(tree.nodes.size());
    header.entries = tree.entries;
    std::memcpy(header.buckets, abs.header.buckets, sizeof header.buckets);

    std::vector<double> regret(tree.entries, 0.0), average(tree.entries, 0.0);
    if (text(args, "resume", "0") == "1") {
        const auto saved = load_checkpoint(checkpoint, regret, average);
        if (saved.abstraction_id != header.abstraction_id || saved.entries != header.entries ||
            saved.stack != header.stack || saved.small_blind != header.small_blind || saved.big_blind != header.big_blind)
            die("the checkpoint was made with another abstraction or game:", checkpoint.c_str());
        header.iterations = saved.iterations;
        header.epoch = saved.epoch;
        std::printf("resumed at %llu iterations\n", (unsigned long long)header.iterations);
    }
    std::printf("tree: %zu nodes, %u regret slots, %.1f MB of tables, %d threads\n", tree.nodes.size(), tree.entries,
                (regret.size() + average.size()) * 8.0 / 1e6, threads);
    std::fflush(stdout);

    const HoldemDealer dealer(abs);
    const u64 seed = number(args, "seed", 1);
    const double start = now();
    const u64 first = header.iterations;
    double last_save = start;
    while (header.iterations < target) {
        const u64 batch = std::min(epoch_size, target - header.iterations);
        run_traversals<false>(tree, regret, average, dealer, batch, threads, seed + header.epoch);
        header.iterations += batch;
        header.epoch++;
        const double factor = double(header.epoch) / double(header.epoch + 1);
        discount(regret, factor);
        discount(average, factor);
        const double t = now();
        if (t - last_save >= double(save_seconds)) {
            save_checkpoint(checkpoint, header, regret, average);
            save_strategy(out, header, tree, average);
            last_save = now();
        }
        std::printf("iter %llu  %.0f iter/s  %.0f s\n", (unsigned long long)header.iterations,
                    double(header.iterations - first) / (t - start), t - start);
        std::fflush(stdout);
    }
    save_checkpoint(checkpoint, header, regret, average);
    save_strategy(out, header, tree, average);
    std::printf("strategy written to %s\n", out.c_str());
    return 0;
}

int cmd_exploit(const Args& args) {
    const auto abs = load_abstraction(text(args, "abs").c_str());
    std::vector<double> regret, average;
    const auto header = load_checkpoint(text(args, "ckpt"), regret, average);
    if (header.abstraction_id != abs.header.id) die("the checkpoint was made with another abstraction");
    std::vector<double>().swap(regret);
    Tree tree;
    TreeBuilder(int(header.stack), int(header.small_blind), int(header.big_blind), abs.header.buckets, tree);
    const HoldemDealer dealer(abs);
    const u64 iterations = number(args, "iters", 20'000'000), samples = number(args, "samples", 20'000'000);
    const int threads = thread_count(args);
    const u64 seed = number(args, "seed", 99);
    const double v0 = best_response_value(tree, average, dealer, 0, iterations, samples, threads, seed);
    const double v1 = best_response_value(tree, average, dealer, 1, iterations, samples, threads, seed + 1);
    // In mbb per hand: 1000 * chips / big blind, averaged over the two seats.
    const double to_mbb = 1000.0 / header.big_blind;
    std::printf("trained for %llu iterations\n", (unsigned long long)header.iterations);
    std::printf("best response as button: %+.1f mbb/hand; as big blind: %+.1f mbb/hand\n", v0 * to_mbb, v1 * to_mbb);
    std::printf("abstract exploitability (lower bound): %.1f mbb/hand\n", (v0 + v1) / 2 * to_mbb);
    return 0;
}

// Reads seven card numbers a line from stdin, prints each hand's score.
int cmd_evalcheck() {
    int c[7];
    while (std::scanf("%d %d %d %d %d %d %d", c, c + 1, c + 2, c + 3, c + 4, c + 5, c + 6) == 7) {
        u16 suits[4] = {};
        for (int card : c) add_card(suits, card);
        std::printf("%u\n", evaluate7(suits));
    }
    return 0;
}

// Checks the parts that have a known right answer.
int cmd_selfcheck() {
    // 1. The sweep against brute force, on random river boards.
    Rng rng(7);
    EquitySweep sweep;
    std::vector<float> equity(HOLES);
    for (int trial = 0; trial < 20; trial++) {
        const u64 board = random_board(rng, 5);
        sweep.run(board, equity.data());
        u16 base[4] = {};
        for (u64 m = board; m; m &= m - 1) add_card(base, std::countr_zero(m));
        auto rank = [&](int h) {
            u16 s[4] = {base[0], base[1], base[2], base[3]};
            add_card(s, HOLE_CARDS[h][0]);
            add_card(s, HOLE_CARDS[h][1]);
            return evaluate7(s);
        };
        for (int h = 0; h < HOLES; h += 37) {
            const auto& c = HOLE_CARDS[h];
            if (board >> c[0] & 1 || board >> c[1] & 1) continue;
            double score = 0;
            int opponents = 0;
            for (int o = 0; o < HOLES; o++) {
                const auto& d = HOLE_CARDS[o];
                if (board >> d[0] & 1 || board >> d[1] & 1) continue;
                if (d[0] == c[0] || d[0] == c[1] || d[1] == c[0] || d[1] == c[1]) continue;
                const u32 a = rank(h), b = rank(o);
                score += a > b ? 1.0 : a == b ? 0.5 : 0.0;
                opponents++;
            }
            if (opponents != 990 || std::fabs(score / opponents - equity[h]) > 1e-5)
                die("sweep disagrees with brute force");
        }
    }

    // 2. MCCFR against Kuhn poker's known solution: the first player's value is
    // -1/18, and it bets a king exactly 3 times as often as a jack.
    Tree kuhn;
    auto terminal = [&](Kind kind, int player, int amount) {
        Node n{};
        n.kind = kind;
        n.player = u8(player);
        n.to[0] = amount;
        kuhn.nodes.push_back(n);
        return int(kuhn.nodes.size()) - 1;
    };
    auto decision = [&](int player, int check, int bet) {
        Node n{};
        n.player = u8(player);
        n.nact = 2;
        n.child[0] = check;
        n.child[1] = bet;
        n.offset = kuhn.entries;
        kuhn.entries += 6;
        kuhn.nodes.push_back(n);
        return int(kuhn.nodes.size()) - 1;
    };
    kuhn.nodes.emplace_back();  // the root must be index 0: filled in last
    int p0_facing_bet = decision(0, terminal(FOLD, 0, 1), terminal(SHOWDOWN, 0, 2));
    int p1_after_check = decision(1, terminal(SHOWDOWN, 0, 1), p0_facing_bet);
    int p1_facing_bet = decision(1, terminal(FOLD, 1, 1), terminal(SHOWDOWN, 0, 2));
    kuhn.nodes[0].nact = 2;
    kuhn.nodes[0].child[0] = p1_after_check;
    kuhn.nodes[0].child[1] = p1_facing_bet;
    kuhn.nodes[0].offset = kuhn.entries;
    kuhn.entries += 6;

    struct KuhnDealer {
        void operator()(Rng& r, Deal& d) const {
            int a = r.below(3), b = (a + 1 + r.below(2)) % 3;
            d.bucket[0][0] = u8(a);
            d.bucket[0][1] = u8(b);
            d.result = a > b ? 1 : -1;
        }
    };
    std::vector<double> regret(kuhn.entries, 0.0), average(kuhn.entries, 0.0);
    for (int epoch = 1; epoch <= 100; epoch++) {
        run_traversals<false>(kuhn, regret, average, KuhnDealer{}, 50'000, 2, u64(epoch));
        const double factor = double(epoch) / double(epoch + 1);
        discount(regret, factor);
        discount(average, factor);
    }
    double value = 0;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
            if (a == b) continue;
            Deal d{};
            d.bucket[0][0] = u8(a);
            d.bucket[0][1] = u8(b);
            d.result = a > b ? 1 : -1;
            struct Walk {
                const Tree& t;
                const std::vector<double>& avg;
                const Deal& d;
                double go(int i) const {
                    const Node& n = t.nodes[i];
                    if (n.kind != DECISION) return terminal_value(n, d, 0);
                    double probs[2];
                    normalise(&avg[n.offset + size_t(d.bucket[0][n.player]) * 2], 2, probs);
                    return probs[0] * go(n.child[0]) + probs[1] * go(n.child[1]);
                }
            };
            value += Walk{kuhn, average, d}.go(0) / 6.0;
        }
    double jack[2], king[2];
    normalise(&average[kuhn.nodes[0].offset], 2, jack);
    normalise(&average[kuhn.nodes[0].offset + 4], 2, king);
    std::printf("kuhn value %.5f (exact %.5f); bet K %.3f, bet J %.3f\n", value, -1.0 / 18, king[1], jack[1]);
    if (std::fabs(value + 1.0 / 18) > 0.003) die("kuhn value is wrong");
    if (std::fabs(king[1] - 3 * jack[1]) > 0.06) die("kuhn betting frequencies are wrong");
    std::printf("selfcheck ok\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string command = argc > 1 ? argv[1] : "";
    const Args args = parse(argc, argv);
    if (command == "abstract") return cmd_abstract(args);
    if (command == "train") return cmd_train(args);
    if (command == "exploit") return cmd_exploit(args);
    if (command == "evalcheck") return cmd_evalcheck();
    if (command == "selfcheck") return cmd_selfcheck();
    die("usage: cfr abstract|train|exploit|evalcheck|selfcheck [--key value ...]");
}

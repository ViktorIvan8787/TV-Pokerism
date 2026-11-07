// The abstract betting tree, built once into a flat array of nodes.
//
// Fold, call/check, 0.5x pot, 1x pot, all-in, under the rules of game_state.py
// for two equal stacks. tests/test_solver.py walks every node through the real
// GameState to check that they agree. An infoset needs no hash table:
//     regrets[node.offset + bucket * node.nact + action]
#pragma once
#include <algorithm>

#include "cards.hpp"

inline constexpr int MAX_ACTIONS = 5;
enum Kind : u8 { DECISION = 0, FOLD = 1, SHOWDOWN = 2 };

#pragma pack(push, 1)
struct Node {  // 48 bytes; the on-disk format too
    int32_t child[MAX_ACTIONS];
    // DECISION: per action, -1 fold, 0 call or check, else the raise-to total.
    // FOLD: to[0] is what the folder (player) loses. SHOWDOWN: what each risks.
    int32_t to[MAX_ACTIONS];
    u32 offset;  // first slot of this node's infosets
    u8 street, player, nact, kind;  // player 0 is the button (small blind)
};
#pragma pack(pop)
static_assert(sizeof(Node) == 48);

struct Tree {
    std::vector<Node> nodes;
    u32 entries = 0;  // total regret slots
};

class TreeBuilder {
    int stack, big_blind;
    const u32* buckets;
    Tree& tree;

    int terminal(Kind kind, int player, int amount) {
        Node n{};
        n.kind = kind;
        n.player = u8(player);
        n.to[0] = amount;
        tree.nodes.push_back(n);
        return int(tree.nodes.size()) - 1;
    }

    // bets: this street; behind: each player's chips from earlier streets.
    int build(int street, int player, int b0, int b1, int behind, int last_raise,
              bool acted0, bool acted1) {
        const int bet[2] = {b0, b1};
        const bool acted[2] = {acted0, acted1};
        const int current = std::max(b0, b1), mine = bet[player];
        const int owed = current - mine, pot = 2 * behind + b0 + b1;
        const int all_in = stack - behind;  // raise-to total that uses every chip

        int to[MAX_ACTIONS], count = 0;
        if (owed > 0) to[count++] = -1;
        to[count++] = 0;
        if (all_in > current) {
            int min_to = std::min(current + last_raise, all_in);
            int half = std::max(current + (pot + owed) / 2, min_to);
            int full = std::max(current + pot + owed, min_to);
            if (half < all_in) to[count++] = half;
            if (full < all_in && full != half) to[count++] = full;
            to[count++] = all_in;
        }

        const int index = int(tree.nodes.size());
        tree.nodes.emplace_back();
        const u32 offset = tree.entries;
        tree.entries += u32(count) * buckets[street];
        int child[MAX_ACTIONS] = {};
        for (int a = 0; a < count; a++) {
            if (to[a] == -1) {
                child[a] = terminal(FOLD, player, behind + mine);
            } else if (to[a] == 0) {
                if (!acted[1 - player]) {  // the other player still has a say
                    child[a] = player == 0
                                   ? build(street, 1, current, b1, behind, last_raise, true, acted1)
                                   : build(street, 0, b0, current, behind, last_raise, acted0, true);
                } else if (behind + current >= stack || street == 3) {
                    child[a] = terminal(SHOWDOWN, 0, behind + current);
                } else {  // street closed: the player after the button opens
                    child[a] = build(street + 1, 1, 0, 0, behind + current, big_blind, false, false);
                }
            } else {
                int raise = to[a] - current;
                int next_last = std::max(raise, last_raise);  // short all-ins do not count
                child[a] = player == 0
                               ? build(street, 1, to[a], b1, behind, next_last, true, false)
                               : build(street, 0, b0, to[a], behind, next_last, false, true);
            }
        }
        Node& n = tree.nodes[index];  // recursion may have moved the vector
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
    TreeBuilder(int stack_, int small_blind, int big_blind_, const u32* buckets_, Tree& tree_)
        : stack(stack_), big_blind(big_blind_), buckets(buckets_), tree(tree_) {
        if (small_blind < 1 || small_blind > big_blind || stack <= big_blind)
            die("need 1 <= small blind <= big blind < stack");
        // The button acts first preflop; the big blind keeps its option.
        build(0, 0, small_blind, big_blind, 0, big_blind, false, false);
    }
};

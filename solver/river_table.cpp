// Replaces the 170 MB river bucket table with score -> bucket, for the submission.
// A river bucket depends only on EHS, and EHS is (2 * wins + ties) / 1980 exactly.
//
//     river_table abstraction.bin river_table.bin
#include <cmath>
#include <cstring>

#include "abstraction.hpp"

int main(int argc, char** argv) {
    if (argc != 3) die("usage: river_table abstraction.bin out.bin");
    const Abstraction abs = load_abstraction(argv[1]);
    std::vector<bool> seen(abs.header.boards[2]);
    int table[1981];
    std::fill(std::begin(table), std::end(table), -1);
    long long points = 0, conflicts = 0;
    EquitySweep sweep;
    std::vector<float> equity(HOLES);
    for (int a = 4; a < 52; a++)
        for (int b = 3; b < a; b++)
            for (int c = 2; c < b; c++)
                for (int d = 1; d < c; d++)
                    for (int e = 0; e < d; e++) {
                        const u64 board = 1ULL << a | 1ULL << b | 1ULL << c | 1ULL << d | 1ULL << e;
                        const u32 id = abs.canon[2][board_rank(board)] >> 5;
                        if (seen[id]) continue;  // suit-isomorphic boards share a row
                        seen[id] = true;
                        sweep.run(board, equity.data());
                        for (int h = 0; h < HOLES; h++) {
                            const int x = HOLE_CARDS[h][0], y = HOLE_CARDS[h][1];
                            if (board >> x & 1 || board >> y & 1) continue;
                            const int score = int(std::lround(equity[h] * 1980.0));
                            const int bucket = abs.bucket(3, board, x, y);
                            points++;
                            if (table[score] >= 0 && table[score] != bucket) conflicts++;
                            table[score] = bucket;
                        }
                    }
    int used = 0;
    for (int s = 0; s <= 1980; s++) used += table[s] >= 0;
    int last = 0;
    for (int s = 0; s <= 1980; s++) {
        if (table[s] < 0) table[s] = last;  // a score no hand reaches
        if (table[s] < last) conflicts++;   // buckets must rise with strength
        last = table[s];
    }
    std::printf("%lld points, %d scores, %d top bucket, %lld conflicts\n", points, used, last, conflicts);
    if (conflicts) die("river buckets are not a function of EHS");
    FILE* f = std::fopen(argv[2], "wb");
    if (!f) die("cannot write", argv[2]);
    for (int s = 0; s <= 1980; s++) std::fputc(table[s], f);
    return std::fclose(f) == 0 ? 0 : 1;
}

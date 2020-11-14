#include <cstring>
#include <cassert>
#include <array>
#include <vector>
#include <sstream>
#include <iostream>
#include <torch/torch.h>

#pragma once

namespace ttt {

// Values selected in such a way that turn is changed by XOR with BOTH.
enum : int {
    EMPTY,	//= UNDECIDED
    X,		//= P1_WON
    O,		//= P2_WON
    BOTH	//= DRAW
};

using Move = int;
using MoveList = std::vector<Move>;
using State = std::array<int, 9>;

struct Game {

    static constexpr int ACTION_SPACE_SIZE = 9;
    static constexpr int REWARD_PERSPECTIVE = X;
    static constexpr Move NO_MOVE = 9;

    static constexpr int to_nn_idx(Move action) { return action; }
    static constexpr Move to_action(int index) { return index; }

    Game() {history.emplace_back();}
    Game(const std::array<int, 9> &board_) {history.push_back(board_);}

    void trim(size_t size) 
    {
        history.resize(history.size() - size);
        result = EMPTY;
        if (size & 1)
            turn ^= BOTH;
    }
    void act(Move move);
    void reset();
    void test();
    bool legal(Move move) const;
    bool terminal();
    int to_play() const { return turn; }
    int get_result() const;
    float get_value() const;

    std::string str() const;

    size_t size() const { return history.size(); }

    MoveList get_actions() const;
    torch::Tensor to_input_planes() const;

private:
    int turn = X;
    int result = EMPTY;
    std::vector<std::array<int, 9>> history;
};

} // namespace
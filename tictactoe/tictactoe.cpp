#include "tictactoe.h"

namespace ttt {

// Without check for legality. User must check isLegal() himself if needed.
void Game::act(Move move) 
{
    State new_state = history.back();

    new_state[move] = turn;
    turn ^= BOTH;

    history.push_back(new_state);
}

bool Game::terminal()
{
    result = get_result();
    return result != EMPTY;
}

// isTerminal() must be called somewhere before this function, to ensure game result is refreshed.
float Game::get_value() const
{
    switch (result) {
        case X: return 1.0f;
        case O: return -1.0f;
        default: return 0.0f;
    }
}

MoveList Game::get_actions() const 
{
    MoveList moves;
    auto board = history.back();

    for (int i = 0; i < ACTION_SPACE_SIZE; ++i)
        if (board[i] == EMPTY)
            moves.push_back(i);

    return moves;
}

torch::Tensor Game::to_input_planes() const 
{
    // 4-th dim is for batches.
    torch::Tensor output = torch::zeros({1, 3, 3, 3});
    auto accessor = output.accessor<float, 4>();
    auto board = history.back();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            switch (board[i*3+j]) {
                case EMPTY:
                    break;
                case X:
                    accessor[0][0][i][j] = 1.0f;
                    break;
                case O:
                    accessor[0][1][i][j] = 1.0f;
                    break;
                default:
                    assert(false);
            }
        }
    }

    if (turn == X) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                accessor[0][2][i][j] = 1.0f;
    }

    return output;
}

// Make sure to refresh game result before printing.
std::string Game::str() const
{
    auto board = history.back();
    std::stringstream ss;
    ss << '\n';

    for (int row = 0; row < 3; ++row) {
        ss << "-------------\n";
        for (int col = 0; col < 3; ++col) {
            ss << '|';
            switch(board[row*3 + col]) {
                case X: ss << " X "; break;
                case O: ss << " O "; break;
                case EMPTY: ss << "   "; break;
                default: assert(false);
            }
        }
        ss << "|\n";
    }
    ss << "-------------\n";
    ss << "\nturn:\t" << (turn == X ? 'X' : 'O') << "\nwinner:\t";
    switch (result) {
        case EMPTY:	ss << '-'; break;
        case X:		ss << 'X'; break;
        case O:		ss << 'O'; break;
        case BOTH:	ss << "draw"; break;
    }
    return ss.str();
}

bool Game::legal(Move move) const 
{
    if (move < 0 || move >= 9)
        return false;

    return history.back()[move] == EMPTY;
}

int Game::get_result() const
{
    // Define player who just made move.
    int player = turn ^ BOTH;
    auto board = history.back();

    for (int cell = 0; cell < 9; cell += 3)
        if ((board[cell] & board[cell+1] & board[cell+2]) == player)
            return player;

    for (int cell = 0; cell < 3; ++cell)
        if ((board[cell] & board[cell+3] & board[cell+6]) == player)
            return player;

    if ((board[0] & board[4] & board[8]) == player)
        return player;

    if ((board[2] & board[4] & board[6]) == player)
        return player;

    for (auto cell : board)
        if (cell == EMPTY)
            return EMPTY;

    return BOTH;
}

void Game::reset() 
{
    history.clear();
    history.emplace_back();
    turn = X;
    result = EMPTY;
}

void Game::test() 
{
    Move move = 9;
    std::cout << str() << std::endl;

    while (result == EMPTY) {
        std::cout << "\nMake move: " << std::flush;
        std::cin >> move;

        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "\nIncorrect input" << std::endl;
        } else if (!legal(move)) {
            std::cout << "\nIllegal move" << std::endl;
        } else {
            act(move);
        }
        result = get_result();
        std::cout << str() << std::endl;
    };
}

} // namespace
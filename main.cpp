#include <iostream>

#include "neural/alphazero.h"
#include "tictactoe/tictactoe.h"

int main(int argc, char **argv) 
{
    bool restore = false;

    std::string weights;

    if (argc <= 1) {
        std::cout << "No arguments provided - exiting." << std::endl;
        std::exit(-1);
    }
    
    if (argc >= 4) {
        if (strcmp(argv[2], "restore") == 0) {
            restore = true;
            weights = std::string(argv[3]);
        }
    }

    if (argc >= 2) {

        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        const auto rp_op = az::ReplayBufferOptions{512, 10000};
        const auto net_op = az::NetworkOptions{3, 3, 3, 128, 2, 9};
        const auto op = az::AlphaZeroOptions{1000, 100, 2, 10, 2.0, 0.25, 0.3, 1.0, 2, 10, 100, -1, device};

        auto alpha = az::AlphaZero<ttt::Game, ttt::Move>(op, net_op, rp_op, weights);

        if (strcmp(argv[1], "train") == 0)
            alpha.start_train(restore);
        else if (strcmp(argv[1], "test") == 0)
            alpha.human_test(restore);
    }
    return 0;
}

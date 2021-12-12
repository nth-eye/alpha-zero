# alpha-zero

AlphaZero implementation using C++ and libtorch library. For testing was written small TicTacToe game also in C++.
Program is divided in 4 main files: 
1. "network.h" - network architecture
2. "alphazero.h" - alphazero algorithm
3. "replay_buffer.h" - class for holding played games and providing samples
4. "node.h" - node in ongoing game tree structure

## How to use

Simply call with either "train" or "test" argument and optionally provide checkpoint with weights from where to start.
Example: ```./alpha_zero train ttt_chk.pt``` 

## TODO
- [ ] Virtual loss and parallel playing
- [ ] Test on more games

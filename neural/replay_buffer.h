#include <deque>
#include <random>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <execution>
#include <torch/torch.h>

#pragma once

namespace az {

struct ReplayBufferOptions {
    int batch_size = 4096;
    int window_size = 1e6;	// Max buffer size.
};

// State, policy and value.
using Sample = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

class ReplayBuffer {
public:
    ReplayBuffer(const ReplayBufferOptions &op_);

    bool ready(int n_batches) const {return buffer.size() > op.batch_size*n_batches;}
    void save_game(std::vector<Sample> &&samples);
    Sample sample_batch();
private:
    const ReplayBufferOptions op;

    std::mutex mut;
    std::mt19937 rng;
    std::deque<Sample> buffer;
};


ReplayBuffer::ReplayBuffer(const ReplayBufferOptions &op_) :
    op(op_),
    rng(static_cast<std::uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()))
{}

void ReplayBuffer::save_game(std::vector<Sample> &&samples)
{
    std::lock_guard lock(mut);
    std::move(std::execution::par_unseq, samples.begin(), samples.end(), std::back_inserter(buffer));

    if (buffer.size() > op.window_size)
        buffer.erase(buffer.begin(), buffer.begin() + (buffer.size() - op.window_size));
}

Sample ReplayBuffer::sample_batch()
{
    std::uniform_int_distribution<long> d(0, buffer.size()-1);

    std::vector<torch::Tensor> states;		states.reserve(op.batch_size);
    std::vector<torch::Tensor> policies;	policies.reserve(op.batch_size);
    std::vector<torch::Tensor> values;		values.reserve(op.batch_size);
    {
    std::lock_guard lock(mut);
    for (int i = 0; i < op.batch_size; ++i) {

        auto [state, p_label, v_label] = buffer[d(rng)];

        states.push_back(std::move(state));
        policies.push_back(std::move(p_label));
        values.push_back(std::move(v_label));
    }
    }
    return {torch::cat(states, 0), torch::cat(policies, 0), torch::cat(values, 0)};
}

} // namespace
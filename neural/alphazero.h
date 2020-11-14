#include <atomic>
#include <shared_mutex>
#include <algorithm>
#include <execution>

#include "node.h"
#include "network.h"
#include "replay_buffer.h"

namespace az {

/**
* \brief Different configs for AZ.
* \param training_steps Epochs during training section execution.
* \param checkpoint_interval Save network weights every N batches.
* \param num_actors Num of threads to fill replay buffer.
* \param games_per_epoch Num of games which all actors must play in total before next epoch.
* \param c_puct Coefficient which controls degree of exploration.
* \param epsilon Epsilon for Dirichlet noise.
* \param alpha Alpha for Dirichlet noise.
* \param temperature Temperature for child selection. Currently not used.
* \param sampling_moves When temperature drops to 0.
* \param max_moves Max moves per game during selfplay.
* \param playouts Num of expansions in MCTS for single step.
* \param milliseconds Time for MCTS expansion if playouts is 0.
* \param device Device on which model will be.
*/
struct AlphaZeroOptions {
    int training_steps = 700e3;
    int checkpoint_interval = 1000;
    int num_actors = 3000;
    int games_per_epoch = 64000;

    float c_puct = 1.25f;
    float epsilon = 0.25f;
    float alpha = 0.3f;
    float temperature = 1.0f;

    int sampling_moves = 30;
    int max_moves = 512;
    int playouts = 800;
    int milliseconds = -1;

    torch::Device device = torch::kCPU;
};


template<class Game, class Action>
class AlphaZero {
public:
    AlphaZero(const AlphaZeroOptions &op_, 
            const NetworkOptions &net_op, 
            const ReplayBufferOptions &rb_op, 
            const std::string &weights_file_);

    void start_train(bool restore);
    void human_test(bool restore);
private:
    void train();
    void selfplay();

    auto play_game();
    void run_mcts(const std::unique_ptr<Node<Action>> &root, Game &game);
    void playout(const std::unique_ptr<Node<Action>> &root, Game &game);

    auto get_probabilities(const std::unique_ptr<Node<Action>> &node) const;
    void add_exploration_noise(const std::unique_ptr<Node<Action>> &node);
    int select_action_idx(const std::vector<float> &probs, Game &game);

    auto select(Node<Action> *node);
    float evaluate(Node<Action> *node, Game &game);
    void propagate(float value, const std::vector<Node<Action>*> &search_path) const;

    const AlphaZeroOptions op;
    AlphaZeroNetwork network;
    ReplayBuffer replay_buffer;
    torch::optim::SGD optimizer;

    std::string weights_file;

    std::mutex mut;
    std::atomic_bool train_flag = false;
    std::atomic_uint idle_actors;
    std::atomic_uint games_played = 0;
    std::condition_variable cv_train;
    std::condition_variable cv_play;

    std::mt19937 rng{static_cast<std::uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};
};


template<class Game, class Action>
AlphaZero<Game, Action>::AlphaZero(	const AlphaZeroOptions &op_, 
                                    const NetworkOptions &net_op, 
                                    const ReplayBufferOptions &rb_op, 
                                    const std::string &weights_file_) :
    op(op_),
    network(net_op),
    replay_buffer(rb_op),
    optimizer(network->parameters(), torch::optim::SGDOptions(0.2).weight_decay(1e-4).momentum(0.9)),
    weights_file(weights_file_)
{
    network->to(op_.device);
}

// Training consisnt of 2 phases.
// 1) Each actor generates given amount of games and saves them to buffer. After this threads are paused. 
// 2) Training thread goes through one training epochs which is equal to [checkpoint_interval] batches.
//    Once epoch is done, training thread is paused and playing threads resumed.
template<class Game, class Action>
void AlphaZero<Game, Action>::start_train(bool restore)
{
    train_flag = true;

    if (restore)
        torch::load(network, weights_file);

    std::vector<std::jthread> threads; threads.reserve(op.num_actors);

    idle_actors = op.num_actors;

    for (int i = 0; i < op.num_actors; ++i)
        threads.emplace_back(&AlphaZero::selfplay, this);

    train();
}

template<class Game, class Action>
void AlphaZero<Game, Action>::human_test(bool restore)
{
    if (restore)
        torch::load(network, weights_file);

    Game game = Game();
    std::unique_ptr<Node<Action>> root = std::make_unique<Node<Action>>(1.0f);
    evaluate(root.get(), game);

    int first = game.to_play();
    int player = first;

    while (!game.terminal()) {

        std::cout << game.str() << std::endl;

        Action action = Game::NO_MOVE;
        int action_idx = 0;

        if (player == first) {

            std::cout << "\nYour turn: " << std::flush;

            while (true) {

                std::cin >> action;

                if (std::cin.fail()) {
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "\nIncorrect input\n" << std::endl;
                } else if (!game.legal(action)) {
                    std::cout << "\nIllegal move\n" << std::endl;
                } else {
                    break;
                }
            };
            auto actions = game.get_actions();

            for (int i = 0; i < actions.size(); ++i) {
                if (actions[i] == action) {
                    action_idx = i;
                    break;
                }
            }
        } else {
            add_exploration_noise(root);		
            run_mcts(root, game);

            auto [actions, probs] = get_probabilities(root);
            action_idx = select_action_idx(probs, game);
            action = actions[action_idx];

            std::cout << "\nAlphaZero has moved\n" << std::endl;
        }
        root = root->remove_child(action_idx);

        game.act(action);
        player = game.to_play();
    }
    std::cout << game.str() << std::endl;
}

template<class Game, class Action>
void AlphaZero<Game, Action>::train()
{
    const int epochs = op.training_steps / op.checkpoint_interval;

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1s);

    for (int i = 1; i < epochs+1; ++i) {

        auto batch_v_loss = torch::zeros({1}, op.device);
        auto batch_p_loss = torch::zeros({1}, op.device);

        // Resume game generation phase.
        cv_play.notify_all();

        // Imitation of barrier. Waiting till all playing threads finish generation phase.
        {
        std::unique_lock lock(mut);
        cv_train.wait(lock);
        }

        for (int batch = 1; batch < op.checkpoint_interval+1; ++batch) {

            network->zero_grad();

            auto [data, p_label, v_label] = replay_buffer.sample_batch();

            data = data.to(op.device);
            p_label = p_label.to(op.device);
            v_label = v_label.to(op.device);

            auto [v_out, p_out] = network->forward(data);

            auto v_loss = torch::mse_loss(v_out, v_label);
            auto p_loss = torch::mean(-torch::sum(p_label *  torch::log_softmax(p_out, 1), 1));

            auto total_loss = v_loss + p_loss;

            batch_v_loss += v_loss;
            batch_p_loss += p_loss;

            total_loss.backward();
            optimizer.step();

            std::printf("\r[%2d/%2d][%3d/%3d]", i, epochs, batch, op.checkpoint_interval);
        }
        // Save network, log metrics and test if needed.
        std::printf(" v_loss: %.4f | p_loss: %.4f | total: %.4f | games_played: %3d -> checkpoint %d\n",
            batch_v_loss.template item<float>()/op.checkpoint_interval,
            batch_p_loss.template item<float>()/op.checkpoint_interval,
            (batch_v_loss+batch_p_loss).template item<float>()/op.checkpoint_interval,
            games_played.load(),
            i
        );
        torch::save(network, "checkpoints/alphazero_checkpoint_"+std::to_string(i)+".pt");
    }
    train_flag = false;
    cv_play.notify_all();
}

template<class Game, class Action>
void AlphaZero<Game, Action>::selfplay()
{
    // Thread local gradient guard.
    torch::NoGradGuard no_grad_guard;

    const int games_per_actor = op.games_per_epoch / op.num_actors;

    {
    std::unique_lock lock(mut);
    cv_play.wait(lock);
    }
    while (train_flag) {

        --idle_actors;
        for (int i = 0; i < games_per_actor; ++i) {
            auto temp = play_game();
            replay_buffer.save_game(std::move(temp));
        }
        games_played += games_per_actor;
        ++idle_actors;

        if (idle_actors == op.num_actors)
            cv_train.notify_one();
        
        std::unique_lock lock(mut);
        cv_play.wait(lock);
    }
}

template<class Game, class Action>
auto AlphaZero<Game, Action>::play_game()
{
    // Create game, tree root and expand it.
    Game game = Game();
    std::unique_ptr<Node<Action>> root = std::make_unique<Node<Action>>(1.0f);
    evaluate(root.get(), game);

    // Samples for training: state, probabilities and value.
    std::vector<Sample> samples;

    while (!game.terminal() && game.size() < op.max_moves) {

        // Add noise and perform playouts from the current root.
        add_exploration_noise(root);
        run_mcts(root, game);

        auto [actions, probs] = get_probabilities(root);

        // Reserve memory for policy tensor.
        auto policy = torch::zeros({1, Game::ACTION_SPACE_SIZE});
        auto pol_ac = policy.template accessor<float, 2>();

        // Fill policy tensor with probability labels.
        for (int i = 0; i < actions.size(); ++i)
            pol_ac[0][Game::to_nn_idx(actions[i])] = probs[i];
    
        // Add to samples for buffer.
        samples.emplace_back(game.to_input_planes(), policy, torch::empty({1, 1}));

        // Select action from probabilities and replace root with new.
        int idx = select_action_idx(probs, game);
        root = root->remove_child(idx);

        game.act(actions[idx]);
    }

    float value = game.get_value();
    // Add the winner to every saved tuple.
    for (auto &v_sample : samples)
        std::get<2>(v_sample)[0][0] = value;

    return samples;
}

template<class Game, class Action>
void AlphaZero<Game, Action>::run_mcts(const std::unique_ptr<Node<Action>> &root, Game &game)
{
    for (int i = 0; i < op.playouts; ++i)
        playout(root, game);
}

template<class Game, class Action>
void AlphaZero<Game, Action>::playout(const std::unique_ptr<Node<Action>> &root, Game &game)
{
    std::vector<Node<Action>*> search_path; 

    Action action = Game::NO_MOVE;
    Node<Action> *node = root.get();
    int n = 0;
    search_path.push_back(node);

    while (node->expanded()) {

        std::tie(action, node) = select(node);
        game.act(action);

        ++n;
        search_path.push_back(node);
    }

    float value;

    if (game.terminal())
        value = game.get_value();
    else
        value = evaluate(node, game);
    game.trim(n);

    propagate(value, search_path);
}

template<class Game, class Action>
auto AlphaZero<Game, Action>::get_probabilities(const std::unique_ptr<Node<Action>> &node) const
{
    std::vector<Action> actions;
    std::vector<float> probs;

    for (auto &[action, child] : node->children) {
        actions.push_back(action);
        probs.push_back(static_cast<float>(child->n)/node->n);
    }

    return std::make_pair(actions, probs);
} 

template<class Game, class Action>
void AlphaZero<Game, Action>::add_exploration_noise(const std::unique_ptr<Node<Action>> &node)
{
    std::gamma_distribution<float> dist(op.alpha, 1.0f);
    std::vector<float> dirichlet_noise(node->children.size(), 0.0f);

    float sum = 0.0f;

    for (auto &it : dirichlet_noise)
        sum += (it = dist(rng));

    for (auto &it : dirichlet_noise)
        it /= sum + 1e-8f;

    for (int i = 0; i < node->children.size(); ++i) {
        node->children[i].second->p *= (1.0f - op.epsilon);
        node->children[i].second->p += dirichlet_noise[i] * op.epsilon;
    }
}

template<class Game, class Action>
int AlphaZero<Game, Action>::select_action_idx(const std::vector<float> &probs, Game &game)
{
    // When temperature is 0 - select child with most visits
    // When temperature is 1 - select from distribution proportional to visits

    if (game.size() > op.sampling_moves)
        return std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));

    return std::discrete_distribution<int>(probs.begin(), probs.end())(rng);
}

// Select the child with the highest UCB score.
template<class Game, class Action>
auto AlphaZero<Game, Action>::select(Node<Action> *node)
{
    float best_value = -std::numeric_limits<float>::infinity();
    Node<Action> *best_child = nullptr;
    Action best_action = Game::NO_MOVE;

    for (auto &[act, child] : node->children) {

        float uct = child->exploitation_term() + op.c_puct * sqrtf(node->n) * child->exploration_term();
        
        if (uct > best_value) {
            best_value = uct;
            best_child = child.get();
            best_action = act;
        }
    }
    return std::make_pair(best_action, best_child);
}

// Evaluate and expand.
template<class Game, class Action>
float AlphaZero<Game, Action>::evaluate(Node<Action> *node, Game &game)
{
    auto [value, policy_logits] = network->forward(game.to_input_planes().to(op.device));
    auto legal_actions = game.get_actions();

    // Reserving memory for probabilities for legal actions only.
    auto policy = torch::empty(legal_actions.size());
    auto pol_ac = policy.template accessor<float, 1>();

    for (int i = 0; i < legal_actions.size(); ++i)
        pol_ac[i] = policy_logits[0][Game::to_nn_idx(legal_actions[i])].template item<float>();

    policy = policy.softmax(0);
    pol_ac = policy.template accessor<float, 1>();

    for (int i = 0; i < legal_actions.size(); ++i)
        node->children.emplace_back(legal_actions[i], std::make_unique<Node<Action>>(pol_ac[i], game.to_play()));

    return value.template item<float>();
}

// Backpropagate value through search path.
template<class Game, class Action>
void AlphaZero<Game, Action>::propagate(float value, const std::vector<Node<Action>*> &search_path) const
{
    for (auto node : search_path) {
        node->n += 1;
        node->w += node->turn == Game::REWARD_PERSPECTIVE ? value : -value;
    }
}

} // namespace
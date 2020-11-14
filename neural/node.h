#include <memory>
#include <vector>

#pragma once

namespace az {

template<class Action>
struct Node {

    Node(const float p_) : p(p_) {}
    Node(const float p_, const int turn_) : p(p_), turn(turn_) {}

    float exploration_term() const
    {
        return p / (1.0f + static_cast<float>(n));
    }
    float exploitation_term() const
    {
        if (!n) return 0.0;
        return (w) / n;
    }

    bool expanded() 
    {
        //std::lock_guard lock(mut);
        return children.size() > 0;
    }
    auto remove_child(int idx)
    {
        auto new_root = std::move(children.at(idx));
        children.erase(children.begin()+idx);

        return std::move(new_root.second);
    }
    
    float p = 0.0f;
    float w = 0.0f;
    int turn = -1;
    int n = 0;

    std::vector<std::pair<Action, std::unique_ptr<Node>>> children;
};

} // namespace
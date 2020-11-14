#include <vector>
#include <torch/torch.h>

#pragma once

namespace az {

/**
* \brief Options for neural network. Defaults are for chess.
* \param planes Number of input planes or depth in other words.
* \param height Input plane height.
* \param width Input plane width.
* \param filters Number of filters in every block (input, residual and heads).
* \param num_res_blocks Number of residual blocks stacked in net.
* \param policy_size Policy head output vector size.
*/
struct NetworkOptions {
    int planes = 119;
    int height = 8;
    int width = 8;
    int filters = 256;
    int num_res_blocks = 20;
    int policy_size = 4672;
};

// Input block for AlphaZero.
struct InputConvImpl : torch::nn::Module {

    InputConvImpl(int in_channels, int out_channels = 256) :
        conv(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)),
        batch_norm(out_channels)
    {
        register_module("input_conv", conv);
        register_module("input_batch_norm", batch_norm);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        return torch::relu(batch_norm(conv(x)));
    }

private:
    torch::nn::Conv2d conv = nullptr;
    torch::nn::BatchNorm2d batch_norm = nullptr;
};
TORCH_MODULE(InputConv);

// Residual block with skip connection.
struct ResBlockImpl : torch::nn::Module {

    ResBlockImpl(int filters = 256) :
        conv_1(torch::nn::Conv2dOptions(filters, filters, 3).stride(1).padding(1)),
        conv_2(torch::nn::Conv2dOptions(filters, filters, 3).stride(1).padding(1)),
        batch_norm_1(filters),
        batch_norm_2(filters)
    {
        register_module("conv_1", conv_1);
        register_module("conv_2", conv_2);
        register_module("batch_norm_1", batch_norm_1);
        register_module("batch_norm_2", batch_norm_2);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        auto identity = x.clone();

        x = batch_norm_1(conv_1(x));
        x = torch::relu(x);
        x = batch_norm_2(conv_2(x));

        x += identity;
        x = torch::relu(x);

        return x;
    }

private:
    torch::nn::Conv2d conv_1 = nullptr;
    torch::nn::Conv2d conv_2 = nullptr;
    torch::nn::BatchNorm2d batch_norm_1 = nullptr;
    torch::nn::BatchNorm2d batch_norm_2 = nullptr;
};
TORCH_MODULE(ResBlock);

// Policy head for AlphaZero.
struct PolicyHeadImpl : torch::nn::Module {

    PolicyHeadImpl(int in_channels, int height, int width, int output_size) :
        conv(torch::nn::Conv2dOptions(in_channels, 2, 1).stride(1)/*.padding(1)*/),
        batch_norm(2),
        fc(2*height*width, output_size)
    {
        register_module("p_conv", conv);
        register_module("p_batch_norm", batch_norm);
        register_module("p_fc", fc);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(batch_norm(conv(x)));
        x = x.view({x.size(0), -1});
        x = fc(x);

        return x;
    }

private:
    torch::nn::Conv2d conv = nullptr;
    torch::nn::BatchNorm2d batch_norm = nullptr;
    torch::nn::Linear fc = nullptr;
};
TORCH_MODULE(PolicyHead);

// Value head for AlphaZero.
struct ValueHeadImpl : torch::nn::Module {

    ValueHeadImpl(int in_channels, int height, int width) :
        conv(torch::nn::Conv2dOptions(in_channels, 1, 1).stride(1)/*.padding(1)*/),
        batch_norm(1),
        fc_1(height*width, in_channels),
        fc_2(in_channels, 1)
    {
        register_module("v_conv", conv);
        register_module("v_batch_norm", batch_norm);
        register_module("v_fc_1", fc_1);
        register_module("v_fc_2", fc_2);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(batch_norm(conv(x)));
        x = x.view({x.size(0), -1});
        x = torch::relu(fc_1(x));
        x = torch::tanh(fc_2(x)); 

        return x;
    }
private:
    torch::nn::Conv2d conv = nullptr;
    torch::nn::BatchNorm2d batch_norm = nullptr;
    torch::nn::Linear fc_1 = nullptr;
    torch::nn::Linear fc_2 = nullptr;
};
TORCH_MODULE(ValueHead);

// AlphaZero
struct AlphaZeroNetworkImpl : torch::nn::Module {

    AlphaZeroNetworkImpl(const NetworkOptions &op) :
        input_block(op.planes, op.filters),
        p_head(op.filters, op.height, op.width, op.policy_size),
        v_head(op.filters, op.height, op.width)
    {
        register_module("input_block", input_block);
        register_module("p_head", p_head);
        register_module("v_head", v_head);

        for (int i = 0; i < op.num_res_blocks; ++i) {
            res_blocks.emplace_back(op.filters);
            register_module("res_block"+std::to_string(i+1), res_blocks.back());
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = input_block(x);
        for (auto &block : res_blocks)
            x = block(x);

        auto policy = p_head(x);
        auto value = v_head(x);

        return {value, policy};
    }

private:
    InputConv input_block;
    std::vector<ResBlock> res_blocks;
    PolicyHead p_head;
    ValueHead v_head;
};
TORCH_MODULE(AlphaZeroNetwork);

} // namespace
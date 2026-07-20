#include "dino.h"
#include "puffernet.h"

void forward_dino_policy(PufferNet* net, float* observations, float* actions) {
    linear(net->encoder, observations);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    argmax_multidiscrete(net->multidiscrete, net->decoder->output, actions);
}

void reset_dino_policy(PufferNet* net) {
    size_t state_size = net->mingru->num_layers *
        net->mingru->batch_size * net->mingru->hidden_size;
    memset(net->mingru->state, 0, state_size * sizeof(float));
}

void demo() {
    Dino env = {
        .num_agents = 1,
        .width = 800,
        .height = 250,
        .rng = 12345,
        .dinosaur = {
            .x = 80,
            .width = 40,
            .height = 48,
        },
        .obstacle = {
            .width = 24,
            .height = 40,
        },
    };
    float observations[5] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};

    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;

    // Depends on the workflow
    // Multiple locations here - need a weights.bin file
    // Currently training on a vast.ai instance to get this
    Weights* weights = load_weights("resources/dino/dino_weights.bin");
    int logit_sizes[1] = {2};
    PufferNet* net = make_puffernet(weights, 1, 5, 128, 1, logit_sizes, 1);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = IsKeyPressed(KEY_SPACE) ? JUMP : NOOP;
        } else {
            forward_dino_policy(net, env.observations, env.actions);
        }
        c_step(&env);
        if (env.terminals[0]) {
            reset_dino_policy(net);
        }
        c_render(&env);
    }

    free_puffernet(net);
    free(weights);
    c_close(&env);
}

int main() {
    demo();
    return 0;
}

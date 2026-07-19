#include "dino.h"
#include "puffernet.h"

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
            forward_puffernet(net, env.observations, env.actions);
        }
        c_step(&env);
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

#include "dino.h"

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

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        env.actions[0] = IsKeyPressed(KEY_SPACE) ? JUMP : NOOP;
        c_step(&env);
        c_render(&env);
    }
    c_close(&env);
}

int main() {
    demo();
    return 0;
}

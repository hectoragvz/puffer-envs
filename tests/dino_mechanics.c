#include <assert.h>
#include <stdio.h>
#include "dino.h"

static Dino make_env(float* observations, float* actions,
        float* rewards, float* terminals, unsigned int seed) {
    Dino env = {
        .num_agents = 1,
        .width = 800,
        .height = 250,
        .rng = seed,
        .dinosaur = {.x = 80, .width = 40, .height = 48},
        .obstacle = {.width = 24, .height = 40},
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
    };
    return env;
}

static void test_jump_cost(void) {
    float observations[5] = {0};
    float actions[1] = {JUMP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);

    c_reset(&env);
    c_step(&env);

    assert(env.dinosaur.y > 0);
    assert(env.rewards[0] == -JUMP_PENALTY);
}

static void test_scripted_controller(void) {
    for (unsigned int seed = 1; seed <= 100; seed++) {
        float observations[5] = {0};
        float actions[1] = {0};
        float rewards[1] = {0};
        float terminals[1] = {0};
        Dino env = make_env(observations, actions, rewards, terminals, seed);
        int passed = 0;

        c_reset(&env);
        for (int step = 0; step < 250; step++) {
            float distance = env.obstacle.x -
                (env.dinosaur.x + env.dinosaur.width);
            env.actions[0] = env.dinosaur.y == 0 && distance <= 48 ? JUMP : NOOP;
            c_step(&env);

            assert(!env.terminals[0]);
            if (env.rewards[0] > 0) {
                passed = 1;
                break;
            }
        }
        assert(passed);
    }
}

int main(void) {
    test_jump_cost();
    test_scripted_controller();
    puts("dino mechanics tests passed");
    return 0;
}

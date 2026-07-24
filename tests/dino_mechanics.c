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
        .auto_reset = 1,
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

static void test_logical_obstacle_respawns_at_original_boundary(void) {
    float observations[5] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);

    c_reset(&env);
    env.dinosaur.y = 50;
    env.obstacle.x = env.dinosaur.x + env.dinosaur.width -
        env.obstacle.width + OBSTACLE_SPEED - 1;
    c_step(&env);

    assert(env.rewards[0] == 1.0f);
    assert(env.obstacles_passed == 1);
    assert(env.cleared_obstacle_active);
    assert(env.cleared_obstacle.x + env.cleared_obstacle.width <
        env.dinosaur.x + env.dinosaur.width);
    assert(env.obstacle.x >= env.width);
    assert(observations[2] ==
        (env.obstacle.x - (env.dinosaur.x + env.dinosaur.width)) /
        (env.width * 1.5f));
}

static void test_cleared_obstacle_is_cosmetic(void) {
    float observations[5] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);

    c_reset(&env);
    env.obstacle.x = 400;
    env.cleared_obstacle = (Obstacle) {
        .x = env.dinosaur.x,
        .width = env.obstacle.width,
        .height = env.obstacle.height,
    };
    env.cleared_obstacle_active = 1;

    c_step(&env);
    assert(env.rewards[0] == 0);
    assert(!env.terminals[0]);
    assert(env.obstacles_passed == 0);
    assert(env.obstacle.x == 400 - OBSTACLE_SPEED);
    assert(env.cleared_obstacle.x == env.dinosaur.x - OBSTACLE_SPEED);
    assert(observations[2] ==
        (env.obstacle.x - (env.dinosaur.x + env.dinosaur.width)) /
        (env.width * 1.5f));

    env.cleared_obstacle.x =
        -env.cleared_obstacle.width + OBSTACLE_SPEED - 1;
    c_step(&env);
    assert(!env.cleared_obstacle_active);

    env.cleared_obstacle_active = 1;
    c_reset(&env);
    assert(!env.cleared_obstacle_active);
}

static void test_training_collision_still_resets(void) {
    float observations[5] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);

    c_reset(&env);
    env.obstacle.x = env.dinosaur.x;
    c_step(&env);

    assert(env.terminals[0]);
    assert(env.tick == 0);
    assert(env.dinosaur.y == 0);
    assert(env.obstacle.x >= env.width);
}

static void test_viewer_collision_does_not_reset(void) {
    float observations[5] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);

    c_reset(&env);
    env.auto_reset = 0;
    env.obstacle.x = env.dinosaur.x;
    c_step(&env);

    assert(env.terminals[0]);
    assert(env.tick == 1);
    assert(env.obstacle.x == env.dinosaur.x - OBSTACLE_SPEED);
}

int main(void) {
    test_jump_cost();
    test_scripted_controller();
    test_logical_obstacle_respawns_at_original_boundary();
    test_cleared_obstacle_is_cosmetic();
    test_training_collision_still_resets();
    test_viewer_collision_does_not_reset();
    puts("dino mechanics tests passed");
    return 0;
}

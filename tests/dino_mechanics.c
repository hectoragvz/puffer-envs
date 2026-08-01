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

static void test_owned_prng(void) {
    Dino env = {.rng = 0};
    const uint32_t expected[] = {
        1013904223u, 1196435762u, 3519870697u, 2868466484u, 1649599747u,
    };
    for (size_t i = 0; i < sizeof(expected) / sizeof(*expected); i++) {
        assert(dino_random(&env) == expected[i]);
    }
}

static void test_recipe_validation(void) {
    DinoChallengeRecipe recipe = {
        .schema_version = DINO_CHALLENGE_SCHEMA_VERSION,
        .environment_version = DINO_ENVIRONMENT_VERSION,
        .policy_version = DINO_POLICY_VERSION,
        .seed = 1,
    };
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_OK);

    recipe.schema_version++;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_UNSUPPORTED_SCHEMA);
    recipe.schema_version = DINO_CHALLENGE_SCHEMA_VERSION;
    recipe.environment_version++;
    assert(dino_validate_recipe(&recipe) ==
        DINO_REPLAY_UNSUPPORTED_ENVIRONMENT);
    recipe.environment_version = DINO_ENVIRONMENT_VERSION;
    recipe.policy_version++;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_UNSUPPORTED_POLICY);
    recipe.policy_version = DINO_POLICY_VERSION;
    recipe.event_count = 1;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_UNSUPPORTED_EVENTS);
}

static void test_seeded_replay_reset(void) {
    float observations[5] = {0};
    float actions[1] = {JUMP};
    float rewards[1] = {1};
    float terminals[1] = {1};
    Dino env = make_env(observations, actions, rewards, terminals, 77);

    c_reset(&env);
    c_reset(&env);
    dino_seeded_replay_reset(&env, 1);
    float obstacle_x = env.obstacle.x;
    uint32_t rng = env.rng;
    assert(actions[0] == 0 && rewards[0] == 0 && terminals[0] == 0);

    env.rng = 999;
    env.cleared_obstacle_active = 1;
    c_reset(&env);
    dino_seeded_replay_reset(&env, 1);
    assert(env.obstacle.x == obstacle_x);
    assert(env.rng == rng);
    assert(!env.cleared_obstacle_active);
}

static void test_different_seeds_change_obstacles(void) {
    float observations_a[5] = {0};
    float observations_b[5] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino a = make_env(observations_a, actions, rewards, terminals, 0);
    Dino b = make_env(observations_b, actions, rewards, terminals, 1);

    c_reset(&a);
    c_reset(&b);
    assert(a.obstacle.x != b.obstacle.x);
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

static void test_forced_collision_metadata(void) {
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
    DinoEpisodeEnding ending = env.terminals[0] ?
        DINO_EPISODE_COLLISION : DINO_EPISODE_TRUNCATED;
    uint32_t collision_tick = (uint32_t)env.tick - 1;
    assert(ending == DINO_EPISODE_COLLISION);
    assert(collision_tick == 0);
}

int main(void) {
    test_owned_prng();
    test_recipe_validation();
    test_seeded_replay_reset();
    test_different_seeds_change_obstacles();
    test_jump_cost();
    test_scripted_controller();
    test_logical_obstacle_respawns_at_original_boundary();
    test_cleared_obstacle_is_cosmetic();
    test_training_collision_still_resets();
    test_forced_collision_metadata();
    puts("dino mechanics tests passed");
    return 0;
}

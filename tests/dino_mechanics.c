#include <assert.h>
#include <stdio.h>
#include "dino.h"

static Dino make_env(float* observations, float* actions, float* rewards, float* terminals, unsigned int seed) {
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
    DinoChallengeEvent events[2] = {
        {.tick = 0, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 2.0f},
        {.tick = 10, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 1.0f},
    };
    DinoChallengeRecipe recipe = {
        .schema_version = DINO_CHALLENGE_SCHEMA_VERSION,
        .environment_version = DINO_ENVIRONMENT_VERSION,
        .policy_version = DINO_POLICY_VERSION,
        .seed = 1,
        .events = events,
    };
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_OK);

    recipe.schema_version++;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_UNSUPPORTED_SCHEMA);
    recipe.schema_version = DINO_CHALLENGE_SCHEMA_VERSION;
    recipe.environment_version++;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_UNSUPPORTED_ENVIRONMENT);
    recipe.environment_version = DINO_ENVIRONMENT_VERSION;
    recipe.policy_version++;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_UNSUPPORTED_POLICY);
    recipe.policy_version = DINO_POLICY_VERSION;

    recipe.event_count = 1;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_OK);
    events[0].value = 3.0f;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_OK);
    events[0].value = 2.0f;

    recipe.events = NULL;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_INVALID_EVENT_LIST);
    recipe.events = events;
    recipe.event_count = DINO_CHALLENGE_EVENT_LIMIT + 1;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_TOO_MANY_EVENTS);
    recipe.event_count = 1;
    events[0].type = DINO_CHALLENGE_EVENT_OBSTACLE;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_UNSUPPORTED_EVENT_TYPE);
    events[0].type = DINO_CHALLENGE_EVENT_SPEED;
    events[0].tick = DINO_REPLAY_LIMIT;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_INVALID_EVENT_TICK);
    recipe.event_count = 2;
    events[0].tick = 10;
    events[1].tick = 9;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_INVALID_EVENT_ORDER);
    events[1].tick = 10;
    assert(dino_validate_recipe(&recipe) == DINO_REPLAY_OK);
}

static void test_seeded_replay_reset(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
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

static void test_meteor_requires_duck(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);

    c_reset(&env);
    env.auto_reset = 0;
    env.obstacle = (Obstacle) {
        .x = env.dinosaur.x + OBSTACLE_SPEED,
        .width = METEOR_WIDTH,
        .height = METEOR_HEIGHT,
        .bottom = METEOR_BOTTOM,
    };
    actions[0] = NOOP;
    c_step(&env);
    assert(env.terminals[0]);

    c_reset(&env);
    env.obstacle = (Obstacle) {
        .x = env.dinosaur.x + OBSTACLE_SPEED,
        .width = METEOR_WIDTH,
        .height = METEOR_HEIGHT,
        .bottom = METEOR_BOTTOM,
    };
    actions[0] = DUCK;
    c_step(&env);
    assert(!env.terminals[0]);

    c_reset(&env);
    env.obstacle = (Obstacle) {
        .x = env.dinosaur.x + OBSTACLE_SPEED,
        .width = 24,
        .height = 40,
        .bottom = 0,
    };
    actions[0] = DUCK;
    c_step(&env);
    assert(env.terminals[0]);
}

static void test_different_seeds_change_obstacles(void) {
    float observations_a[DINO_OBSERVATION_COUNT] = {0};
    float observations_b[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino a = make_env(observations_a, actions, rewards, terminals, 0);
    Dino b = make_env(observations_b, actions, rewards, terminals, 1);
    c_reset(&a);
    c_reset(&b);
    assert(a.obstacle.x != b.obstacle.x);
}

static void test_seeded_obstacle_types_are_repeatable(void) {
    float observations_a[DINO_OBSERVATION_COUNT] = {0};
    float observations_b[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino a = make_env(observations_a, actions, rewards, terminals, 1);
    Dino b = make_env(observations_b, actions, rewards, terminals, 1);
    int meteors = 0;
    int cacti = 0;

    for (int i = 0; i < 32; i++) {
        spawn_obstacle(&a);
        spawn_obstacle(&b);

        assert(a.obstacle.x == b.obstacle.x);
        assert(a.obstacle.width == b.obstacle.width);
        assert(a.obstacle.height == b.obstacle.height);
        assert(a.obstacle.bottom == b.obstacle.bottom);

        if (a.obstacle.bottom == METEOR_BOTTOM) {
            assert(a.obstacle.width == METEOR_WIDTH);
            assert(a.obstacle.height == METEOR_HEIGHT);
            meteors++;
        } else {
            assert(a.obstacle.bottom == 0);
            assert(a.obstacle.width == 24);
            assert(a.obstacle.height == 40);
            cacti++;
        }
    }
    assert(meteors > 0);
    assert(cacti > 0);
}

static void test_speed_change_and_observation(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    c_reset(&env);
    assert(env.speed_multiplier == DINO_DEFAULT_SPEED);
    assert(observations[5] == 0.0f);
    assert(dino_change_speed(&env, DINO_MAX_SPEED) == ENV_SPEED_ACCEPTED);
    assert(env.speed_multiplier == DINO_MAX_SPEED);
    assert(observations[5] == 1.0f);
    assert(dino_change_speed(&env, 1.5f) == ENV_SPEED_INVALID);
    assert(env.speed_multiplier == DINO_MAX_SPEED);
    assert(observations[5] == 1.0f);
    assert(dino_change_speed(&env, DINO_MIN_SPEED) == ENV_SPEED_ACCEPTED);
    assert(env.speed_multiplier == DINO_MIN_SPEED);
    assert(observations[5] == 0.0f);
}

static void test_speed_changes_world_movement_only(void) {
    float observations_a[DINO_OBSERVATION_COUNT] = {0};
    float observations_b[DINO_OBSERVATION_COUNT] = {0};
    float actions_a[1] = {NOOP};
    float actions_b[1] = {NOOP};
    float rewards_a[1] = {0};
    float rewards_b[1] = {0};
    float terminals_a[1] = {0};
    float terminals_b[1] = {0};
    Dino one_x = make_env(observations_a, actions_a, rewards_a, terminals_a, 1);
    Dino two_x = make_env(observations_b, actions_b, rewards_b, terminals_b, 1);
    c_reset(&one_x);
    c_reset(&two_x);
    one_x.obstacle.x = 400;
    two_x.obstacle.x = 400;
    one_x.cleared_obstacle = (Obstacle) {.x = 200, .width = 24, .height = 40};
    two_x.cleared_obstacle = one_x.cleared_obstacle;
    one_x.cleared_obstacle_active = 1;
    two_x.cleared_obstacle_active = 1;
    dino_change_speed(&two_x, DINO_MAX_SPEED);
    c_step(&one_x);
    c_step(&two_x);
    assert(one_x.obstacle.x == 400 - OBSTACLE_SPEED);
    assert(two_x.obstacle.x == 400 - 2 * OBSTACLE_SPEED);
    assert(one_x.cleared_obstacle.x == 200 - OBSTACLE_SPEED);
    assert(two_x.cleared_obstacle.x == 200 - 2 * OBSTACLE_SPEED);
    assert(one_x.dinosaur.y == two_x.dinosaur.y);
    assert(one_x.dinosaur.y_velocity == two_x.dinosaur.y_velocity);
    assert(one_x.tick == 1 && two_x.tick == 1);
}

static void test_reset_restores_default_speed(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    c_reset(&env);
    dino_change_speed(&env, DINO_MAX_SPEED);
    c_reset(&env);
    assert(env.speed_multiplier == DINO_DEFAULT_SPEED);
    assert(observations[5] == 0.0f);
}

static void test_speed_event_timing_order_and_rejection(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    DinoSpeedChangeResult results[3] = {0};
    DinoChallengeEvent events[3] = {
        {.tick = 0, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 2.0f},
        {.tick = 0, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 1.0f},
        {.tick = 1, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 3.0f},
    };
    c_reset(&env);
    uint32_t event_index = dino_apply_speed_events_at_tick(&env, events, 3, 0, 0, results);
    assert(event_index == 2);
    assert(results[0] == ENV_SPEED_ACCEPTED);
    assert(results[1] == ENV_SPEED_ACCEPTED);
    assert(env.speed_multiplier == DINO_MIN_SPEED);
    assert(observations[5] == 0.0f);
    event_index = dino_apply_speed_events_at_tick(&env, events, 3, event_index, 1, results);
    assert(event_index == 3);
    assert(results[2] == ENV_SPEED_INVALID);
    assert(env.speed_multiplier == DINO_MIN_SPEED);
}

static float pass_obstacle(Dino* env) {
    env->dinosaur.y = 50;
    env->dinosaur.y_velocity = 0;
    env->obstacle.x = env->dinosaur.x + env->dinosaur.width - env->obstacle.width + OBSTACLE_SPEED * env->speed_multiplier - 1;
    float obstacle_x = env->obstacle.x;
    c_step(env);
    return obstacle_x;
}

static void test_training_speedup_after_passes(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    env.training_speedup_after_passes = DINO_CURRICULUM_SPEEDUP_AFTER_PASSES;
    c_reset(&env);
    assert(env.speed_multiplier == DINO_MIN_SPEED);
    assert(observations[5] == 0.0f);
    assert(!env.episode_reached_2x);

    for (int pass = 1; pass <= 8; pass++) {
        float speed_before_pass = env.speed_multiplier;
        float obstacle_x = pass_obstacle(&env);
        assert(env.rewards[0] == 1.0f);
        assert(env.obstacles_passed == pass);
        if (pass <= 6) {
            assert(env.speed_multiplier == DINO_MIN_SPEED);
            assert(observations[5] == 0.0f);
            assert(!env.episode_reached_2x);
        } else {
            assert(env.speed_multiplier == DINO_MAX_SPEED);
            assert(observations[5] == 1.0f);
            assert(env.episode_reached_2x);
        }
        if (pass == 7) {
            assert(speed_before_pass == DINO_MIN_SPEED);
            assert(env.cleared_obstacle.x == obstacle_x - OBSTACLE_SPEED);
        }
        if (pass == 8) {
            assert(speed_before_pass == DINO_MAX_SPEED);
            assert(env.cleared_obstacle.x == obstacle_x - 2 * OBSTACLE_SPEED);
        }
    }
    env.dinosaur.y = 0;
    env.obstacle.x = env.dinosaur.x;
    c_step(&env);
    assert(env.terminals[0]);
    assert(env.log.reached_2x == 1.0f);
    assert(env.log.n == 1.0f);
    assert(env.speed_multiplier == DINO_MIN_SPEED);
    assert(observations[5] == 0.0f);
    assert(!env.episode_reached_2x);
}

static void test_training_speedup_zero_stays_at_1x(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    env.training_speedup_after_passes = 0;
    c_reset(&env);
    for (int pass = 1; pass <= 8; pass++) {
        pass_obstacle(&env);
        assert(env.obstacles_passed == pass);
        assert(env.speed_multiplier == DINO_MIN_SPEED);
        assert(observations[5] == 0.0f);
    }
}

static void test_jump_cost(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {JUMP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    c_reset(&env);
    c_step(&env);
    assert(env.dinosaur.y > 0);
    assert(env.rewards[0] == -JUMP_PENALTY);
}

static void test_duck_is_held_and_ground_only(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    c_reset(&env);
    env.obstacle.x = env.width;
    actions[0] = DUCK;
    c_step(&env);
    assert(env.dinosaur.ducking);
    assert(env.dinosaur.y == 0);
    actions[0] = NOOP;
    c_step(&env);
    assert(!env.dinosaur.ducking);
    actions[0] = JUMP;
    c_step(&env);
    assert(!env.dinosaur.ducking);
    assert(env.dinosaur.y > 0);
    actions[0] = DUCK;
    c_step(&env);
    assert(!env.dinosaur.ducking);
}

static void assert_scripted_controller_passes(
        const char* name, const DinoChallengeEvent* events,
        uint32_t event_count) {
    for (unsigned int seed = 1; seed <= 100; seed++) {
        float observations[DINO_OBSERVATION_COUNT] = {0};
        float actions[1] = {0};
        float rewards[1] = {0};
        float terminals[1] = {0};
        Dino env = make_env(observations, actions, rewards, terminals, seed);
        int passed = 0;
        uint32_t event_index = 0;
        c_reset(&env);
        for (int step = 0; step < 250; step++) {
            event_index = dino_apply_speed_events_at_tick(&env, events, event_count, event_index, (uint32_t)env.tick, NULL);
            float distance = env.obstacle.x - (env.dinosaur.x + env.dinosaur.width);
            float reaction_distance = 8 * OBSTACLE_SPEED * env.speed_multiplier;
            if (env.dinosaur.y == 0 && distance <= reaction_distance) {
                env.actions[0] = env.obstacle.bottom == 0 ? JUMP : DUCK;
            } else {
                env.actions[0] = NOOP;
            }
            c_step(&env);
            if (env.terminals[0]) {
                fprintf(stderr, "scripted suite=%s seed=%u step=%d\n", name, seed, step);
            }
            assert(!env.terminals[0]);
            if (env.rewards[0] > 0) {
                passed = 1;
                break;
            }
        }
        assert(passed);
    }
}

static void test_scripted_controller_speed_suites(void) {
    const DinoChallengeEvent fixed_2x[] = {
        {.tick = 0, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 2.0f},
    };
    const DinoChallengeEvent up[] = {
        {.tick = 40, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 2.0f},
    };
    const DinoChallengeEvent down[] = {
        {.tick = 0, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 2.0f},
        {.tick = 20, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 1.0f},
    };
    const DinoChallengeEvent multiple[] = {
        {.tick = 20, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 2.0f},
        {.tick = 25, .type = DINO_CHALLENGE_EVENT_SPEED, .value = 1.0f},
    };
    assert_scripted_controller_passes("1x", NULL, 0);
    assert_scripted_controller_passes("2x", fixed_2x, 1);
    assert_scripted_controller_passes("up", up, 1);
    assert_scripted_controller_passes("down", down, 2);
    assert_scripted_controller_passes("multi", multiple, 2);
}

static void test_scripted_controller_crosses_curriculum_transition(void) {
    for (unsigned int seed = 1; seed <= 100; seed++) {
        float observations[DINO_OBSERVATION_COUNT] = {0};
        float actions[1] = {0};
        float rewards[1] = {0};
        float terminals[1] = {0};
        Dino env = make_env(observations, actions, rewards, terminals, seed);
        env.training_speedup_after_passes = DINO_CURRICULUM_SPEEDUP_AFTER_PASSES;
        c_reset(&env);
        for (int step = 0; step < 2000 && env.obstacles_passed < 8; step++) {
            float distance = env.obstacle.x - (env.dinosaur.x + env.dinosaur.width);
            float reaction_distance = 8 * OBSTACLE_SPEED * env.speed_multiplier;
            if (env.dinosaur.y == 0 && distance <= reaction_distance) {
                env.actions[0] = env.obstacle.bottom == 0 ? JUMP : DUCK;
            } else {
                env.actions[0] = NOOP;
            }
            c_step(&env);
            if (env.terminals[0]) {
                fprintf(stderr, "scripted curriculum seed=%u step=%d\n", seed, step);
            }
            assert(!env.terminals[0]);
        }
        assert(env.obstacles_passed == 8);
        assert(env.speed_multiplier == DINO_MAX_SPEED);
        assert(observations[5] == 1.0f);
    }
}

static void test_logical_obstacle_respawns_at_original_boundary(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    c_reset(&env);
    env.dinosaur.y = 50;
    env.obstacle.x = env.dinosaur.x + env.dinosaur.width - env.obstacle.width + OBSTACLE_SPEED - 1;
    c_step(&env);
    assert(env.rewards[0] == 1.0f);
    assert(env.obstacles_passed == 1);
    assert(env.cleared_obstacle_active);
    assert(env.cleared_obstacle.x + env.cleared_obstacle.width < env.dinosaur.x + env.dinosaur.width);
    assert(env.obstacle.x >= env.width);
    assert(observations[2] == (env.obstacle.x - (env.dinosaur.x + env.dinosaur.width)) / (env.width * 1.5f));
}

static void test_cleared_obstacle_is_cosmetic(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
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
    assert(observations[2] == (env.obstacle.x - (env.dinosaur.x + env.dinosaur.width)) / (env.width * 1.5f));
    env.cleared_obstacle.x =- env.cleared_obstacle.width + OBSTACLE_SPEED - 1;
    c_step(&env);
    assert(!env.cleared_obstacle_active);
    env.cleared_obstacle_active = 1;
    c_reset(&env);
    assert(!env.cleared_obstacle_active);
}

static void test_training_collision_still_resets(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    c_reset(&env);
    dino_change_speed(&env, DINO_MAX_SPEED);
    env.obstacle.x = env.dinosaur.x;
    c_step(&env);
    assert(env.terminals[0]);
    assert(env.tick == 0);
    assert(env.dinosaur.y == 0);
    assert(env.obstacle.x >= env.width);
    assert(env.speed_multiplier == DINO_DEFAULT_SPEED);
    assert(observations[5] == 0.0f);
}

static void test_forced_collision_metadata(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
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
    DinoEpisodeEnding ending = env.terminals[0] ? DINO_EPISODE_COLLISION : DINO_EPISODE_TRUNCATED;
    uint32_t collision_tick = (uint32_t)env.tick - 1;
    assert(ending == DINO_EPISODE_COLLISION);
    assert(collision_tick == 0);
}

static void test_observation_includes_obstacle_bottom(void) {
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {NOOP};
    float rewards[1] = {0};
    float terminals[1] = {0};
    Dino env = make_env(observations, actions, rewards, terminals, 1);
    c_reset(&env);
    env.obstacle.bottom = METEOR_BOTTOM;
    update_observations(&env);
    assert(observations[6] == METEOR_BOTTOM / env.height);
}

int main(void) {
    test_owned_prng();
    test_recipe_validation();
    test_seeded_replay_reset();
    test_different_seeds_change_obstacles();
    test_seeded_obstacle_types_are_repeatable();
    test_speed_change_and_observation();
    test_speed_changes_world_movement_only();
    test_reset_restores_default_speed();
    test_speed_event_timing_order_and_rejection();
    test_training_speedup_after_passes();
    test_training_speedup_zero_stays_at_1x();
    test_jump_cost();
    test_duck_is_held_and_ground_only();
    test_meteor_requires_duck();
    test_scripted_controller_speed_suites();
    test_scripted_controller_crosses_curriculum_transition();
    test_logical_obstacle_respawns_at_original_boundary();
    test_cleared_obstacle_is_cosmetic();
    test_training_collision_still_resets();
    test_forced_collision_metadata();
    test_observation_includes_obstacle_bottom();
    puts("dino mechanics tests passed");
    return 0;
}

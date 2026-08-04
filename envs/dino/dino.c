#include "dino.h"
#include "puffernet.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#define EVALUATION_MAX_STEPS DINO_REPLAY_LIMIT

typedef struct {
    uint32_t tick;
    float observations[DINO_OBSERVATION_COUNT];
    uint32_t action;
    float reward;
    uint32_t terminal;
    Dinosaur dinosaur;
    Obstacle obstacle;
    Obstacle cleared_obstacle;
    uint32_t cleared_obstacle_active;
    uint32_t score;
    float episode_return;
    uint32_t rng;
    uint32_t result_tick;
    float speed_multiplier;
} DinoTrajectoryStep;

typedef struct {
    DinoReplayStatus status;
    DinoEpisodeEnding ending;
    uint32_t step_count;
    uint32_t end_tick;
    uint32_t collision_tick;
    uint32_t jump_count;
    uint32_t jump_ticks[DINO_REPLAY_LIMIT];
    uint32_t event_result_count;
    DinoSpeedChangeResult event_results[DINO_CHALLENGE_EVENT_LIMIT];
    uint64_t digest;
    DinoTrajectoryStep steps[DINO_REPLAY_LIMIT];
} DinoReplay;

typedef struct {
    uint32_t seed;
    DinoEpisodeEnding ending;
    uint32_t score;
    float episode_return;
    uint32_t end_tick;
    uint32_t collision_tick;
    uint32_t jump_count;
    const uint32_t* jump_ticks;
    uint64_t digest;
} DinoGoldenReplay;

typedef struct {
    Dino* env;
    PufferNet* net;
} DemoArgs;

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

static void digest_u32(uint64_t* digest, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        *digest ^= (value >> shift) & 0xffu;
        *digest *= UINT64_C(1099511628211);
    }
}

static void digest_float(uint64_t* digest, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    digest_u32(digest, bits);
}

static void digest_obstacle(uint64_t* digest, Obstacle obstacle) {
    digest_float(digest, obstacle.x);
    digest_float(digest, obstacle.height);
    digest_float(digest, obstacle.width);
}

static void digest_replay_event(uint64_t* digest, uint32_t event_index,
        const DinoChallengeEvent* event, DinoSpeedChangeResult result) {
    digest_u32(digest, event_index);
    digest_u32(digest, event->tick);
    digest_u32(digest, event->type);
    digest_float(digest, event->value);
    digest_u32(digest, (uint32_t)result);
}

static void digest_trajectory_step(uint64_t* digest,
        const DinoTrajectoryStep* step) {
    digest_u32(digest, step->tick);
    for (uint32_t i = 0; i < DINO_OBSERVATION_COUNT; i++) {
        digest_float(digest, step->observations[i]);
    }
    digest_u32(digest, step->action);
    digest_float(digest, step->reward);
    digest_u32(digest, step->terminal);
    digest_float(digest, step->dinosaur.y);
    digest_float(digest, step->dinosaur.y_velocity);
    digest_float(digest, step->dinosaur.x);
    digest_float(digest, step->dinosaur.height);
    digest_float(digest, step->dinosaur.width);
    digest_obstacle(digest, step->obstacle);
    digest_obstacle(digest, step->cleared_obstacle);
    digest_u32(digest, step->cleared_obstacle_active);
    digest_u32(digest, step->score);
    digest_float(digest, step->episode_return);
    digest_u32(digest, step->rng);
    digest_u32(digest, step->result_tick);
    digest_float(digest, step->speed_multiplier);
}

static DinoReplayStatus replay_dino(Dino* env, PufferNet* net,
        const DinoChallengeRecipe* recipe, DinoReplay* replay) {
    memset(replay, 0, sizeof(*replay));
    replay->collision_tick = UINT32_MAX;
    replay->status = dino_validate_recipe(recipe);
    if (replay->status != DINO_REPLAY_OK) return replay->status;

    env->auto_reset = 0;
    env->randomize_speed = 0;
    dino_seeded_replay_reset(env, recipe->seed);
    reset_dino_policy(net);
    replay->digest = UINT64_C(14695981039346656037);
    uint32_t event_index = 0;

    for (uint32_t tick = 0; tick < DINO_REPLAY_LIMIT; tick++) {
        uint32_t first_event = event_index;
        event_index = dino_apply_speed_events_at_tick(
            env, recipe->events, recipe->event_count, event_index, tick,
            replay->event_results
        );
        for (uint32_t i = first_event; i < event_index; i++) {
            digest_replay_event(
                &replay->digest, i, &recipe->events[i],
                replay->event_results[i]
            );
        }
        replay->event_result_count = event_index;

        DinoTrajectoryStep* step = &replay->steps[tick];
        step->tick = tick;
        memcpy(step->observations, env->observations,
            sizeof(step->observations));
        forward_dino_policy(net, env->observations, env->actions);
        step->action = (uint32_t)env->actions[0];
        if (step->action == JUMP && env->dinosaur.y == 0) {
            replay->jump_ticks[replay->jump_count++] = tick;
        }

        c_step(env);
        step->reward = env->rewards[0];
        step->terminal = (uint32_t)env->terminals[0];
        step->dinosaur = env->dinosaur;
        step->obstacle = env->obstacle;
        step->cleared_obstacle = env->cleared_obstacle;
        step->cleared_obstacle_active =
            (uint32_t)env->cleared_obstacle_active;
        step->score = (uint32_t)env->obstacles_passed;
        step->episode_return = env->episode_return;
        step->speed_multiplier = env->speed_multiplier;
        step->rng = env->rng;
        step->result_tick = (uint32_t)env->tick;
        digest_trajectory_step(&replay->digest, step);
        replay->step_count++;

        if (env->terminals[0]) {
            replay->ending = DINO_EPISODE_COLLISION;
            replay->collision_tick = tick;
            replay->end_tick = tick;
            return DINO_REPLAY_OK;
        }
    }

    replay->ending = DINO_EPISODE_TRUNCATED;
    replay->end_tick = DINO_REPLAY_LIMIT - 1;
    return DINO_REPLAY_OK;
}

static int replay_steps_equal(const DinoTrajectoryStep* a,
        const DinoTrajectoryStep* b) {
    if (a->tick != b->tick || a->action != b->action ||
            a->reward != b->reward || a->terminal != b->terminal ||
            memcmp(a->observations, b->observations,
                sizeof(a->observations)) != 0 ||
            memcmp(&a->dinosaur, &b->dinosaur, sizeof(a->dinosaur)) != 0 ||
            memcmp(&a->obstacle, &b->obstacle, sizeof(a->obstacle)) != 0 ||
            memcmp(&a->cleared_obstacle, &b->cleared_obstacle,
                sizeof(a->cleared_obstacle)) != 0 ||
            a->cleared_obstacle_active != b->cleared_obstacle_active ||
            a->score != b->score ||
            a->episode_return != b->episode_return || a->rng != b->rng ||
            a->result_tick != b->result_tick ||
            a->speed_multiplier != b->speed_multiplier) {
        return 0;
    }
    return 1;
}

static int replays_equal(const DinoReplay* a, const DinoReplay* b) {
    if (a->status != b->status || a->ending != b->ending ||
            a->step_count != b->step_count || a->end_tick != b->end_tick ||
            a->collision_tick != b->collision_tick ||
            a->jump_count != b->jump_count || a->digest != b->digest ||
            a->event_result_count != b->event_result_count ||
            memcmp(a->event_results, b->event_results,
                a->event_result_count * sizeof(*a->event_results)) != 0 ||
            memcmp(a->jump_ticks, b->jump_ticks,
                a->jump_count * sizeof(*a->jump_ticks)) != 0) {
        return 0;
    }
    for (uint32_t i = 0; i < a->step_count; i++) {
        if (!replay_steps_equal(&a->steps[i], &b->steps[i])) return 0;
    }
    return 1;
}

void init_dino(Dino* env, float* observations, float* actions,
        float* rewards, float* terminals) {
    *env = (Dino) {
        .num_agents = 1,
        .width = 800,
        .height = 250,
        .rng = 12345,
        .auto_reset = 1,
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

    env->observations = observations;
    env->actions = actions;
    env->rewards = rewards;
    env->terminals = terminals;
}

void demo_frame(void* data) {
    DemoArgs* args = (DemoArgs*)data;
    Dino* env = args->env;

#ifndef __EMSCRIPTEN__
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
        env->actions[0] = IsKeyPressed(KEY_SPACE) ? JUMP : NOOP;
    } else
#endif
    {
        forward_dino_policy(args->net, env->observations, env->actions);
    }

    c_step(env);
    if (env->terminals[0]) {
        c_reset(env);
        env->terminals[0] = 0;
        reset_dino_policy(args->net);
    }
    c_render(env);
}

void demo() {
    Dino env;
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    init_dino(&env, observations, actions, rewards, terminals);
    env.auto_reset = 0;

    Weights* weights = load_weights("resources/dino/dino_weights.bin");
    int logit_sizes[1] = {2};
    PufferNet* net = make_puffernet(
        weights, 1, DINO_OBSERVATION_COUNT, 128, 1, logit_sizes, 1
    );

    c_reset(&env);
    c_render(&env);
    DemoArgs args = {.env = &env, .net = net};

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg(demo_frame, &args, 0, true);
#else
    while (!WindowShouldClose()) {
        demo_frame(&args);
    }

    free_puffernet(net);
    free(weights);
    c_close(&env);
#endif
}

void run_headless_evaluation(int episodes, int trace) {
    Dino env;
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    init_dino(&env, observations, actions, rewards, terminals);

    Weights* weights = load_weights("resources/dino/dino_weights.bin");
    int logit_sizes[1] = {2};
    PufferNet* net = make_puffernet(
        weights, 1, DINO_OBSERVATION_COUNT, 128, 1, logit_sizes, 1
    );
    int total_passes = 0;
    int total_jumps = 0;
    int total_steps = 0;
    int crashes_before_first_obstacle = 0;
    int truncated_episodes = 0;

    for (int episode = 0; episode < episodes; episode++) {
        int episode_passes = 0;
        int episode_steps = 0;
        int near_obstacle = 0;
        c_reset(&env);
        env.terminals[0] = 0;
        reset_dino_policy(net);

        while (!env.terminals[0] &&
                episode_steps < (int)EVALUATION_MAX_STEPS) {
            int tick = env.tick;
            float distance = env.obstacle.x -
                (env.dinosaur.x + env.dinosaur.width);
            forward_dino_policy(net, env.observations, env.actions);
            int action = (int)env.actions[0];
            int jumped = action == JUMP && env.dinosaur.y == 0;

            if (trace && distance <= 160 && !near_obstacle) {
                printf(
                    "near_obstacle step=%d distance=%.0f y=%.0f "
                    "velocity=%.0f action=%s\n",
                    tick, distance, env.dinosaur.y, env.dinosaur.y_velocity,
                    action == JUMP ? "JUMP" : "NOOP"
                );
            }
            near_obstacle = distance <= 160;
            if (trace && jumped) {
                printf("takeoff step=%d distance=%.0f\n", tick, distance);
            }

            c_step(&env);
            total_steps++;
            episode_steps++;
            total_jumps += jumped;
            if (env.rewards[0] > 0) {
                total_passes++;
                episode_passes++;
                if (trace) {
                    printf("passed_obstacle step=%d\n", tick);
                }
                near_obstacle = 0;
            }
            if (trace && env.terminals[0]) {
                printf("collision step=%d\n", tick);
            }
        }

        if (env.terminals[0] && episode_passes == 0) {
            crashes_before_first_obstacle++;
        }
        if (!env.terminals[0]) truncated_episodes++;
        reset_dino_policy(net);
    }

    printf(
        "episodes=%d mean_passes=%.2f mean_steps=%.1f "
        "mean_jumps=%.2f crashes_before_first_obstacle=%d "
        "truncated_episodes=%d\n",
        episodes,
        total_passes / (float)episodes,
        total_steps / (float)episodes,
        total_jumps / (float)episodes,
        crashes_before_first_obstacle,
        truncated_episodes
    );

    free_puffernet(net);
    free(weights);
}

static void print_replay(const DinoChallengeRecipe* recipe,
        const DinoReplay* replay) {
    const DinoTrajectoryStep* final =
        &replay->steps[replay->step_count - 1];
    printf(
        "schema_version=%u environment_version=%u policy_version=%u "
        "policy_sha256=%s seed=%u\n",
        recipe->schema_version, recipe->environment_version,
        recipe->policy_version, DINO_POLICY_WEIGHTS_SHA256, recipe->seed
    );
    printf(
        "outcome=%s score=%u return=%.9g end_tick=%u collision_tick=",
        replay->ending == DINO_EPISODE_COLLISION ? "COLLISION" : "TRUNCATED",
        final->score, final->episode_return, replay->end_tick
    );
    if (replay->collision_tick == UINT32_MAX) printf("none");
    else printf("%u", replay->collision_tick);
    printf("\njump_ticks=");
    for (uint32_t i = 0; i < replay->jump_count; i++) {
        if (i != 0) putchar(',');
        printf("%u", replay->jump_ticks[i]);
    }
    printf("\ntrajectory_digest=%016" PRIx64 "\n", replay->digest);
}

static int run_replay_cli(uint32_t seed) {
    Dino env;
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    init_dino(&env, observations, actions, rewards, terminals);

    Weights* weights = load_weights("resources/dino/dino_weights.bin");
    int logit_sizes[1] = {2};
    PufferNet* net = make_puffernet(
        weights, 1, DINO_OBSERVATION_COUNT, 128, 1, logit_sizes, 1
    );
    DinoReplay* replay = (DinoReplay*)calloc(1, sizeof(*replay));
    DinoChallengeRecipe recipe = {
        .schema_version = DINO_CHALLENGE_SCHEMA_VERSION,
        .environment_version = DINO_ENVIRONMENT_VERSION,
        .policy_version = DINO_POLICY_VERSION,
        .seed = seed,
    };
    DinoReplayStatus status = replay_dino(&env, net, &recipe, replay);
    if (status == DINO_REPLAY_OK) print_replay(&recipe, replay);

    free(replay);
    free_puffernet(net);
    free(weights);
    return status == DINO_REPLAY_OK ? 0 : 1;
}

static const uint32_t SEED_0_JUMPS[] = {
    100, 206, 315, 404, 496, 594, 695, 786, 890, 980, 1080, 1210, 1299,
    1421, 1544, 1667, 1769, 1872, 1992,
};
static const uint32_t SEED_1_JUMPS[] = {
    97, 228, 352, 442, 565, 687, 798, 910, 1005, 1106, 1216, 1350, 1440,
    1553, 1688, 1806, 1903,
};
static const uint32_t SEED_12345_JUMPS[] = {
    120, 212, 304, 433, 522, 643, 767, 904, 1029, 1150, 1254, 1360, 1450,
    1570, 1664, 1800, 1931,
};

static const DinoGoldenReplay OFFICIAL_REPLAYS[] = {
    {
        .seed = 0,
        .ending = DINO_EPISODE_TRUNCATED,
        .score = 18,
        .episode_return = 17.0499992f,
        .end_tick = 1999,
        .collision_tick = UINT32_MAX,
        .jump_count = sizeof(SEED_0_JUMPS) / sizeof(*SEED_0_JUMPS),
        .jump_ticks = SEED_0_JUMPS,
        .digest = UINT64_C(0xa9618683f40a23a1),
    },
    {
        .seed = 1,
        .ending = DINO_EPISODE_TRUNCATED,
        .score = 17,
        .episode_return = 16.1499977f,
        .end_tick = 1999,
        .collision_tick = UINT32_MAX,
        .jump_count = sizeof(SEED_1_JUMPS) / sizeof(*SEED_1_JUMPS),
        .jump_ticks = SEED_1_JUMPS,
        .digest = UINT64_C(0x52b8eee0217a6ecf),
    },
    {
        .seed = 12345,
        .ending = DINO_EPISODE_TRUNCATED,
        .score = 17,
        .episode_return = 16.1499977f,
        .end_tick = 1999,
        .collision_tick = UINT32_MAX,
        .jump_count = sizeof(SEED_12345_JUMPS) / sizeof(*SEED_12345_JUMPS),
        .jump_ticks = SEED_12345_JUMPS,
        .digest = UINT64_C(0x9ce7075622c9ef32),
    },
};

static int replay_matches_golden(const DinoReplay* replay,
        const DinoGoldenReplay* golden) {
    const DinoTrajectoryStep* final =
        &replay->steps[replay->step_count - 1];
    return replay->ending == golden->ending && final->score == golden->score &&
        final->episode_return == golden->episode_return &&
        replay->end_tick == golden->end_tick &&
        replay->collision_tick == golden->collision_tick &&
        replay->jump_count == golden->jump_count &&
        memcmp(replay->jump_ticks, golden->jump_ticks,
            golden->jump_count * sizeof(*golden->jump_ticks)) == 0 &&
        replay->digest == golden->digest;
}

static int verify_official_replays(void) {
    Dino env;
    float observations[DINO_OBSERVATION_COUNT] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    init_dino(&env, observations, actions, rewards, terminals);

    Weights* weights = load_weights("resources/dino/dino_weights.bin");
    int logit_sizes[1] = {2};
    PufferNet* net = make_puffernet(
        weights, 1, DINO_OBSERVATION_COUNT, 128, 1, logit_sizes, 1
    );
    DinoReplay* first = (DinoReplay*)calloc(1, sizeof(*first));
    DinoReplay* second = (DinoReplay*)calloc(1, sizeof(*second));
    int valid = 1;

    for (size_t i = 0; i < sizeof(OFFICIAL_REPLAYS) /
            sizeof(*OFFICIAL_REPLAYS); i++) {
        const DinoGoldenReplay* golden = &OFFICIAL_REPLAYS[i];
        DinoChallengeRecipe recipe = {
            .schema_version = DINO_CHALLENGE_SCHEMA_VERSION,
            .environment_version = DINO_ENVIRONMENT_VERSION,
            .policy_version = DINO_POLICY_VERSION,
            .seed = golden->seed,
        };
        replay_dino(&env, net, &recipe, first);

        /* Exercise unrelated state before proving the seeded reset isolates it. */
        env.rng = golden->seed + 77u;
        c_reset(&env);
        memset(net->mingru->state, 0x5a,
            net->mingru->num_layers * net->mingru->batch_size *
                net->mingru->hidden_size * sizeof(float));
        replay_dino(&env, net, &recipe, second);

        int deterministic = replays_equal(first, second);
        int golden_match = replay_matches_golden(first, golden);
        printf("seed=%u deterministic=%s golden=%s\n", golden->seed,
            deterministic ? "ok" : "FAILED",
            golden_match ? "ok" : "FAILED");
        if (!deterministic || !golden_match) {
            print_replay(&recipe, first);
            valid = 0;
        }
    }

    free(second);
    free(first);
    free_puffernet(net);
    free(weights);
    return valid ? 0 : 1;
}

static int parse_seed(const char* text, uint32_t* seed) {
    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || *text == '\0' || *end != '\0' || value > UINT32_MAX) {
        return 0;
    }
    *seed = (uint32_t)value;
    return 1;
}

int main(int argc, char* argv[]) {
    if (argc == 1) {
        demo();
    } else if (strcmp(argv[1], "--headless") == 0) {
        run_headless_evaluation(100, 0);
    } else if (strcmp(argv[1], "--trace") == 0) {
        run_headless_evaluation(1, 1);
    } else if (strcmp(argv[1], "--replay") == 0 && argc == 3) {
        uint32_t seed;
        if (!parse_seed(argv[2], &seed)) {
            fprintf(stderr, "Invalid replay seed: %s\n", argv[2]);
            return 1;
        }
        return run_replay_cli(seed);
    } else if (strcmp(argv[1], "--verify-replay") == 0 && argc == 2) {
        return verify_official_replays();
    } else {
        fprintf(stderr,
            "Usage: %s [--headless|--trace|--replay <seed>|--verify-replay]\n",
            argv[0]);
        return 1;
    }
    return 0;
}

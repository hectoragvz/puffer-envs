#include "dino.h"
#include "puffernet.h"

#define EVALUATION_MAX_STEPS 2000

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

void init_dino(Dino* env, float* observations, float* actions,
        float* rewards, float* terminals) {
    *env = (Dino) {
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

    env->observations = observations;
    env->actions = actions;
    env->rewards = rewards;
    env->terminals = terminals;
}

void demo() {
    Dino env;
    float observations[5] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    init_dino(&env, observations, actions, rewards, terminals);

    Weights* weights = load_weights("resources/dino_weights_20260722_234440_UTC.bin");
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

void run_headless_evaluation(int episodes, int trace) {
    Dino env;
    float observations[5] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    init_dino(&env, observations, actions, rewards, terminals);

    Weights* weights = load_weights("resources/dino_weights_20260722_234440_UTC.bin");
    int logit_sizes[1] = {2};
    PufferNet* net = make_puffernet(weights, 1, 5, 128, 1, logit_sizes, 1);
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

        while (!env.terminals[0] && episode_steps < EVALUATION_MAX_STEPS) {
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

int main(int argc, char* argv[]) {
    if (argc == 1) {
        demo();
    } else if (strcmp(argv[1], "--headless") == 0) {
        run_headless_evaluation(100, 0);
    } else if (strcmp(argv[1], "--trace") == 0) {
        run_headless_evaluation(1, 1);
    } else {
        fprintf(stderr, "Usage: %s [--headless|--trace]\n", argv[0]);
        return 1;
    }
    return 0;
}

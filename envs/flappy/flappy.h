#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;

// What can be on a tile
const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char WALL = 2;

// This keeps a log of the activity
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

// Required that you have some struct for your env
typedef struct {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int tick;
    int agent_row;
    int wall_row;
    unsigned int rng;
} Flappy;

void add_log(Flappy* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required
void c_reset(Flappy* env) {
    int tiles = 2;
    memset(env->observations, 0, tiles*sizeof(unsigned char));
    int a_row = 0;
    int w_row = 1;
    env->observations[a_row] = AGENT;
    env->observations[w_row] = WALL;
    env->tick = 0;
    env->agent_row = a_row;
    env->wall_row = w_row;
}

// Required function
void c_step(Flappy* env) {
    env->tick += 1;
    int action = (int)env->actions[0]; // UP or DOWN from flappy.c
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    if (action == DOWN) {
        env->agent_row = 1;
    } else if (action == UP) {
        env->agent_row = 0;
    }

    if (env->agent_row == env->wall_row) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        add_log(env);
        c_reset(env);
        return;
    }
    env->observations[0] = EMPTY;
    env->observations[1] = EMPTY;

    env->wall_row = 1 - env->agent_row;

    env->observations[env->agent_row] = AGENT;
    env->observations[env->wall_row] = WALL;
}

// Required function. Should handle creating the client on first call
void c_render(Flappy* env) {
    if (!IsWindowReady()) {
        InitWindow(64, 128, "PufferLib Flappy");
        SetTargetFPS(5);
    }
    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    int px = 64;
    for (int row = 0; row < 2; row++) {
        int tex = env->observations[row];
        Color color = (Color){40, 40, 40, 255};
        if (tex == AGENT) {
            color = (Color){0, 187, 187, 255};
        } else if (tex == WALL) {
            color = (Color){140, 90, 50, 255};
        }
        DrawRectangle(0, row*px, px, px, color);
    }
    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Flappy* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

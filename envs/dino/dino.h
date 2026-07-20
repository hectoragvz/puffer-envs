#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char NOOP = 0;
// Jumping
const unsigned char JUMP = 1;

#define GRAVITY 2.0f
#define JUMP_IMPULSE 18.0f
#define OBSTACLE_SPEED 8.0f
#define JUMP_PENALTY 0.01f

typedef struct {
    // Required - only use floats!
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    float x; // X position
    float height;
    float width;
} Obstacle ;

typedef struct {
    float y; // Position
    float y_velocity; // jumping physics
    float x; // for positioning
    // Body of the dino
    float height;
    float width;
} Dinosaur;

typedef struct {
    Log log;
    Dinosaur dinosaur;
    Obstacle obstacle;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int tick;
    int obstacles_passed;
    float episode_return;
    // Size of the env
    float height;
    float width;
    unsigned int rng;
} Dino;

void add_log(Dino* env) {
    env->log.perf += (env->obstacles_passed > 0) ? 1 : 0;
    env->log.score += env->obstacles_passed;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.n++;
}

void spawn_obstacle(Dino* env) {
    int extra_distance = rand_r(&env->rng) % ((int)env->width / 2 + 1);
    env->obstacle.x = env->width + extra_distance;
}

void update_observations(Dino* env) {
    float max_obstacle_x = env->width * 1.5f;
    env->observations[0] = env->dinosaur.y / env->height;
    env->observations[1] = env->dinosaur.y_velocity / JUMP_IMPULSE;
    env->observations[2] =
        (env->obstacle.x - (env->dinosaur.x + env->dinosaur.width)) /
        max_obstacle_x;
    env->observations[3] = env->obstacle.width / env->width;
    env->observations[4] = env->obstacle.height / env->height;
}

void c_reset(Dino* env){
    // Rest dino to starting position
    env->dinosaur.y = 0;
    env->dinosaur.y_velocity = 0;
    // Reset obstacle beyond the right edge with a varied gap.
    spawn_obstacle(env);
    env->tick = 0;
    env->obstacles_passed = 0;
    env->episode_return = 0;
    update_observations(env);
}

void c_step(Dino* env) {
    env->tick += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0;
    int action = (int)env->actions[0]; // NOOP or JUMP
    // If dino on ground, we jump and mod y_velocity
    if (action == JUMP && env->dinosaur.y == 0){
        env->dinosaur.y_velocity = JUMP_IMPULSE;
        env->rewards[0] = -JUMP_PENALTY;
        env->episode_return += env->rewards[0];
    }
    // if dino not on ground, gravity acts
    env->dinosaur.y_velocity -= GRAVITY;
    // Jumping or descending, we update the position
    env->dinosaur.y += env->dinosaur.y_velocity;
    // Have we landed?
    if (env->dinosaur.y <= 0){
        env->dinosaur.y = 0;
        env->dinosaur.y_velocity = 0;
    }
    // move obstacle
    env->obstacle.x -= OBSTACLE_SPEED;
    // Collision
    if (env->dinosaur.x + env->dinosaur.width > env->obstacle.x &&
        env->dinosaur.x < env->obstacle.x + env->obstacle.width &&
        env->dinosaur.y < env->obstacle.height){
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        env->episode_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }
    // Reward passing the obstacle
    if (env->obstacle.x + env->obstacle.width < env->dinosaur.x + env->dinosaur.width){
        env->rewards[0] = 1.0f;
        env->obstacles_passed += 1;
        env->episode_return += env->rewards[0];
        spawn_obstacle(env);
    }
    update_observations(env);
}

void c_render(Dino* env) {
    if (!IsWindowReady()) {
        InitWindow((int)env->width, (int)env->height, "PufferLib Dino");
        SetTargetFPS(60);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    int ground_y = (int)env->height - 20;
    int dino_x = (int)env->dinosaur.x;
    int dino_y = ground_y - (int)env->dinosaur.height - (int)env->dinosaur.y;
    int obstacle_x = (int)env->obstacle.x;
    int obstacle_y = ground_y - (int)env->obstacle.height;

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawLine(0, ground_y, (int)env->width, ground_y, (Color){200, 200, 200, 255});
    DrawRectangle(
        dino_x,
        dino_y,
        (int)env->dinosaur.width,
        (int)env->dinosaur.height,
        (Color){0, 187, 187, 255}
    );
    DrawRectangle(
        obstacle_x,
        obstacle_y,
        (int)env->obstacle.width,
        (int)env->obstacle.height,
        (Color){187, 0, 0, 255}
    );
    EndDrawing();
}

void c_close(Dino* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

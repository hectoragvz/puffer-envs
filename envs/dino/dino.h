#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "raylib.h"

const unsigned char NOOP = 0;
// Jumping
const unsigned char JUMP = 1;

#define GRAVITY 2.0f
#define JUMP_IMPULSE 18.0f
#define OBSTACLE_SPEED 8.0f
#define JUMP_PENALTY 0.05f

#define DINO_CHALLENGE_SCHEMA_VERSION 1u
#define DINO_ENVIRONMENT_VERSION 1u
#define DINO_POLICY_VERSION 1u
#define DINO_OBSERVATION_COUNT 5u
#define DINO_REPLAY_LIMIT 2000u
#define DINO_POLICY_WEIGHTS_SHA256 \
    "35136a8c398b2c47affeb0a3a55673d18c6dc082db03849f1f2ce9a57c49923e"

typedef enum {
    DINO_CHALLENGE_EVENT_SPEED = 1,
    DINO_CHALLENGE_EVENT_OBSTACLE = 2,
} DinoChallengeEventType;

typedef struct {
    uint32_t tick;
    uint32_t type;
    float value;
} DinoChallengeEvent;

typedef struct {
    uint32_t schema_version;
    uint32_t environment_version;
    uint32_t policy_version;
    uint32_t seed;
    const DinoChallengeEvent* events;
    uint32_t event_count;
} DinoChallengeRecipe;

typedef enum {
    DINO_REPLAY_OK = 0,
    DINO_REPLAY_UNSUPPORTED_SCHEMA,
    DINO_REPLAY_UNSUPPORTED_ENVIRONMENT,
    DINO_REPLAY_UNSUPPORTED_POLICY,
    DINO_REPLAY_UNSUPPORTED_EVENTS,
} DinoReplayStatus;

typedef enum {
    DINO_EPISODE_COLLISION = 1,
    DINO_EPISODE_TRUNCATED = 2,
} DinoEpisodeEnding;

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
    Texture2D dinosaur;
    Texture2D cactus;
} DinoClient;

typedef struct {
    Log log;
    Dinosaur dinosaur;
    Obstacle obstacle;
    Obstacle cleared_obstacle;
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
    int auto_reset;
    int cleared_obstacle_active;
    DinoClient* client;
} Dino;

void add_log(Dino* env) {
    env->log.perf += (env->obstacles_passed > 0) ? 1 : 0;
    env->log.score += env->obstacles_passed;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.n++;
}

uint32_t dino_random(Dino* env) {
    uint32_t state = (uint32_t)env->rng;
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    env->rng = state;
    return state;
}

void spawn_obstacle(Dino* env) {
    int extra_distance = dino_random(env) % ((int)env->width / 2 + 1);
    env->obstacle.x = env->width + extra_distance;
}

DinoReplayStatus dino_validate_recipe(const DinoChallengeRecipe* recipe) {
    if (recipe->schema_version != DINO_CHALLENGE_SCHEMA_VERSION) {
        return DINO_REPLAY_UNSUPPORTED_SCHEMA;
    }
    if (recipe->environment_version != DINO_ENVIRONMENT_VERSION) {
        return DINO_REPLAY_UNSUPPORTED_ENVIRONMENT;
    }
    if (recipe->policy_version != DINO_POLICY_VERSION) {
        return DINO_REPLAY_UNSUPPORTED_POLICY;
    }
    if (recipe->event_count != 0) {
        return DINO_REPLAY_UNSUPPORTED_EVENTS;
    }
    return DINO_REPLAY_OK;
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
    env->cleared_obstacle_active = 0;
    // Reset obstacle beyond the right edge with a varied gap.
    spawn_obstacle(env);
    env->tick = 0;
    env->obstacles_passed = 0;
    env->episode_return = 0;
    update_observations(env);
}

void dino_seeded_replay_reset(Dino* env, uint32_t seed) {
    env->rng = seed;
    env->log = (Log) {0};
    env->actions[0] = 0;
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->cleared_obstacle = (Obstacle) {0};
    c_reset(env);
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
    if (env->cleared_obstacle_active) {
        env->cleared_obstacle.x -= OBSTACLE_SPEED;
        if (env->cleared_obstacle.x + env->cleared_obstacle.width < 0) {
            env->cleared_obstacle_active = 0;
        }
    }
    // Collision
    if (env->dinosaur.x + env->dinosaur.width > env->obstacle.x &&
        env->dinosaur.x < env->obstacle.x + env->obstacle.width &&
        env->dinosaur.y < env->obstacle.height){
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        env->episode_return += env->rewards[0];
        add_log(env);
        if (env->auto_reset) c_reset(env);
        return;
    }
    // Reward and replace the logical obstacle at the original pass boundary.
    if (env->obstacle.x + env->obstacle.width <
        env->dinosaur.x + env->dinosaur.width) {
        env->cleared_obstacle = env->obstacle;
        env->cleared_obstacle_active = 1;
        env->rewards[0] = 1.0f;
        env->obstacles_passed += 1;
        env->episode_return += env->rewards[0];
        spawn_obstacle(env);
    }
    update_observations(env);
}

void c_render(Dino* env) {
    if (env->client == NULL) {
        InitWindow((int)env->width, (int)env->height, "PufferLib Dino");
        SetTargetFPS(60);
        env->client = (DinoClient*)calloc(1, sizeof(DinoClient));
        env->client->dinosaur = LoadTexture("resources/dino/dinosaur.png");
        env->client->cactus = LoadTexture("resources/dino/cactus.png");
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
    const char* score = TextFormat("Score: %d", env->obstacles_passed);
    DrawText(score, (int)env->width - MeasureText(score, 20) - 16,
        16, 20, RAYWHITE);
    DrawTexture(env->client->dinosaur, dino_x, dino_y, WHITE);
    DrawTexture(env->client->cactus, obstacle_x, obstacle_y, WHITE);
    if (env->cleared_obstacle_active) {
        DrawTexture(
            env->client->cactus,
            (int)env->cleared_obstacle.x,
            ground_y - (int)env->cleared_obstacle.height,
            WHITE
        );
    }
    EndDrawing();
}

void c_close(Dino* env) {
    if (env->client != NULL) {
        UnloadTexture(env->client->dinosaur);
        UnloadTexture(env->client->cactus);
        free(env->client);
        env->client = NULL;
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
}

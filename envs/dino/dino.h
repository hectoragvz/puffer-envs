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
#define DINO_ENVIRONMENT_VERSION 2u
#define DINO_POLICY_VERSION 2u
#define DINO_OBSERVATION_COUNT 6u
#define DINO_REPLAY_LIMIT 2000u
#define DINO_CHALLENGE_EVENT_LIMIT DINO_REPLAY_LIMIT
#define DINO_TRAINING_SPEED_EVENT_LIMIT 2u
#define DINO_POLICY_WEIGHTS_SHA256 \
    "29f8c9a21e911daaaf36f8049bb648bd9257d9175c351c613b4758a88fc18a5e"

#define DINO_MIN_SPEED 1.0f
#define DINO_MAX_SPEED 2.0f
#define DINO_DEFAULT_SPEED 1.0f

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
    DINO_REPLAY_INVALID_EVENT_LIST,
    DINO_REPLAY_TOO_MANY_EVENTS,
    DINO_REPLAY_UNSUPPORTED_EVENT_TYPE,
    DINO_REPLAY_INVALID_EVENT_TICK,
    DINO_REPLAY_INVALID_EVENT_ORDER,
} DinoReplayStatus;

typedef enum {
    ENV_SPEED_ACCEPTED,
    ENV_SPEED_INVALID,
} DinoSpeedChangeResult;

typedef enum {
    DINO_TRAINING_NORMAL_1X,
    DINO_TRAINING_CONSTANT_2X,
    DINO_TRAINING_ONE_TRANSITION,
    DINO_TRAINING_TWO_TRANSITIONS,
} DinoTrainingSpeedScenario;

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
    float speed_multiplier;
    int randomize_speed;
    DinoChallengeEvent training_speed_events[DINO_TRAINING_SPEED_EVENT_LIMIT];
    uint32_t training_speed_event_count;
    uint32_t training_speed_event_index;
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
    if (recipe->event_count > DINO_CHALLENGE_EVENT_LIMIT) {
        return DINO_REPLAY_TOO_MANY_EVENTS;
    }
    if (recipe->event_count > 0 && recipe->events == NULL) {
        return DINO_REPLAY_INVALID_EVENT_LIST;
    }
    for (uint32_t i = 0; i < recipe->event_count; i++) {
        const DinoChallengeEvent* event = &recipe->events[i];
        if (event->type != DINO_CHALLENGE_EVENT_SPEED) {
            return DINO_REPLAY_UNSUPPORTED_EVENT_TYPE;
        }
        if (event->tick >= DINO_REPLAY_LIMIT) {
            return DINO_REPLAY_INVALID_EVENT_TICK;
        }
        if (i > 0 && event->tick < recipe->events[i - 1].tick) {
            return DINO_REPLAY_INVALID_EVENT_ORDER;
        }
    }
    return DINO_REPLAY_OK;
}

void update_observations(Dino* env) {
    // Normalized variables here - usually / by the max value of the variable
    float max_obstacle_x = env->width * 1.5f;
    env->observations[0] = env->dinosaur.y / env->height;
    env->observations[1] = env->dinosaur.y_velocity / JUMP_IMPULSE;
    env->observations[2] =
        (env->obstacle.x - (env->dinosaur.x + env->dinosaur.width)) /
        max_obstacle_x;
    env->observations[3] = env->obstacle.width / env->width;
    env->observations[4] = env->obstacle.height / env->height;
    env->observations[5] =
        (env->speed_multiplier - DINO_MIN_SPEED) /
        (DINO_MAX_SPEED - DINO_MIN_SPEED);
}

DinoSpeedChangeResult dino_change_speed(Dino* env, float new_speed) {
    if (new_speed != DINO_MIN_SPEED && new_speed != DINO_MAX_SPEED) {
        return ENV_SPEED_INVALID;
    }
    env->speed_multiplier = new_speed;
    // Since we have a new speed for the env
    update_observations(env);
    return ENV_SPEED_ACCEPTED;
}

uint32_t dino_apply_speed_events_at_tick(Dino* env,
        const DinoChallengeEvent* events, uint32_t event_count,
        uint32_t event_index, uint32_t tick,
        DinoSpeedChangeResult* results) {
    while (event_index < event_count && events[event_index].tick == tick) {
        DinoSpeedChangeResult result =
            dino_change_speed(env, events[event_index].value);
        if (results != NULL) results[event_index] = result;
        event_index++;
    }
    return event_index;
}

DinoTrainingSpeedScenario dino_training_speed_scenario(uint32_t roll) {
    roll %= 100u;
    if (roll < 40u) return DINO_TRAINING_NORMAL_1X;
    if (roll < 60u) return DINO_TRAINING_CONSTANT_2X;
    if (roll < 80u) return DINO_TRAINING_ONE_TRANSITION;
    return DINO_TRAINING_TWO_TRANSITIONS;
}

void dino_configure_training_speed_events(Dino* env) {
    env->training_speed_event_count = 0;
    env->training_speed_event_index = 0;
    if (!env->randomize_speed) return;

    DinoTrainingSpeedScenario scenario =
        dino_training_speed_scenario(dino_random(env));
    if (scenario == DINO_TRAINING_NORMAL_1X) return;

    DinoChallengeEvent* first = &env->training_speed_events[0];
    *first = (DinoChallengeEvent) {
        .type = DINO_CHALLENGE_EVENT_SPEED,
        .value = DINO_MAX_SPEED,
    };
    env->training_speed_event_count = 1;

    if (scenario == DINO_TRAINING_CONSTANT_2X) return;
    if (scenario == DINO_TRAINING_ONE_TRANSITION) {
        first->tick = 40u + dino_random(env) % 161u;
        return;
    }

    first->tick = 40u + dino_random(env) % 111u;
    env->training_speed_events[1] = (DinoChallengeEvent) {
        .tick = first->tick + 40u + dino_random(env) % 111u,
        .type = DINO_CHALLENGE_EVENT_SPEED,
        .value = DINO_MIN_SPEED,
    };
    env->training_speed_event_count = 2;
}

void c_reset(Dino* env){
    env->speed_multiplier = DINO_DEFAULT_SPEED;
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
    dino_configure_training_speed_events(env);
    env->training_speed_event_index = dino_apply_speed_events_at_tick(
        env, env->training_speed_events, env->training_speed_event_count,
        env->training_speed_event_index, 0, NULL
    );
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
    // move obstacle at nx speed
    float movement = OBSTACLE_SPEED * env->speed_multiplier;
    env->obstacle.x -= movement;
    if (env->cleared_obstacle_active) {
        env->cleared_obstacle.x -= movement;
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
    /* env->tick is now the next zero-based pre-step tick. */
    env->training_speed_event_index = dino_apply_speed_events_at_tick(
        env, env->training_speed_events, env->training_speed_event_count,
        env->training_speed_event_index, (uint32_t)env->tick, NULL
    );
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

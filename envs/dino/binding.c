#include "dino.h"

#define ENV_HEIGHT 250
#define ENV_WIDTH 800

#define DINOSAUR_PADDING 80

#define DINOSAUR_HEIGHT 48
#define DINOSAUR_WIDTH 40

#define OBSTACLE_HEIGHT 40
#define OBSTACLE_WIDTH 24

#define OBS_SIZE 5

#define Env Dino

#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor

#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = ENV_WIDTH;
    env->height = ENV_HEIGHT;
    env->dinosaur.x = DINOSAUR_PADDING;
    env->dinosaur.width = DINOSAUR_WIDTH;
    env->dinosaur.height = DINOSAUR_HEIGHT;
    env->obstacle.width = OBSTACLE_WIDTH;
    env->obstacle.height = OBSTACLE_HEIGHT;
    env->auto_reset = 1;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}

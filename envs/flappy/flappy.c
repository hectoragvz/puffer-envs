#include "flappy.h"
#include "puffernet.h"

void demo() {
    Flappy env = {0};
    env.observations = (unsigned char*)calloc(2, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    Weights* weights = load_weights("resources/squared/squared_weights.bin");
    int logit_sizes[1] = {3};
    PufferNet* net = make_puffernet(weights, 1, 2, 128, 1, logit_sizes, 1);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0.0f;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
        } else {
            float obs_f[2];
            for(int i=0; i<2; i++) obs_f[i] = (float)env.observations[i];
            forward_puffernet(net, obs_f, env.actions);
        }
        c_step(&env);
        c_render(&env);
    }

    free_puffernet(net);
    free(weights);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

int main() {
    demo();
    return 0;
}

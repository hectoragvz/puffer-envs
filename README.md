# puffer-envs

### Squared and Target

Both of these coming from puffer's github profile.

### Flappy

"Keep it so simple as to not even be useful. That will come next. If you can't think of anything, do a stripped down version of flappy bird on a 2-block tall grid. The agent can move either up or down. It observes whether there is a wall on the roof or floor. -1 reward for hitting the ceiling, 0 otherwise."

### Pufferosaurus web demo

The Dino environment is the source of truth. PufferLib 4.0 is used only as a
local build workspace, and the generated browser files are committed to the
website repository.

From the PufferLib checkout, sync the environment and assets:

```sh
cp ../Puffer/envs/dino/{dino.c,dino.h,binding.c} ocean/dino/
cp ../Puffer/envs/dino/dino.ini config/dino.ini
cp ../Puffer/resources/dino/{dino_weights.bin,dinosaur.png,dinosaur_ducking.png,cactus.png,meteor.png} resources/dino/
```

Install Emscripten, then build and test locally:

```sh
brew install emscripten
bash build.sh dino --fast
./dino
bash build.sh dino --web
python3 -m http.server 8000 --directory build/web/dino
```

### Dino deterministic replay baseline

Environment version 1 uses a Dino-owned 32-bit LCG and policy version 1 is tied
to the five-observation checkpoint whose SHA-256 is
`35136a8c398b2c47affeb0a3a55673d18c6dc082db03849f1f2ce9a57c49923e`.
Recipe event ticks are zero-based pre-step ticks: tick 0 is after reset and
before the first policy action. Version 1 recipes contain no events; speed and
obstacle events are reserved for a later phase and are currently rejected.
Future same-tick events will execute in their listed order.

After synchronizing the source and resources into PufferLib as shown above,
inspect or verify the checked-in baselines with:

```sh
bash build.sh dino --fast
./dino --replay 0
./dino --replay 1
./dino --replay 12345
./dino --verify-replay
```

Evaluate a compatible checkpoint on 100 fixed seeds under each Phase 1 speed
scenario with:

```sh
./dino --evaluate-suite all resources/dino/dino_weights.bin
./dino --evaluate-suite 1x path/to/checkpoint.bin
```

The available suites are `1x`, `2x`, `up`, `down`, `multi`, `curriculum`, and
`all`. The `curriculum` suite starts at 1× and switches permanently to 2× after
the seventh obstacle pass; the other suites use only their explicit speed
events. Output includes passes, returns, episode lengths, first-obstacle
crashes, truncations, mean takeoff distance at each speed, and meteor behavior.
Every suite must report `meteor_jump_qualification=passed`: a grounded jump
while the active obstacle is a meteor rejects the candidate. A checkpoint must
match the input dimension of the executable; evaluate five-observation
checkpoints with a Phase 0 build.

The official records live in `envs/dino/dino.c`. To intentionally regenerate
them after an environment or policy version change, run the three `--replay`
commands, update each golden summary, jump trace, and digest from their output,
then run `--verify-replay` and both native and WebAssembly builds. A digest is
FNV-1a over every recorded trajectory scalar, encoded least-significant byte
first; floats are hashed by their IEEE-754 bit patterns.

The Colab notebook pins both source repositories and archives every checkpoint,
the console training log, configuration, and a run manifest to Drive. Treat its
`latest-candidate.bin` as unevaluated until it has been compared with the other
archived checkpoints using the suites above. After selecting a candidate that
passes every suite, update the environment and policy versions, weight hash,
and official replay fixtures together with the promoted weight file.

Ordinary training starts every episode at 1× and switches permanently to 2×
after `training_speedup_after_passes` successful obstacle passes. The default
threshold is `7`. Set it to `0` in the configuration's `[env]` section for a
fixed-1× diagnostic run. Record a distinct `[train] seed` for repeated runs.

Open `http://localhost:8000/game.html`, then copy the bundle into the site:

```sh
mkdir -p ../site/pufferosaurus/game
cp build/web/dino/game.{js,wasm,data,wasm.map} ../site/pufferosaurus/game/
```

Serve the site repository and open `http://localhost:8000/pufferosaurus/`:

```sh
python3 -m http.server 8000 --directory ../site
```

After verifying the page and iframe, publish through the site's existing GitHub
Pages setup:

```sh
cd ../site
git add index.html pufferosaurus
git commit -m "Add Pufferosaurus showcase"
git push
```

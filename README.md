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
cp ../Puffer/resources/dino/{dino_weights.bin,dinosaur.png,cactus.png} resources/dino/
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

The official records live in `envs/dino/dino.c`. To intentionally regenerate
them after an environment or policy version change, run the three `--replay`
commands, update each golden summary, jump trace, and digest from their output,
then run `--verify-replay` and both native and WebAssembly builds. A digest is
FNV-1a over every recorded trajectory scalar, encoded least-significant byte
first; floats are hashed by their IEEE-754 bit patterns.

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

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

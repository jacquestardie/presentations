
Robosat
------------------------------------------------------------------------------- 

- TODO: Create a slide deck
- TODO: Bundle prepared data
- TODO: Find a bunch of flash drives
- TODO: CFN template for SageMaker


Resources
------------------------------------------------------------------------------- 

OSM Diaries:
- [daniel-j-h - RoboSat - robots at the edge of space!](https://www.openstreetmap.org/user/daniel-j-h/diary/44145)
- [daniel-j-h - RoboSat loves Tanzania](https://www.openstreetmap.org/user/daniel-j-h/diary/44321)
    - Associated tmp data `s3://mapbox/playground/danieljh/tanzania/`
    - OAM drone images, probabilities, masks `s3://robosat-public/28f28ffa-6b4e-46ee-91dc-a6404790a491/`
    - Raster tile endpoint (chrome-only) - `https://s3.amazonaws.com/robosat-public/28f28ffa-6b4e-46ee-91dc-a6404790a491/probabilities2/index.html#18.5/-6.1933296/39.2698587`
- [maning's - Data preparation for feature detection with RoboSat](https://www.openstreetmap.org/user/maning/diary/44462)


Outline
------------------------------------------------------------------------------- 

### RoboSat

> RoboSat is a pipeline for feature extraction from aerial or satellite imagery. 
> Features can be anything visually distinguishable - buildings, parking lots, roads, etc.
> You don't need to know or understand the network architecture if you don't want to.

### 🙏

[Daniel](https://github.com/daniel-j-h) & [Bhargav](https://github.com/bkowshik/).

### Why

- Data is easy now. 
- Grokking it is still hard (maybe harder?)
- Computers can be helpful.

> Some close-at-hand ideas
> - Validate OSM changesets. Is there really a building there, or are you meme-ing?
> - Show your mappers where they should look first.
> - Get a sense for how complete your map is in a particular region, for a particular feature. Does your town have enough soccer fields per capita? Probably not.

### Do you have special data?

Great! Robosat doesn't care, as long as you bring:

1. Imagery
2. Masks

### Design goals

1. Stay simple
2. Do a good job.

> We follow the 80/20 rule where 80% of the effects come from 20% of the causes: we strive for simplicity and maintainability over pixel-perfect results. If you can improve the model's accuracy by two percent points but have to add thousands of lines of code we most likely won't accept your changeset.

### Today's plan

We're going to:

1. Prepare some data
2. Train a model
3. Evaluate our predicted buildings

> Didn't bring your graphics card? That's fine - we have intermediate datasets you can work with while you're here.

Walkthrough
------------------------------------------------------------------------------- 

### Setup & Requirements

You're running Ubuntu/Debian and want to install from source.
```sh
See https://github.com/mapbox/robosat#installation

We don't have enough time to cover installation today.
```

You can run Docker, and have a GPU.

```sh
# configure nvidia-docker
# See https://github.com/NVIDIA/nvidia-docker#quickstart

$ docker pull mapbox/robosat:latest-gpu
$ docker run --runtime=nvidia -it mapbox/robosat:latest-gpu /bin/bash

# Check that robosat is available
$ ./rs --help
```

You can run Docker, and don't have a GPU.

```sh
$ docker pull mapbox/robosat:latest-cpu
$ docker run -it mapbox/robosat:latest-cpu /bin/bash

# Check that robosat is available
$ ./rs --help
```





1. A computer
2. Docker
> I can 
Installation options:
- Prebuilt docker images for CPU/GPU
- SageMaker processing stack
- Install from source yourself.

We've used AWS's p2/p3 instances, and 1080 TI's to train on.

Prediction can be run on either GPU or CPU.

#### Data sources

RoboSat was built to be as data-agnostic as possible. So long as your imagery can be transformed into a [slippy map format](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames), Robosat can operate on them.

A few options you might want to consider:

Any of the imagery you can see in [**mapbox.satellite**](https://www.mapbox.com/maps/satellite/) can be used.

```
https://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token={token}
```

Find your token at [/account/access-tokens](https://www.mapbox.com/account/access-tokens).

**OpenAerialMap** hosts imagery - this is what we'll use today.

**DigitalGlobe** has a [Maps API](https://platform.digitalglobe.com/maps-api/).

**Planet** runs a [Tile Service](https://developers.planet.com/docs/api/tile-services/)

```
https://tiles{0-3}.planet.com/data/v1/{item_type}/{item_id}/{z}/{x}/{y}.png?api_key={api-key}
```

Your source is going to impact the features you can predict. Higher zoom, smaller features.

#### Data preparation

First, we need to generate a training dataset. It will consist a) imagery b) masks for the features we want to predict.

1. Extract the category of GeoJSON features we're interesting in predicting from OpenStreetMap - `rs extract`
2. Determine which tiles overlap those features - `rs cover`
3. Get the data you need.
    - Download the tiles from a map endpoint - `rs download`
    - Download and tile them out yourself.
4. Rasterize the GeoJSON features which you're interested in predicting into mask images which we'll train our model with - `rs rasterize`
5. Create a configuration file.

#### Training and Modeling

Next, we'll generate the segmentation models we'll later use to predict with.

1. Use the `(image, mask)` pairs, to generate checkpoints containing our weights - `rs train`
2. Predict class probabilities for each image in our slippy map directory using those weights - `rs predict`
3. Generate segmentation masks for each class probability - `rs masks`

#### Post processing

Transform our segmentation results into cleaned and simplified geometries which will match what we expect to see in "reality".

1. Transform segmentation masks into simplified GeoJSON features - `rs features`
2. Merge adjacent GeoJSON features into single features - `rs merge`
3. Deduplicate those features - `rs dedupe`

#### And also

- rs export - Export a model in ONNX format for prediction with different backends
- rs serve - Serve tile masks with an on-demand segmentation tileserver

### Architecture

PyTorch

Uses a pretrained ResNet50 encoder. 

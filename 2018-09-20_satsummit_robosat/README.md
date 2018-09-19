- TODO: Create a slide deck
- TODO: Flush out the hard negative mining bit
- TODO: Bundle prepared data
- TODO: Find a bunch of flash drives
- TODO: CFN template for SageMaker
    - [Get RoboSat off of morecs and onto SageMaker](https://github.com/mapbox/robosat-internal/issues/192)
    - [RoboSat python3 environment on SageMaker](https://github.com/mapbox/sagemaker/issues/2)
- TODO: Try to integrate RoboSat with LabelMaker

# Introduction

### RoboSat

> RoboSat is a pipeline for feature extraction from aerial or satellite imagery. 
> Features can be anything visually distinguishable - buildings, parking lots, roads, etc.
> You don't need to know or understand the network architecture if you don't want to.

---

### 🙏

[Daniel](https://github.com/daniel-j-h) & [Bhargav](https://github.com/bkowshik/).

---

### Why

- Data is easy now. 
- Grokking it is still hard (maybe harder?)
- Computers can be helpful.

> Some close-at-hand ideas
> - Validate [OSM changesets](https://wiki.openstreetmap.org/wiki/Changeset). Is there really a building there, or are you meme-ing?
> - Show your mappers where they should look first.
> - Get a sense for how complete your map is in a particular region, for a particular feature. Does your town have enough soccer fields per capita? Probably not.

---

### Do you have special data?

Great! Robosat doesn't care, as long as you bring:

1. Imagery
2. Masks

---

### Design goals

1. Stay simple
2. Do a good job.

> We follow the 80/20 rule where 80% of the effects come from 20% of the causes: we strive for simplicity and maintainability over pixel-perfect results. If you can improve the model's accuracy by two percent points but have to add thousands of lines of code we most likely won't accept your changeset.

---

### Today's plan

We're going to:

1. Prepare some data
2. Train a model
3. Evaluate our predicted buildings

> Didn't bring your graphics card? That's fine - we have intermediate datasets you can work with while you're here.

---


Walkthrough
=============================================================================== 

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
3. 
> I can 
Installation options:
- Prebuilt docker images for CPU/GPU
- SageMaker processing stack
- Install from source yourself.

We've used AWS's p2/p3 instances, and 1080 TI's to train on.

Prediction can be run on either GPU or CPU.

---

### Data sources

RoboSat was built to be data-agnostic. If your imagery can be transformed into a [slippy map format](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames), Robosat can operate on them.

- **Mapbox** - any of the imagery you can see in [**mapbox.satellite**](https://www.mapbox.com/maps/satellite/) can be used.
    - `https://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token={token}`
    - Find your token at [/account/access-tokens](https://www.mapbox.com/account/access-tokens).
- **OpenAerialMap** hosts imagery - this is what we'll use today.
- **DigitalGlobe** has a [Maps API](https://platform.digitalglobe.com/maps-api/).
- **Planet** runs a [Tile Service](https://developers.planet.com/docs/api/tile-services/)
- 💁‍♂️

> Your source is going to impact the features you can predict. Higher zoom, smaller features.

---

![](https://raw.githubusercontent.com/mapbox/robosat/master/assets/pipeline-01.png)

We need a training dataset. Specifically, **images** and **masks**, for those features we want to predict.

1. Extract the category of GeoJSON features we're interesting in predicting from OpenStreetMap
    - `rs extract`
3. Determine which tiles overlap those features
    - `rs cover`
5. Get the data you need.
    - Download the tiles from a map endpoint - `rs download`
    - Download and tile them out yourself.
6. Rasterize the GeoJSON features which you're interested in predicting into mask images which we'll train our model with - `rs rasterize`
7. Create a configuration file.

---


![](https://raw.githubusercontent.com/mapbox/robosat/master/assets/pipeline-02.png)

Next, we'll generate the segmentation models we'll later use to predict with.

1. Use the `(image, mask)` pairs, to generate checkpoints containing our weights - `rs train`
2. Predict class probabilities for each image in our slippy map directory using those weights - `rs predict`
3. Generate segmentation masks for each class probability - `rs masks`

---

![](https://raw.githubusercontent.com/mapbox/robosat/master/assets/pipeline-03.png)

Transform our segmentation results into cleaned and simplified geometries which will match what we expect to see in "reality".

1. Transform segmentation masks into simplified GeoJSON features - `rs features`
2. Merge adjacent GeoJSON features into single features - `rs merge`
3. Deduplicate those features - `rs dedupe`

---

### And also

- rs export - Export a model in ONNX format for prediction with different backends
- rs serve - Serve tile masks with an on-demand segmentation tileserver

---

See Also
=============================================================================== 

### OSM Diaries

Daniel's introductory post, [RoboSat - robots at the edge of space!](https://www.openstreetmap.org/user/daniel-j-h/diary/44145)

[RoboSat loves Tanzania](https://www.openstreetmap.org/user/daniel-j-h/diary/44321)

- Associated tmp data `s3://mapbox/playground/danieljh/tanzania/`
- OAM drone images, probabilities, masks `s3://robosat-public/28f28ffa-6b4e-46ee-91dc-a6404790a491/`
- Raster tile endpoint (chrome-only) - `https://s3.amazonaws.com/robosat-public/28f28ffa-6b4e-46ee-91dc-a6404790a491/probabilities2/index.html#18.5/-6.1933296/39.2698587`
    
[maning's - Data preparation for feature detection with RoboSat](https://www.openstreetmap.org/user/maning/diary/44462)
### Architecture

Currently:
- PyTorch
- [U-net](https://arxiv.org/abs/1505.04597)
- [ResNet](https://arxiv.org/abs/1512.03385) 50 encoder.
- Symmetrical decoder

> The encoder turns the image into a fixed size feature vector.
> The decoder turns those features back into full-resolution masks.
> Skip connections link encoder blocks deeper into the decoder, so that more of the high resolution information can be passed through. Without these connections, it would be much harder to decode the encoder's last output on it's own.

Previously:
- a [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105).

> Good for multiclass detections, and it's "context block" allows for the network to understand that if it identifies roads, it's then more likely to also detect cars, and less likely to detect water. Training and prediction were both very slow.

Future:
- [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) 
- [RetinaNet](https://arxiv.org/abs/1708.02002)
    - [Pull Request](https://github.com/mapbox/robosat/pull/75)

> FPNs are similar to u-nets. Channel addition rather than concatenation. FPN is commonly used as a base for other state-of-the-art networks.

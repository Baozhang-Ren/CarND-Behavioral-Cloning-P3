# CarND-Behavioral-Cloning-P3
Starting files for the Udacity CarND Behavioral Cloning Project

## Behavior Cloning Project from Udacity's Nanodegree, Self-Driving Engineer

## Content

### `model.py`
-- Python File to train model
Usage:

```sh
python model.py <data dir> <epochs> <batch_size>
```
A `model.h5` will be saved as output

### `Behavior_Cloning.ipynb`
--- Detail Explaination of `model.py`

### `drive.py`
--- After `model.h5` is created, it can be used with drive.py using this command:
```sh
python drive.py model.h5
```

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

### `video.py`
--- The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.
```sh
python video.py run1
```

### Output Video
<video src="./assets/output.mp4" width="320" height="200" controls preload></video>

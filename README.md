# Computer Vision with OpenCV

This repo is all my code for learning OpenCV in Python/C++

## Course Link

[Udemy Link](https://www.udemy.com/share/101ryu3@KoaCsdAdXgqsXaByBPRKdoUDlFw8K2GF3xPM4v5g-RmOJARN_VzDRULVDQPhqhsEEw==/)

## Requirments

[uv](https://docs.astral.sh/uv/) was used to build the projects on a MacBook. These can be run on Linux/Windows without change to code or method, except for the c++ projects which may require OS dependent compiler.

### Getting Started

Easiest way to try the python based notebooks and scripts is with uv and jupyter.

- Step-1: Clone repo:

```bash
git clone https://github.com/c2p-cmd/computer_vision_python.git
```

- Step-2: Sync with uv:
- This will automatically do all the heavy lifting

```bash
uv sync
```

- Step-3: Launch jupyter in the root of the project.

```bash
uv run jupyter lab
```

- Step-4: To run any of the `py` scripts is by using `uv run`
- For example to run [36_connecting_camera.py](./opencv_learning/cv_video/36_connecting_camera.py)

```bash
cd ./opencv_learning/cv_video
uv run 36_connecting_camera.py
```

***Happy Learning!***

## Repo Structure

### The repo is split into OpenCV Learning Projects, Deep Learning Projects and data used for projects

```bash
├── deep_learning
│   └── hand_segmentation
├── opencv_learning
│   ├── course_assessments
│   ├── cv_video
│   │   └── drawing_on_live_camera
│   ├── fundamentals
│   ├── object_detection
│   └── vision_applications
│       ├── live_face_detection_cpp
│       ├── optical_flow_cpp
│       ├── russian_plate_detector_cpp
│       └── watershed_image_segmentation_cpp
└── resources
    ├── 00-NumPy-and-Image-Basics
    ├── 01-Image-Basics-with-OpenCV
    ├── 02-Image-Processing
    ├── 03-Video-Basics
    │   └── __pycache__
    ├── 04-Object-Detection
    ├── 05-Object-Tracking
    ├── 06-Deep-Learning-Computer-Vision
    │   └── 06-YOLOv3
    │       ├── cfg
    │       ├── data
    │       ├── images
    │       │   ├── res
    │       │   └── test
    │       ├── model
    │       │   └── __pycache__
    │       └── videos
    │           ├── res
    │           └── test
    ├── 07-Capstone-Project
    ├── CATS_DOGS
    │   ├── test
    │   │   ├── CAT
    │   │   └── DOG
    │   └── train
    │       ├── CAT
    │       └── DOG
    └── DATA
        ├── haarcascades
        └── lbpcascades

45 directories
```

> **Note:** *Each folder ending with `_cpp` is the C++ equivalent of the Python algorithm.*

For example: [`live_face_detection_cpp`](./opencv_learning/vision_applications/live_face_detection_cpp/) has a simple script to build and run it using the `clang` compiler.

## Contributing

I am open to any suggestions in terms of code improvement or projects/challenges! <br>
Just open a GitHub Issue! :-)

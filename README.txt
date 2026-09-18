Traffic Cone Detection and Segmentation for Formula Student Autonomous Driving
Authors: Alberto granati, Elena Novkovic, Edoardo Renzi

Requirements
    C++ requirements

    - C++17 compatible compiler
    - CMake >= 3.10
    - OpenCV
    - ZLIB

    The C++ part of the project is responsible for:
    - launching the Python deep-learning inference pipeline;
    - loading detection and segmentation predictions;
    - computing evaluation metrics;
    - generating result visualizations and reports.


    Python requirements
    Required packages:

    - Python 3.x
    - TensorFlow
    - KerasCV
    - OpenCV-Python
    - NumPy
    - Matplotlib


Dataset

The provided test set must be placed in the following directory structure:

dataset/
└── test_set/
    └── segmentation_test/
        ├── ann/
        └── img/

In order to run the trainining of the detection part you can use:
    python -m detection.detection_yolo
    if you are in the python directory (formula-student-cone-perception/python)

    PYTHONPATH=python python -m detection.detection_yolo
    if you are in the root of the directory (formula-student-cone-perception)
In detection_yolo.py at line 136 there's flag variable is_train.
    if it is False then the training is skip and it's performed the validation directly.
    if it is True then the training is done and then there will be a comparison between the new model and the old best model
        based on the validation set.

In order to run the training of the segmentation part you can use:
    python -m segmentation.train_segmentation
    if you are in the python directory (formula-student-cone-perception/python)

    PYTHONPATH=python python -m segmentation.train_segmentation
    if you are in the root of the directory (formula-student-cone-perception)

In order to run the complete pipeline on the test_set in which each image is captured one by one and after all
we measure the metrics you just need to enters in formula-student-cone-perception/build and directory
    make
    ./main

It will generate a directory called "evaluation" that will contain every test image with classification, detection and
segmentation. Also there will be the score metrics.
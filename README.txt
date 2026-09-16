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
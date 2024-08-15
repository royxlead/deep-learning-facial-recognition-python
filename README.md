# Deep Learning Facial Recognition

## Overview

This project implements a facial recognition system using a Siamese network. The model is trained to distinguish between different individuals by learning to measure the similarity between image pairs. This repository includes a Jupyter notebook that contains the complete implementation, from data collection to model training and real-time verification.

## Features

- **Image Collection**: Collects anchor, positive, and negative images using a webcam.
- **Data Preprocessing**: Preprocesses images for training using TensorFlow.
- **Model Training**: Implements a Siamese network for facial recognition.
- **Real-Time Verification**: Verifies facial recognition in real-time using OpenCV.

## Prerequisites

Ensure you have the following libraries installed:

- TensorFlow
- OpenCV
- NumPy
- Matplotlib

You can install these libraries using pip:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/heysouravroy/DL_Facial_Recognition.git
   cd DL_Facial_Recognition
   ```

2. **Directory Structure**

   Ensure the following directory structure is in place:

   ```
   project/
   ├── data/
   │   ├── positive/
   │   ├── negative/
   │   └── anchor/
   └── application_data/
       ├── input_image/
       └── verification_images/
   ```

   The `data/` directory is used for collecting and storing images for training, while `application_data/` is used for storing images that will be used for real-time verification.

3. **Running the Notebook**

   Open the Jupyter notebook (`deep_learning_facial_recognition.ipynb`) in Jupyter Notebook or JupyterLab and execute the cells to run the project. 

   ```bash
   jupyter notebook deep_learning_facial_recognition.ipynb
   ```

## Usage

1. **Collect Images**

   - Run the notebook and follow the instructions to collect anchor and positive images using your webcam.
   - Press `a` to save an anchor image.
   - Press `p` to save a positive image.
   - Press `q` to quit image collection.

2. **Train the Model**

   The notebook includes code to train the Siamese network for facial recognition. Adjust the number of epochs and batch size as needed.

3. **Verify Faces**

   - Press `v` to save the current webcam frame as the input image for verification.
   - The model will predict whether the face in the image matches the collected images and print the result.

## Notes

- Ensure you have a good balance of positive, negative, and anchor images for effective training.
- Modify the paths in the notebook as needed based on your directory structure.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow for providing powerful machine learning tools.
- OpenCV for real-time image processing capabilities.

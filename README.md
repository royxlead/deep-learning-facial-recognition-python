<!-- Badges: update these if you add CI or a model release -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)

# Deep Learning Facial Recognition (Python)

A compact, educational demo that implements facial verification using a Siamese neural network. The main workflow is provided as a Jupyter notebook (`deep_learning_facial_recognition.ipynb`) that walks through image collection, preprocessing, model training, and a simple verification routine using OpenCV.

This repository is intended for learning and experimentation only. It demonstrates concepts (pairwise learning, L1 distance layer, webcam capture) and is not intended as a production-ready facial recognition system.

## Quick links

- Notebook: `deep_learning_facial_recognition.ipynb`
- Example inference script: `predict.py`
- Requirements: `requirements.txt`

## Features

- Webcam-based image collection helpers (anchor / positive / negative)
- Data preprocessing and dataset construction using TensorFlow
- Compact Siamese architecture with a custom L1 distance layer
- Training loop with checkpointing and a lightweight inference helper

## Quickstart (Windows PowerShell)

1. Clone the repository and change into it:

```powershell
git clone https://github.com/royxlead/deep-learning-facial-recognition-python.git
cd deep-learning-facial-recognition-python
```

2. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. (Optional) Create example directories the notebook expects:

```powershell
python -c "import os; [os.makedirs(p, exist_ok=True) for p in ['data/anchor','data/positive','data/negative','application_data/input_image','application_data/verification_images']]"
```

4. Start Jupyter and open the notebook:

```powershell
jupyter notebook deep_learning_facial_recognition.ipynb
```

Run the top-of-notebook setup cell first (it checks the environment, GPU availability, and sets seeds), then run cells in order.

## Running inference (example)

After training and saving a model (the notebook saves `siamese_model.keras` by default), you can run the provided `predict.py` script:

```powershell
python predict.py --input application_data/input_image/input_image.jpg --verification_dir application_data/verification_images --model siamese_model.keras --threshold 0.5
```

The script prints per-image similarity scores and a final verification ratio.

## Project layout

```
deep-learning-facial-recognition-python/
├── deep_learning_facial_recognition.ipynb   # Notebook (main workflow)
├── predict.py                               # Minimal inference wrapper
├── requirements.txt                          # Pinned dependency list
├── .gitignore
├── data/                                     # training data (anchor/positive/negative) - NOT committed
└── application_data/                         # input/verification images - NOT committed
```

## Notes & recommendations

- The notebook contains simplified preprocessing and a small model for educational use. For better results consider:
	- performing face alignment and cropping using a robust face detector
	- increasing dataset size and variety
	- replacing the toy embedding network with a pretrained backbone (MobileNet/ResNet) and fine-tuning

- Security & privacy: collecting and using face images has legal and ethical implications. Only use images you have permission to use and follow local laws and best practices.

## Contributing

Contributions and improvements are welcome. Suggested small first PRs:

- add a GitHub Actions workflow that runs a quick smoke test on `predict.py` (use CPU-only TensorFlow and small sample images)
- parameterize notebook paths and hyperparameters for non-interactive runs
- add an example dataset or scripts to download a small public sample

When opening PRs, include a short description and a smoke-test showing the change works.

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

## Acknowledgements

- TensorFlow — model building and training
- OpenCV — webcam capture and visualization

---
If you want, I can add a small sample image and a CI workflow that runs `predict.py` as a smoke test (CPU-only). Which would you prefer next?
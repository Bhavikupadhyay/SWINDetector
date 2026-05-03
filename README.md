# SWINDetector

## Overview
SWINDetector is a deep learning pipeline designed to detect deepfakes and manipulated media. It leverages the Swin Transformer architecture to classify image frames extracted from video sequences into real or manipulated categories. The project includes a complete workflow from dataset preprocessing to model training, evaluation, and a web-based inference interface.

## Features
* **Automated Video Preprocessing:** Extracts frames from video files at specified intervals and resizes them for model compatibility.
* **Train/Test Data Splitting:** Automatically shuffles and partitions extracted frames into an 80:20 training and testing split.
* **Swin Transformer Integration:** Fine-tunes the `microsoft/swin-tiny-patch4-window7-224` vision transformer using the Hugging Face ecosystem.
* **Comprehensive Evaluation:** Computes standard classification metrics including accuracy, F1 score, precision, and recall.
* **Interactive Web Demo:** Includes a Gradio-based web interface for running inferences on new images.

## Architecture
1. **Data Extraction:** Reads raw `.mp4` video files using OpenCV, capturing a frame every 5 frames, and resizes them to 224x224 pixels.
2. **Model Training:** Utilizes Hugging Face's `Trainer` API to fine-tune a pre-trained Swin Transformer for image classification on the processed dataset.
3. **Inference UI:** A Gradio application that loads the locally saved model and provides top-5 classification probabilities for uploaded images.

## Setup & Installation

1. Clone the repository and navigate into the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
  Ensure your data is organized in the root directory as follows:
  
  * dataset/original_sequences/ (for real videos)
  
  * dataset/manipulated_sequences/Deepfakes/ (for manipulated videos)
  
  * dataset/manipulated_sequences/Face2Face/

  * dataset/manipulated_sequences/FaceSwap/
  
  * dataset/manipulated_sequences/NeuralTextures/

## Usage
1. Data Preprocessing
Run the extraction script to process videos into images and split them into training and testing directories:

```bash
python image_extractor.py
```
2. Model Training
Begin the fine-tuning process. The script will automatically load the preprocessed data and train the Swin Transformer model:

```bash
python swin-tiny-complete-training.py
```
Note: The model checkpoints and evaluation metrics will be saved in the ./models/ and ./results/ directories.

3. Model Testing
To evaluate the trained model against the test dataset without initiating a training loop:

```bash
python model-testing.py
```
4. Run the Web Interface
Launch the Gradio demo to interact with your trained model:

```bash
python demo.py
```

## Tech Stack
* Python

* PyTorch

* Hugging Face Transformers & Datasets

* OpenCV

* Gradio

* Evaluate

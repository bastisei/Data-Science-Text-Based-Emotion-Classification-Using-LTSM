# Emotion Recognition with Bidirectional LSTM

This repository contains code for an **Emotion Recognition** project using a Bidirectional LSTM neural network. The model aims to classify textual data into different emotional categories. The project is designed with Python and leverages TensorFlow/Keras for deep learning, alongside other standard NLP and data science libraries.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training & Evaluation](#training--evaluation)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview
- **Goal**: Recognize and classify emotions in textual data.
- **Dataset**: The project uses a publicly available dataset that includes training, validation, and test sets.
- **Classes**: The data is categorized into the following emotional classes:
  - `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`.

## Features
- Text preprocessing with stemming and stopword filtering.
- Data visualization for class distributions and performance metrics.
- Bidirectional LSTM model with Dropout for generalization.
- Early stopping and learning rate reduction for better model performance.

## Requirements
Make sure you have the following libraries installed before running the project:
- `tensorflow` (2.x)
- `keras`
- `pandas`
- `matplotlib`
- `seaborn`
- `nltk`
- `sklearn`

Install required packages with:
```bash
pip install -r requirements.txt ````

### Data Preparation
1. Place the training, validation, and test datasets in the appropriate folders.
2. Preprocess the text data using stemming and stopword filtering.
3. Tokenize and pad sequences to a consistent length.

### Model Architecture
1. Embedding Layer: Turns input words into dense vectors.
2. Bidirectional LSTM Layer 1: Captures both past and future context.
3. Dropout Layer: Prevents overfitting.
4. LSTM Layer 2: Further contextual analysis.
5. Dense Output Layer: Final classification into emotion classes.

### Training & Evaluation
1. Training: The model is trained using cross-entropy loss and the Adam optimizer.
2. Evaluation: Confusion matrix and accuracy metrics are used to evaluate model performance.

### Results
1. Accuracy: The model achieved an accuracy of over 90% on the validation set.
2. Confusion Matrix: Visualized for detailed class performance.

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

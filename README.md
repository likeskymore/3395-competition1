# Beer quality dection

Team name: ift3395_trung_nguyen
Team members: Trung Nguyen - 20238006

This repository contains a complete pipeline for training a model to predict beer quality. The model is trained to classify the beer quality from 3-9 .

## Overview

The pipeline consists of several key components:

- **Data Preprocessing**: Apply preprocessing techniques on the data
- **Train/Validation Split**: Split the data into train and val set
- **Model Training**: Trains a DistilBERT classifier using HuggingFace Transformers
- **Evaluate Models**: Evaluate models using training and validation results
- **Generate Prediction**: Using the model to predict on unseen data

## Project Structure

```
3395-competition1/
├── notebook.ipynb                      # Notebook file for the training pipeline            
├── README.md                           # This file
├── requirements.txt                    # Required libraries to run the notebook
├── submission.csv/                     # Prediction output
├── data/
│   ├── sample_submission.csv/          # Kaggle submission sample
│   ├── test.csv/                       # Unseen test data
│   └── train.csv/                      # Training data
```

## Prerequisites

### Required Python Packages

Install the required packages:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Step-by-Step Training Process

### Step 1: Data Preprocessing

- Rows with missing values are dropped from the table
- For numeric values in the dataset, normalization is applied. And for non-numeric values, one-hot encoding is applied.
- Feature selection using SelectKBest

### Step 2: Train/Validation Split

- Split the dataset into train and val set while keeping both sets proportionated to each other

### Step 3: Model Training

- Apply linear regression with gradient descent onto the training/validation set

### Step 4: Evaluate Model

- Evalute model using the result from train and val set
- Evaluation by precision, recall, F1-score and confusion matrix

### Step 5: Generate Prediction

- Apply the same data preprocessing techniques onto the test set
- Pass the processed test data onto the model for prediction
- Output is a csv file formatted according to the ```sample_submission.csv```

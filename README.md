# A CNN model to classify the healthy cell and damaged cell.

# Overview 

This Project is a part of learning of use of AI in life science. Broder vision of this project is develope a model to predict the Drug Induced Liver Injury by using microscopic image of Liver on Chip.

# Motivation

Liver toxicity is a major reason for drug failure. Early prediction using AI can reduce costs and improve safety.

# Dataset

Before using the actual Datasets synthetic Cell images data is generated using Numpy and cv2.

Classes:
- Healthy cells
- Damaged cells

## Model Architecture
Custom CNN with:
- Conv2D
- MaxPooling
- Dropout
- Dense Layers

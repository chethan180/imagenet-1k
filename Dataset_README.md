# Tiny ImageNet-200 Dataset

## Overview
This directory contains the Tiny ImageNet-200 dataset, a subset of ImageNet designed for computer vision tasks with reduced computational requirements.

## Datset link to download
https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet?resource=download

## Dataset Summary

- **Classes**: 200 classes
- **Total Images**: 120,000 images
  - **Training**: 100,000 images (500 per class)
  - **Validation**: 10,000 images (50 per class)  
  - **Test**: 10,000 images (50 per class)

## Directory Structure

```
tiny-imagenet-200/
├── train/           # Training images organized by class directories
├── val/             # Validation images
│   └── val_annotations.txt
├── test/            # Test images
│   └── images/
├── wnids.txt        # WordNet IDs for all 200 classes
└── words.txt        # Class names and descriptions
```

## Dataset Characteristics

- **Image Size**: 64x64 pixels (RGB)
- **Balanced Dataset**: Each training class contains exactly 500 images
- **Class Organization**: Training images are organized in subdirectories by WordNet ID
- **Format**: JPEG images

## Files Description

- `wnids.txt`: Contains the WordNet IDs for all 200 classes
- `words.txt`: Contains class names and descriptions corresponding to WordNet IDs
- `val_annotations.txt`: Contains validation image labels and bounding boxes

This dataset is ideal for experimenting with image classification models while maintaining reasonable computational requirements.
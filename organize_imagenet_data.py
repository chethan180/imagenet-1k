#\!/usr/bin/env python3
import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path

def organize_imagenet_data():
    print('Organizing ImageNet-1K data into proper folder structure...')
    
    # Check if download completed
    cache_dir = '/home/ubuntu/datasets/imagenet-1k/cache'
    
    if not os.path.exists(cache_dir):
        print(f'Cache directory not found: {cache_dir}')
        return False
    
    # Create proper ImageNet structure
    imagenet_root = '/home/ubuntu/datasets/imagenet-1k/data'
    train_dir = os.path.join(imagenet_root, 'train')
    val_dir = os.path.join(imagenet_root, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print('Looking for dataset in cache...')
    
    try:
        # Try to load the dataset from cache
        print('Attempting to load dataset from cache...')
        from datasets import load_dataset
        from huggingface_hub import login
        
        # Login with token
        login('YOUR_HF_TOKEN_HERE')
        
        # Load dataset from cache
        dataset = load_dataset('ILSVRC/imagenet-1k', cache_dir=cache_dir)
        
        print('Dataset loaded successfully\!')
        print(f'Train samples: {len(dataset["train"])}')
        print(f'Validation samples: {len(dataset["validation"])}')
        
        # Now organize the data
        organize_hf_dataset(dataset, imagenet_root)
        
        return True
        
    except Exception as e:
        print(f'Error organizing data: {e}')
        return False

def organize_hf_dataset(dataset, output_dir):
    """Organize HuggingFace dataset into ImageNet folder structure"""
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    # Get class labels
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    
    # Process training data
    print('Processing training data...')
    process_split(train_dataset, train_dir, 'train')
    
    print('Processing validation data...')  
    process_split(val_dataset, val_dir, 'val')
    
    print('Data organization complete\!')

def process_split(dataset, output_dir, split_name):
    """Process a dataset split and save images"""
    
    # Get class names
    label_names = dataset.features['label'].names
    
    # Create directories for each class
    for class_name in label_names:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Process images in batches to avoid memory issues
    print(f'Processing {len(dataset)} images for {split_name}...')
    
    for idx in tqdm(range(len(dataset)), desc=f'Processing {split_name}'):
        try:
            sample = dataset[idx]
            image = sample['image']
            label = sample['label']
            class_name = label_names[label]
            
            # Save image
            image_path = os.path.join(output_dir, class_name, f'{idx:08d}.JPEG')
            image.save(image_path, 'JPEG', quality=95)
            
        except Exception as e:
            print(f'Error processing sample {idx}: {e}')
            continue

if __name__ == '__main__':
    success = organize_imagenet_data()
    if success:
        print('ImageNet data organization successful\!')
        print('Data structure:')
        print('  data/')
        print('    train/')
        print('      <class_name>/')
        print('        <image_files>')
        print('    val/')
        print('      <class_name>/')
        print('        <image_files>')
    else:
        print('Failed to organize ImageNet data')

#\!/usr/bin/env python3
import torch
import yaml
from src.models import get_model

def test_setup():
    print('Testing ImageNet training setup...')
    
    # Load config
    with open('configs/imagenet1k_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f'Config loaded: {config["model"]["name"]} with {config["model"]["num_classes"]} classes')
    
    # Test model creation
    model = get_model(config['model']['name'], num_classes=config['model']['num_classes'])
    print(f'Model created: {type(model).__name__}')
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f'Model output shape: {output.shape}')
    print(f'Expected shape: (1, {config["model"]["num_classes"]})')
    
    # Check CUDA availability
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name()}')
        print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    print('Setup test completed successfully\!')

if __name__ == '__main__':
    test_setup()

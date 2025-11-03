#\!/bin/bash

echo "Starting ImageNet-1K ResNet50 Training"
echo "======================================"

# Activate virtual environment
source ~/venv/bin/activate

# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check data is accessible
echo "Data verification:"
echo "Train samples:        0"
echo "Val samples:        0"

# Start training with wandb logging
echo "Starting training..."
nohup python train.py --config configs/imagenet1k_config.yaml > training.log 2>&1 &

echo "Training started in background. Monitor with: tail -f training.log"
echo "Wandb dashboard: https://wandb.ai/chyten9-personal/imagenet1k-resnet50"

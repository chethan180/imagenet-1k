import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import json
import requests

# -----------------------------
# ✅ Load ImageNet Class Labels
# -----------------------------
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return [line.strip() for line in response.text.split("\n")]

IMAGENET_CLASSES = load_imagenet_labels()


# -----------------------------
# ✅ Model Wrapper
# -----------------------------
class ImageNetResNet50Classifier:
    def __init__(self, model_path="resnet50_imagenet1k_cpu.pth"):
        # Force CPU
        self.device = torch.device("cpu")

        # Initialize model
        self.model = resnet50(num_classes=1000)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded trained model from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model, using random weights: {e}")

        self.model.eval()

        # Transform consistent with ImageNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image):
        if isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image.astype("uint8"), "RGB")

        # Preprocess image
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs[0], dim=0)

        # Top-5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        results = {
            IMAGENET_CLASSES[top5_catid[i].item()]: round(top5_prob[i].item(), 4)
            for i in range(5)
        }
        return results


# -----------------------------
# ✅ Gradio Interface
# -----------------------------
classifier = ImageNetResNet50Classifier()

def classify_image(image):
    if image is None:
        return {"Error": "Please upload an image"}
    return classifier.predict(image)


def create_demo():
    with gr.Blocks(title="ResNet50 ImageNet-1K Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 ResNet50 - ImageNet Classifier
        Upload any image and get top-5 ImageNet predictions.

        **Model:** ResNet50 trained on ImageNet-1K  
        **Runtime:** CPU (optimized for Hugging Face Spaces)
        """)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="numpy", height=300)
            with gr.Column():
                output = gr.Label(label="Predictions (Top-5)", num_top_classes=5)

        image_input.change(fn=classify_image, inputs=image_input, outputs=output)
        gr.Markdown("---\nBuilt with ❤️ using PyTorch + Gradio")

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)

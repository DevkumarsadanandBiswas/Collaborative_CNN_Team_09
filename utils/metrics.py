# metrices.py

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm


# ---------------------------------------------------------
# Create folders
# ---------------------------------------------------------
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------
# Save JSON utility
# ---------------------------------------------------------
def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------
# Compute evaluation metrics
# ---------------------------------------------------------
def compute_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


# ---------------------------------------------------------
# MAIN TEST SCRIPT
# ---------------------------------------------------------
def evaluate_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------
    # Load model
    # ------------------------------------------
    print("\nLoading Model Checkpoint...")
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(r"D:\PROJECT\Collaborative_CNN_Team_09\notebooks\checpoint_epoch_4.pt")["model_state_dict"])
    model.to(device)
    model.eval()

    # ------------------------------------------
    # Load Test Dataset
    # ------------------------------------------
    print("\nLoading Test Dataset...")

    test_dataset = ImageFolder(
        "D:/PROJECT/Collaborative_CNN_Team_09/data/test",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    )

    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    class_names = test_dataset.classes  # ["cat", "dog"]

    print("Detected Classes:", class_names)

    # ------------------------------------------
    # Generate Predictions
    # ------------------------------------------
    all_labels = []
    all_preds = []
    test_results = []

    print("\nEvaluating...")
    with torch.no_grad():
        for img, label in tqdm(dataloader, desc="Evaluating", unit="batch"):

            img = img.to(device)
            label = label.to(device)

            output = model(img)
            _, pred = torch.max(output, 1)

            true_label = label.item()
            pred_label = pred.item()

            all_labels.append(true_label)
            all_preds.append(pred_label)

            # store file path
            idx = len(test_results)
            img_path, _ = test_dataset.imgs[idx]

            test_results.append({
                "file": img_path,
                "true": class_names[true_label],
                "pred": class_names[pred_label]
            })

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # ------------------------------------------
    # Compute Metrics
    # ------------------------------------------
    print("\nComputing Metrics...")
    metrics = compute_metrics(all_labels, all_preds, class_names)

    # ------------------------------------------
    # Save Results
    # ------------------------------------------
    save_json(metrics, "results/metricsv2.json")
    save_json(test_results, "results/testv2user1.json")

    print("\nSaved:")
    print("  ✔ results/metricsv2.json")
    print("  ✔ results/testv2user1.json")
    print("\nEvaluation completed.")


# ---------------------------------------------------------
# Run main
# ---------------------------------------------------------
if __name__ == "__main__":
    evaluate_model()

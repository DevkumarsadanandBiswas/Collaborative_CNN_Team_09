# Final report.md including:

# 1) Base models used by both users
User1 base model: Simple CNN model
User2 base model : resnet50 model

# 2) Dataset descriptions
Dataset Description: Cat vs Dog — “user1- Cat & Dog Dataset (Tong Python)” / “user2- Dogs vs Cats Redux”

The Cat vs Dog dataset, commonly a popular image dataset used for binary image classification tasks. This dataset is widely used for training and evaluating convolutional neural networks (CNNs) due to its simplicity, balanced classes, and real-world variability.

## Overview

Task: Binary image classification

Classes:

Cats

Dogs

Type: RGB natural images

data/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/

 # 3) Metrics on both datasets
user1- training 
| **Epoch** | **Train Loss** | **Train Acc** | **Train F1** | **Val Loss** | **Val Acc** | **Val F1** |
| --------- | -------------- | ------------- | ------------ | ------------ | ----------- | ---------- |
| **1**     | 0.5963         | 0.6743        | 0.6743       | 0.5604       | 0.7123      | 0.7121     |
| **2**     | 0.5080         | 0.7493        | 0.7493       | 0.5005       | 0.7479      | 0.7477     |
| **3**     | 0.4570         | 0.7863        | 0.7862       | 0.4914       | 0.7637      | 0.7632     |
| **4**     | 0.4003         | 0.8141        | 0.8141       | 0.4837       | 0.7647      | 0.7645     |
| **5**     | 0.3339         | 0.8500        | 0.8500       | 0.5177       | 0.7756      | 0.7754     |

user1- testing on user2 dataset
| **Metric**        | **Value**  |
| ----------------- | ---------- |
| **Test Accuracy** | **0.7756** |
| **Test F1 Score** | **0.7754** |

user2 -training 
| **Metric**          | **Value**       |
| ------------------- | --------------- |
| **Final Accuracy**  | 98%         |
| **Correct / Total** | 1973 / 2001 |
| **Average Loss**    | 4.2053e-11  |

user2 -testing on user1 dataset
| **Class**            | **Precision** | **Recall** | **F1-Score** | **Support** |
| -------------------- | ------------- | ---------- | ------------ | ----------- |
| **Cat**              | 0.99          | 0.99       | 0.99         | 500         |
| **Dog**              | 0.99          | 0.99       | 0.99         | 500         |
| **Overall Accuracy** | 0.99          | —          | —            | 1000        |
| **Macro Avg**        | 0.99          | 0.99       | 0.99         | 1000        |
| **Weighted Avg**     | 0.99          | 0.99       | 0.99         | 1000        |

Confusion Matrix (Tabular Format)
| **Actual \ Predicted** | **Cat** | **Dog** |
| ---------------------- | ------- | ------- |
| **Cat**                | 493     | 7       |
| **Dog**                | 7       | 493     |


# 4) Observations on generalization and domain shift

# 1. Generalization Performance

The model demonstrates strong generalization ability, as shown by its high performance on the unseen test set:

99% test accuracy

Balanced precision, recall, and F1 across both cats and dogs

Very low misclassification (14/1000 images)

This indicates that:

1)The model is not simply memorizing the training data

It is learning robust visual features (fur texture, ear shapes, facial structures, posture patterns) that generalize well across different images.

2) No signs of overfitting

Even though training losses dropped significantly in some epochs, the test accuracy remained stable and high, meaning the network did not overfit to training noise.

3) The learned representation is stable

Both classes show equal performance, suggesting the model extracted class-invariant features that apply across cat and dog varieties.

# 2. Observations on Domain Shift

Domain shift refers to differences between training data distribution and test data distribution. This can include changes in:

Lighting conditions

Backgrounds

Image resolutions

Camera quality

Animal breeds

Occlusions

Pose variations

Despite these potential variations, our model achieves 99% accuracy,

# Collaborative_CNN_Team_09
CNN Model classification model Cat-Dog classification 

Assignment: Fork-Based Collaborative

CNN Development
Cross-Dataset CNN Collaboration Challenge

AI

ML Course Project

Objective
In this assignment, you will work in teams of two to collaboratively design, train, and
evaluate convolutional neural network (CNN) models for the same image classification
task, but using different datasets that cannot be shared.
This project emphasizes:
• Deep learning implementation with PyTorch or TensorFlow

• Full GitHub collaboration workflow (repository creation, forks, branches, pull re-
quests, and issues)

• Cross-domain testing and model comparison
Scenario
You and your teammate are studying the same computer vision problem (e.g., Cats vs
Dogs classification), but each uses a different dataset. Your goal is to build and version
models, test each others work without sharing raw data, and manage all collaboration
through GitHub.
Tasks
Step 1 Team Formation and Dataset Selection
1. Form a team of two students (User 1 and User 2).
2. Each student must select a different dataset for the same classification task.
Example options:
Note: You must not share dataset files; only trained models and metrics may be shared.
Task User 1 Dataset User 2 Dataset
Cat vs Dog Cat & Dog Dataset (Tong Python) Dogs vs Cats Redux
Pneumonia Detection Chest X-Ray (Pneumonia) COVID-19 Radiography Database
Leaf Disease Classification PlantVillage Dataset New Plant Diseases Dataset

1

Step 2 Repository Setup (User 1 Creates Base Repository)
1. User 1 creates a new public GitHub repository named:
colla bo ra ti v e c n n t eamXX
where XX is your team number or initials.
2. Create the following directory structure:
colla bo ra ti v e c n n t eamXX /
README.md
r e q ui r em e n t s . t x t
models /
no tebook s /
r e s u l t s /
u t i l s /

3. Add a .gitignore for datasets and model checkpoints (e.g., data/, *.pth, *.h5).
4. Add User 2 as a collaborator with Write permission.
5. Push an initial commit describing the project and dataset link.
Step 3 Model Version 1 (User 1)
1. Create a branch:
g i t checkou t −b d e v u s e r1
2. Choose a CNN base model (custom, ResNet, MobileNet, etc.).
3. Implement and train the model as models/model v1.py.
4. Save trained weights as models/model v1.pth.
5. Record metrics (accuracy, F1, etc.) in results/metrics v1.json.
6. Commit and push:
g i t add .
g i t commit −m ”Add model v1 and t r a i n i n g m e t ri c s ”
g i t push o r i g i n d e v u s e r1
7. Open a Pull Request (PR) to merge dev user1 into main.
Step 4 Testing by User 2
1. User 2 forks the base repository created by User 1.
2. Clone the fork locally and test model v1.pth on their dataset.
3. Save results as results/test v1 user2.json.
4. Open a GitHub Issue in the base repository titled:

Model v1 results on User 2 dataset, including performance observations and feed-
back.

Step 5 Model Version 2 (User 2)
1. On their fork, create a new branch:
g i t checkou t −b d e v u s e r2

2. Design an improved CNN model (add dropout, data augmentation, or new archi-
tecture).

2

3. Save as models/model v2.py and weights models/model v2.pth.
4. Log metrics in results/metrics v2.json.
5. Commit, push, and open a Pull Request titled:
Add model v2 trained on User 2 dataset.
Step 6 Cross-Test by User 1
1. Pull the updated main branch:
g i t p u l l o r i g i n main
2. Test model v2 on User 1s dataset.
3. Record results in results/test v2 user1.json.
4. Write a summary comparison of model v1 vs model v2 in report.md.
Expected Repository Structure
colla bo ra ti v e c n n t eamXX /
README.md
models /
model v1 . py
model v2 . py
no tebook s /
t r a i n v 1 . ipynb
t e s t v 1 . ipynb
t r a i n v 2 . ipynb
t e s t v 2 . ipynb
r e s u l t s /
m e t ri c s v 1 . j s o n
t e s t v 1 u s e r 2 . j s o n
m e t ri c s v 2 . j s o n
t e s t v 2 u s e r 1 . j s o n
u t i l s /
m e t ri c s . py
r e p o r t .md

Deliverables
Each team must submit:
• Link to the base GitHub repository (created by User 1).
• Link to User 2s fork and both merged Pull Requests.
• Final report.md including:
– Base models used by both users
– Dataset descriptions
– Metrics on both datasets
– Observations on generalization and domain shift

3

Optional Bonus Work
For additional credit:
• Add Grad-CAM visualizations for both models.
• Create a GitHub Action that verifies repository structure and metrics files.

• Implement domain adaptation or transfer learning for better cross-dataset perfor-
mance.

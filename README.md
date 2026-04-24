# MPHY0050
Repository for MPHY0050 Group Project.

The work is divided into the following core tasks:
1) Baseline classification solution
2) Solution including strategies to account for imbalance data
3) Solution using multi-task learning
4) Solution with explainability markers
5) Solution incorporating uncertainty evaluation
6) Solution accounting for some source of bias
7) A mixture of at least 3 out of (imbalance, multi-task, interpretability, uncertainty, bias)

Each script (e.g. Task1.ipynb) corresponds to the tasks as listed above.

## Task1.ipynb
Currently implemented: 
- Histograms to view intensity distributions for the 5 classes
- Basic segmentation using blurring + thresholding

## Task2.ipynb
Currently implemented:
- Bar chart to show class imbalance
- Dataloader with transformations (grayscale, basic normalisation)
- RGB ResNet training without class imbalance 
- RGB ResNet training with class weights
- RGB ResNet training with focal loss loss function
 
## Task3.ipynb
Currently implemented:
- Multi-task learning with hard parameter sharing (ResNet-18 backbone + 3 task-specific heads)
- GradNorm for task weight balancing during training
- Single-task baseline
- Task evaluation to verify that age and sex heads are being learned

## Task4.ipynb
Currently implemented:
- ResNet-18 classifier for 5-class myopic maculopathy grading
- Grad-CAM
- Occlusion sensitivity
- Comparison of both explainability methods across all grades

## Task5.ipynb
Currently implemented:
- Dataloader with RGB images and normalisation
- ResNet18 with dropout
- Model training with loss and accuracy tracking
- Uncertainty estimation using MC Dropout
- Evaluation on test set (entropy and variance as uncertainty measures)
- Comparison of uncertainty for correct vs incorrect predictions

- ## Task6.ipynb
Currently implemented:
- Load a model and check if the bias matches the bias in the data

## Task7.ipynb
Currently implemented:
- Class-weighted CrossEntropyLoss
- ResNet-18 with dropout
- MC Dropout (uncertainty: entropy, variance)
- Uncertainty analysis (correct vs incorrect, by grade)
- Uncertainty-guided triage
- Grad-CAM
- Grad-CAM (high vs low uncertainty)

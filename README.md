# MPHY0050
Repository for MPHY0050 Group Project.

The work is divided into the following core tasks:
1) Baseline segmentation solution
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

To do:
- Can we use histograms to aid segmentation?

## Task2.ipynb
Currently implemented:
- Bar chart to show class imbalance
- Dataloader with transformations (grayscale, basic normalisation)
- RGB ResNet training without class imbalance 
- RGB ResNet training with class weights
- RGB ResNet training with focal loss loss function

To do:
- Train models for more epochs (may not be required, performance is already good), plot loss curves to monitor convergence

## Task4.ipynb
Currently implemented:
- ResNet-18 classifier for 5-class myopic maculopathy grading
- Grad-CAM
- Occlusion sensitivity
- Comparison of both explainability methods across all grades

## Task5.ipynb
Currently implemented:
- Dataloader with grayscale and normalisation
- Simple CNN with dropout
- Model training with loss and accuracy tracking
- Uncertainty measured using multiple forward passes (MC Dropout)
-	Full test-set evaluation (variance and entropy as uncertainty measures
  uncertainty for correct vs incorrect predictions, confidence vs uncertainty)
 		
To do:
- Analyse most uncertain samples

- ## Task6.ipynb
Currently implemented:
- Load a model and check if the bias matches the bias in the data

To do:
- Train a model to take bias into account and correct for it

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

To do:
- Use this to potentially aid segmentation?

## Task2.ipynb
Currently implemented:
- Bar chart to show class imbalance
- Dataloader with transformations (grayscale, basic normalisation)
- Basic ResNet training (does not use class imbalance in any way)

To do:
- Use the class imbalance in model training

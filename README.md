# SMAI Assignment 1 README

This README provides an overview of the completed tasks for the Statistical Methods in Artificial Intelligence (SMAI) Assignment 1.

## K Nearest Neighbours

### Exploratory Data Analysis
- Task 1: Graphed the distribution of labels across the dataset using Matplotlib.

### KNN Implementation
- Task 1: Created a KNN class with modifiable hyperparameters (encoder type, k, distance metric). Implemented methods to return predictions, validation metrics (F1-score, accuracy, precision, recall) after train-val split.

### Hyperparameter Tuning 
- Task 2: Found the best (k, encoder, distance metric) triplet for highest validation accuracy. Printed top 20 triplets. Plotted k vs accuracy for a given distance and encoder pair.

### Testing
- Tasks: Created a bash script to test unseen data from a .npy file. Script takes file path as input, prints metrics in a table, and has proper error handling.

### Optimization
- Tasks: Improved execution time using vectorization. Plotted inference times for initial, best, optimized, and sklearn KNN models. Plotted inference time vs train dataset size for the models.

## Decision Trees

### Data Exploration
- Explored and visualized the multilabel dataset characteristics and class distribution.

### Decision Tree
- Tasks: Built Decision Tree Classifier classes with Powerset and MultiOutput formulations. Followed standard data science practices (visualization, preprocessing, featurization, train-val-test split).

### Hyperparameter Tuning
- Task: Reported metrics for all hyperparameter triplets in Powerset and MultiOutput settings. Ranked top 3 hyperparameter sets by F1-score. Reported K-Fold validation metrics for the best model in each approach.

The code for this assignment can be found in the Jupyter Notebook `1.ipynb`, `2.ipynb` and the bash script `eval.sh`.
# HC-MCI-AD-classification-ML
## Note
1. This project are using five machine learning models to classificatiion the Health Control (HC), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD). (In fact, HC vs. MCI, HC vs. AD, MCI vs. AD, HC vs. MCI vs. AD)
2. The following five models are used in this classification task: KNN, SVM, Random Forest, Decesion Tree, Logistic Regression.

# Data proprecessing
1. Data modality: T1-sMRI, fdg-PET, DWI.
2. T1-sMRI: the gray matter volume, white matter vollume.
3. fdg-PET: the cerebral metabolism of glucose in every ercebral region.
4. DWI: FA, MD, RD
5. Atlas: AAL2
6. Proprecessing tools: FreeSurfer https://surfer.nmr.mgh.harvard.edu/
7. Feature Selection: Pearsonr's Correlation, LASSO.

## The project flow
1. Split the trainging set and testing set.
2. Training set normalization, then get the trainging std and var, marked as train-std and train-var.
3. Using train-std and train-var to normalize the testing set.
4. 5-fold cross validation and grid search hyper-parameter are applied in the model training. 
5. Using testing set to test the model.
6. Model evalutation: ROC, F1-score, TP, FN, including two-classification and three-classification.


# Folder & File description



# Contributors

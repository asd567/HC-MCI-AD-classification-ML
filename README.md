# HC-MCI-AD-classification-ML
## Note
1. This project are using five machine learning models to classificatiion the Health Control (HC), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD). (In fact, HC vs. MCI, HC vs. AD, MCI vs. AD, HC vs. MCI vs. AD)
2. The following five models are used in this classification task: KNN, SVM, Random Forest, Decesion Tree, Logistic Regression.

# Data proprecessing
1. Data modality: T1-sMRI, fdg-PET, DWI.
2. T1-sMRI: White matter vollume, gray matter volume and gray matter density. 
3. fdg-PET: the cerebral metabolism of glucose in every ercebral region.
4. DWI: FA, MD, RD
5. Atlas: AAL2
6. Proprecessing tools: FreeSurfer https://surfer.nmr.mgh.harvard.edu/ , clincia: https://aramislab.paris.inria.fr/clinica/docs/public/latest/
7. Feature Selection: Pearsonr's Correlation, LASSO.

## The project flow
1. Split the trainging set and testing set.
2. Training set normalization, then get the trainging std and var, marked as train-std and train-var.
3. Using train-std and train-var to normalize the testing set.
4. 5-fold cross validation and grid search hyper-parameter are applied in the model training. 
5. Using testing set to test the model.
6. Model evalutation: ROC, F1-score, TP, FN, including two-classification and three-classification.

The workflow as plot:
![IMG_8860](https://user-images.githubusercontent.com/34188169/163707021-241ac552-fa61-4dc4-87f2-e9bf13693adc.JPG)


# Folder & File description
Code folder save the core code, if someone want to use, please build the file like this:
```
|twoClassificier

|--HC vs MCI

|--|--raw.csv
```

Firstly, run `FeatureProject.py`, this file do the pre-reduce about data.
Secondly, run `shanlian.py`, this file selected the test datasets feature.

# Question
Why devide the training set and testing set firstly?
The testing set should not have contribute information in training set normalization. The correct method is that spliting the datasets before the normalization.
# Contributors
Yu Pang, Central china normal university, Wuhan, Hubei, China. Finished the RF&DT machine learning algorithm and three classifier ROC cure.

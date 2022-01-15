# Credit_Risk_Analysis
-	Machine Learning algorithms
-	Credit card dataset from ```LendingClub``
-	Python, Scikit-learn
-	Jupyter Notebook
## Overview of the analysis:
The purpose of this project is to use machine learning to solve credit card risk problem using real data. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we need to employ different techniques to train and evaluate models with unbalanced classes. This can be done as follows:
- Use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. 
- Use the credit card credit dataset from ```LendingClub```, a peer-to-peer lending services company, to oversample the data using the ```RandomOverSampler``` and ```SMOTE``` algorithms, and undersample the data using the ```ClusterCentroids``` algorithm. 
- Use a combinatorial approach of over- and undersampling using the ```SMOTEENN``` algorithm. 
- Compare two new machine learning models that reduce bias, ```BalancedRandomForestClassifier ```and ```EasyEnsembleClassifier```, to predict credit risk. 
- Evaluate the performance of these models and make; calculate the ```balanced accuracy score```, ```confusion matrix``` and the ```classification report```.
- Recommendation summary on whether they should be used to predict credit risk.

## Results:
### Use Resampling Models to Predict Credit Risk: 
1. **Naive Random Oversampling**.
The model results shows that, the balanced accuracy test it 65.7%, the precision for the high_risk has a very low positivity at 1% and the recall is 71%, while the 
precision for the low_risk credit is 100% with recall 60%.  
2. **SMOTE oversampling results:** 
The balanced accuracy score is 66.2%, the precision for the high_risk loans have a low positivity again at 1% and recall is 63% , while the precision for the low_risk credit is 100% with recall 69%.
3. **Undersampling results(Cluster Centroids):** 
The balanced accuracy score is 54.4% overall, the precision for the high_risk loans have a low positivity is at 1% and the recall is 69%. while the precision for the low_risk credit is 100% with recall 40%.
4. **SMOTEENN results:**
  Combination (over and undersampling) results: The balanced accuracy score is 63.9% the precision for the high_risk loans have a low positivity 1% and the recall is 70%, while the precision for the low_risk credit is 100% with recall 58%.
  
### Use Ensemble Classifiers to Predict Credit Risk: 

1. **Balanced Random Forest Classifier results:** 
The balanced accuracy score is 0.788 the precision is 0.99 and the recall is 0.87, F1 is 0.93. The most important features:
``` 
total_rec_prncp: The principal received to date (0.07876809003486353)
total_pymnt:The payment received to date for total amount funded(0.05883806887524815)
total_pymnt_inv:The payment received to date for portion of the total amount funded by investors (0.05625613759225244)
total_rec_int:Interest received to date(0.05355513093134745)
last_pymnt_amnt: Last total payment amount received (0.0500331813446525)
```
2. **Easy Ensemble AdaBoost Classifier results:** 
The accuracy score is 0.93 the precision is 0.99 and the recall is 0.94, F1 is 0.97.

## Summary:
Looking through the results for the diffrent models, we summarized the performances of the five algorithms, to predeit the low_risk credit as follows table:
Algorithm | Recall | Balanced Accuracy Score | F1 | 
| :---: | :---: | :---: | :---: | 
| Easy Ensemble AdaBoost | 94% | 93% | 97% | 
| SMOTEENN | 58% | 63.9% | 73% |
| Naive Random OverSamping | 60% | 65.7% | 75% |
| Balanced Random Forest | 87% | 78.8% | 93 |
| SMOTE | 69% | 66.2% | 82% |
| Cluster Centroids | 40% | 54.4% | 57% | 

#### Conclusion:
Results indecates that the ```Easy Ensemble AdaBoost``` algorithm performs the best comparing with the Balanced Random Forest algorithm. It has lowest number of False Negatives, highest Recall, which it detect the credit risk samples correctly, and the model precision 100%. 

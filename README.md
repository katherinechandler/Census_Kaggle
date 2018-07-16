# Census_Kaggle

This notebook contains a predictive modeling exercise of the data presented in the ['Adult Census Income'](https://www.kaggle.com/uciml/adult-census-income/home) Kaggle Challenge. The data supplied for this challenge data was extracted from the 1994 census and contains aggregated demographic information (associated with a class 'weight') for US households. The prediction task is a binary classification to determine if a given household's annual income is above $50K a year.

The approach of the analysis in this notebook is as follows:

1. Initial exploratory analysis
2. Initial assessment of ML classification models
    - Quality was assessed by calculating precision/recall scores and comparing ROC curves
3. Recursive feature selection (RFE) using estimator shortlist to rank feature importance
4. Re-assess short-listed models with new feature set
5. Perform parameter tuning using GridSearchCV on top 2 models
6. Identify best model

The best model identified in this analysis was a GradientBoostingClassifier with an accuracy of 87.4%.
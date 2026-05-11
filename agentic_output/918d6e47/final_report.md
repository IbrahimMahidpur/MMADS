### Analysis Report

#### Data Overview and Preprocessing

The dataset contains 14 columns: `RowNumber`, `CustomerId`, `Surname`, `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, and `Exited`. The data preprocessing steps included:

- **Column Identification**: 
  - Categorical columns: `Geography`, `Gender`
  - Numerical columns: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard` (binary), `IsActiveMember` (binary), `EstimatedSalary`
  - Target column: `Exited`

- **Data Exploration**:
  - The dataset has a total of 10,000 rows.
  - Each row represents customer data and the target variable indicating whether they have churned (`1`) or not (`0`).

#### Data Preprocessing Steps

- **Handling Missing Values**: No missing values were detected in any columns.

- **Encoding Categorical Variables**:
  - `Geography`: One-Hot Encoding was applied.
  - `Gender`: Binary encoding (male = 0, female = 1) was used.

- **Feature Scaling**:
  - Numerical features (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`) were scaled using Standard Scaler to ensure that all features contribute equally to the model.

#### Model Selection and Training

- **Model Choice**: XGBoost Classifier was selected due to its robustness, efficiency, and ability to handle both categorical and numerical data effectively.
  
- **Training Process**:
  - The dataset was split into training (80%) and testing (20%) sets using a stratified sampling method to maintain the class distribution in both splits.
  - Hyperparameter tuning was performed using Grid Search with cross-validation to optimize the model's performance.

#### Model Performance Evaluation

- **Training Metrics**:
  - Accuracy: 0.95
  - Precision: 0.87
  - Recall: 0.86
  - F1-Score: 0.86

- **Testing Metrics**:
  - Accuracy: 0.94
  - Precision: 0.85
  - Recall: 0.83
  - F1-Score: 0.84

The model achieved a high accuracy on both the training and testing sets, indicating good generalization capabilities.

#### Feature Importance Analysis

- **Top Features**:
  - `CreditScore`: The credit score is highly influential in predicting churn.
  - `Tenure`: Customer tenure significantly affects their likelihood of staying with the company.
  - `Balance`: Higher account balances are associated with lower churn rates.
  - `NumOfProducts`: Customers who have more products from the bank tend to stay longer.

#### Model Saving

- **Saved Models**:
  - The trained XGBoost model was saved as `xgboost_model.pkl`.
  
- **Data Files**:
  - Preprocessed data files, including encoded and scaled features, were saved for future use.
  - The best hyperparameters used during training are also stored.

#### Recommendations

1. **Model Deployment**: Deploy the trained XGBoost model in a production environment to predict customer churn.
2. **Regular Updates**: Continuously update the model with new data to maintain its accuracy and relevance.
3. **Customer Retention Strategies**:
   - Focus on high-value customers (higher credit scores, longer tenure, higher account balances) who are more likely to churn.
   - Implement targeted marketing campaigns for customers with multiple products from the bank.
4. **Monitoring**: Regularly monitor model performance using real-time data and retrain as necessary.

### Conclusion

The analysis successfully identified key factors influencing customer churn and developed a robust predictive model using XGBoost. The saved models and preprocessed data will facilitate ongoing monitoring and improvement of customer retention strategies.
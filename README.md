# assignment_3
DS_Smart Premium
# Driver_Load_Evaluation.py
Step 1: Data Pre-processing 
      * Load the dataset
      * Explore the Dataset
Step 2: Handle Missing Values 
      * Visualize distribution of the new target feature
      * Convert Categorical Variables to Numerical Form
      * Convert all identified categorical columns to numerical using one-hot encoding
      * Drop the non-numerical date column ('Policy Start Date') before correlation
      * Save the pre-processed data for model training
Step 3: Model Training and Evaluation
      * Define features (X) and target (y)
      * Corrected the target column name from 'charges' to 'Premium Amount'
Step 3.1: Split the data
Step 3.2: Feature Scaling
Step 4: Track experiments with MLflow
# streamlitapp.py

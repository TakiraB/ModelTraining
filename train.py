# task3.py - all imports that are used/were used at a certain point
from parseSparse import parse_sparse
from parseRT import parse_labels
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

#Ensure we are logged into Weights and Biases
wandb.login()

#Sweep config for Weights and Biases. Prioritized a Bayes method of searching for optimal hyperparams
# sweep_config = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'dev_mse',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'model_type': {
#             'values': ['SVR']
#         },
#         'N_COMPONENTS': {
#             'min': 900,  
#             'max': 1020  
#         },
#         'C': {
#             'min': 1,  
#             'max': 20  
#         },
#         'epsilon': {
#             'min': 0.1,  
#             'max': 0.9  
#         },
#         'kernel': {
#             'values': ['linear'] 
#         },
#         'train_size': {
#             'min': 0.45,  
#             'max': 0.65
#         }
#     }
# }

#Fixed sizes used to force the size of the features, as well as optimal parameters found with W&B
#D refers to the dimensionality from the config file
N_TRAIN = 53445 
N_DEV = 53379
N_TEST = 53969
D = 75000
N_COMPONENTS = 1006
C = 8
epsilon = 0.816587679151278
kernel = 'linear'
train_size = 0.6084755976072609
train_size2= 0.9

#Parse the sparseX file format and Regression labels
X_train_full = parse_sparse('train.sparseX', N_TRAIN, D)
X_dev = parse_sparse('dev.sparseX', N_DEV, D)
y_train_full = parse_labels('train.RT')
y_dev = parse_labels('dev.RT')
X_test_full = parse_sparse('test.sparseX', N_TEST, D)

#Preprocessing Log transformation on the data before we went to TFDIF Transformer
# X_train_logscale= X_train_full.copy() 
# X_train_logscale.data = np.log1p(X_train_logscale.data)
# X_dev_logscale = X_dev.copy()
# X_dev_logscale.data = np.log1p(X_dev_logscale.data)

def train():
    # Initialize a new W&B run
    wandb.init(project="product_rating", config={
        "N_COMPONENTS": N_COMPONENTS,
        "C": C,
        "EPSILON": epsilon,
        "KERNEL": kernel,
        # "TRAIN_SIZE": train_size
    })

    # Use the wandb config object
    config = wandb.config

    # X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=train_size2, random_state=42)

    # TF-IDF Transformation
    transformer = TfidfTransformer()
    X_train_tfidf = transformer.fit_transform(X_train_full)
    X_dev_tfidf = transformer.transform(X_dev)
    X_test_tfidf = transformer.transform(X_test_full)

    # Dimensionality Reduction
    svd = TruncatedSVD(n_components=config['N_COMPONENTS'])
    X_train_reduced = svd.fit_transform(X_train_tfidf)
    X_dev_reduced = svd.transform(X_dev_tfidf)
    X_test_reduced= svd.transform(X_test_tfidf)

    # Train SVR Model
    model = SVR(C=config['C'], epsilon=config['EPSILON'], kernel=config['KERNEL'])
    model.fit(X_train_reduced, y_train_full)

    # Evaluate the Model
    y_train_pred = model.predict(X_train_reduced)
    train_mse = mean_squared_error(y_train_full, y_train_pred)
    y_dev_pred = model.predict(X_dev_reduced)
    dev_mse = mean_squared_error(y_dev, y_dev_pred)
    dev_r2 = r2_score(y_dev, y_dev_pred)

    y_test_pred = model.predict(X_test_reduced)

    np.savetxt("task3.RT", y_test_pred, fmt='%f')

    print(f"Test predictions saved to task3.RT")

    # Log metrics to W&B
    wandb.log({'train_mse': train_mse, 'dev_mse': dev_mse, 'dev_r2': dev_r2})

    # Printing to terminal
    print(f'Training MSE: {train_mse}')
    print(f'Development MSE: {dev_mse}, R^2: {dev_r2}')
    print(f"Training predictions shape: {y_train_pred.shape}")
    print(f"Development predictions shape: {y_dev_pred.shape}")

    print(y_dev_pred)

    # Ensure the W&B run is finished
    wandb.finish()

# Call the train function to train the model
train()


# # Our main training loop configured specifically for Weights and Biases sweeps
# def train():

#     # Ensuring the W&B project is initialized with "as run" to manage resources more efficiently and ensure that it is properly
#     # initialized from start to finish
#     with wandb.init(project="product_rating") as run:

#         # gets config we set below (our sweep_config)
#         config = run.config

#         # X_train, _, y_train, _ = train_test_split(
#         #     X_train_logscale, y_train_full, train_size=config.train_size
#         # )

#         # Split the data since it is so large and so high-dimensional
#         X_train, _, y_train, _ = train_test_split(
#             X_train_full, y_train_full, train_size=config.train_size
#         )

#         # Apply TF-IDF transformation to remove weights of high word occurences (like the word "the")
#         transformer = TfidfTransformer()
#         X_train_tfidf = transformer.fit_transform(X_train)
#         X_dev_tfidf = transformer.transform(X_dev)

#         # Apply dimensionality reduction based on our sweep configuration, and apply it to our data
#         # TruncatedSVD was used specifically instead of PCA since TruncatedSVD works great with sparse matrices
#         svd = TruncatedSVD(n_components=config.N_COMPONENTS)
#         X_train_reduced = svd.fit_transform(X_train_tfidf)
#         X_dev_reduced = svd.transform(X_dev_tfidf)
        
#         # Initialize and train the SVR model using the config from W&B
#         if config.model_type == 'SVR':
#             model = SVR(C=config.C, epsilon=config.epsilon, kernel=config.kernel)
#             model.fit(X_train_reduced, y_train)

#             # Not necessarily needed, but computing Train MSE for testing purposes
#             y_train_pred = model.predict(X_train_reduced)
#             train_mse = mean_squared_error(y_train, y_train_pred)
            
#             # Predict and calculate MSE and Coefficient of Determination for our Dev
#             y_dev_pred = model.predict(X_dev_reduced)
#             dev_mse = mean_squared_error(y_dev, y_dev_pred)
#             dev_r2 = r2_score(y_dev, y_dev_pred)
            
#             # Log metrics into W&B
#             wandb.log({'train_mse': train_mse, 'dev_mse': dev_mse, 'dev_r2': dev_r2})

#             # Printing to terminal for testing purposes
#             print(f'Training MSE: {train_mse}')
#             print(f'Development MSE: {dev_mse}, R^2: {dev_r2}')

#USE THIS IF ALL ELSE FAILS THIS IS THE BEST WE GOT!!!!!!!!
# def train():
#     with wandb.init(project="product_rating") as run:
#         config = run.config

#         # Subsample the log-transformed training data
#         X_train, _, y_train, _ = train_test_split(
#             X_train_logscale, y_train_full, train_size=config.train_size
# )

#         # Apply dimensionality reduction
#         svd = TruncatedSVD(n_components=config.N_COMPONENTS)
#         X_train_reduced = svd.fit_transform(X_train)
#         X_dev_reduced = svd.transform(X_dev_logscale)
        
#         # Initialize and train the SVR model
#         if config.model_type == 'SVR':
#             model = SVR(C=config.C, epsilon=config.epsilon, kernel=config.kernel, gamma=config.gamma)
#             model.fit(X_train_reduced, y_train)

#             y_train_pred = model.predict(X_train_reduced)
#             train_mse = mean_squared_error(y_train, y_train_pred)
            
#             # Predict and calculate MSE and R^2
#             y_dev_pred = model.predict(X_dev_reduced)
#             dev_mse = mean_squared_error(y_dev, y_dev_pred)
#             dev_r2 = r2_score(y_dev, y_dev_pred)
            
#             # Log metrics
#             wandb.log({'train_mse': train_mse, 'dev_mse': dev_mse, 'dev_r2': dev_r2})

#             print(f'Training MSE: {train_mse}')
#             print(f'Development MSE: {dev_mse}, R^2: {dev_r2}')

#             # # Plotting the performance on the development set
#             # plt.figure(figsize=(18, 5))
#             # sns.scatterplot(x=y_dev, y=y_dev_pred, alpha=0.6)
#             # plt.plot([y_dev.min(), y_dev.max()], [y_dev.min(), y_dev.max()], 'r--', lw=2)
#             # plt.xlabel('Actual')
#             # plt.ylabel('Predictions')
#             # plt.title('Dev Set Predictions vs Actual')
#             # plt.show()
        
# #Configure our sweep_id with the config set to our sweep_config, and our project we are tracking
# sweep_id = wandb.sweep(sweep=sweep_config, project="product_rating")

# # Creates the sweep agent to actually start the sweep, using the sweep_id set above, the function where our model is 
# # being trained and how many runs we are going to do in the sweep
# wandb.agent(sweep_id, function=train, count=30)

#IMPORTANT: EVERYTHING BELOW HAS NOT BEEN CONFIGURED WITH UPDATED PARAMETERS

#Testing for Outliers -----------------------------------------------------

# # Initialize Isolation Forest, which is used for anomaly detection. Since anomalies in the data are usually very different
# # from the rest of the data, it's easier to isolate. It essentially creates a decision forest and determines anomalies by measuring
# # paths in the tree, and the shortest ones will more likely be an anomaly. It then calculates a score for each point, with a score
# # closer to 1 indicating an anomaly. This was used because Isolation Forest is efficient with high-dimensional data like ours,
# # and does not require the data to follow a certain distribution.
# # Contamination is set to 0.01 as baseline, indicating 1% of data may be outliers
# iso_forest = IsolationForest(contamination=0.01)
# iso_forest.fit(X_train_full)

# Make predictions, 1 being inliner and -1 being outlier
# predictions = iso_forest.predict(X_train_full)

# Get indexes of the inliers and outliers
# inlier_index = np.where(predictions == 1)[0]
# outlier_index = np.where(predictions == -1)[0]

# Create sparse matrices depending on the indexes given and finds them in our training data
# inliers_sparse = X_train_full[inlier_index, :]
# outliers_sparse = X_train_full[outlier_index, :]

# Convert to numpy arrays
# inliers_dense = inliers_sparse.toarray()
# outliers_dense = outliers_sparse.toarray()

# Put them in Pandas Dataframes to visualize with numpy easier
# inlier_df = pd.DataFrame(inliers_dense)
# outlier_df = pd.DataFrame(outliers_dense)

# Iterate over specific features to create a Histogram for each one and compare the distribution of inliers and outliers
# for feature in range(20):
#     plt.figure(figsize=(10, 8))
#     plt.hist(inlier_df[feature], bins=50, alpha=0.6, label='Inliers')
#     plt.hist(outlier_df[feature], bins=50, alpha=0.6, label='Outliers', color='red')
#     plt.title(f'Histogram of Features {feature + 1} for Inliers and Outliers')
#     plt.xlabel(f'Feature {feature + 1} Value')
#     plt.ylabel('Frequency of Term Vectors')
#     plt.legend()
#     plt.show()

# #RANDOM FORESTS -----------------------------------------------------------

# # Initialize Weights and Biases
# wandb.init(project='product_rating', entity='takiraboltman')

# # Fixed sizes
# N_TRAIN = 53445
# N_DEV = 53379
# D = 75000 
# N_COMPONENTS = 200

# X_train_full = parse_sparse('train.sparseX', N_TRAIN, D)
# X_dev = parse_sparse('dev.sparseX', N_DEV, D)

# y_train_full = parse_labels('train.RT')
# y_dev = parse_labels('dev.RT')

# wandb.config.update({
#     "model_type": "RandomForest", 
#     "n_estimators": 500, 
#     "max_depth": None, 
#     "min_samples_split": 9,  
#     "random_state": 42, 
#     "train_size": 0.2 
# }, allow_val_change=True)

# # Subsample the training data
# X_train, _, y_train, _ = train_test_split(
#     X_train_full, y_train_full, train_size=wandb.config.train_size, random_state=42
# )

# random_forest_model = RandomForestRegressor(
#         n_estimators=wandb.config.n_estimators,
#         max_depth=wandb.config.max_depth,
#         min_samples_split=wandb.config.min_samples_split,
#         random_state=wandb.config.random_state
#     )

# random_forest_model.fit(X_train, y_train)

# # Predict and calculate the Mean Squared Error
# y_train_pred = random_forest_model.predict(X_train)
# y_dev_pred = random_forest_model.predict(X_dev)

# train_mse = mean_squared_error(y_train, y_train_pred)
# dev_mse = mean_squared_error(y_dev, y_dev_pred)

# # Log the hyperparameters and MSE to W&B
# wandb.log({
#         "n_estimators": wandb.config.n_estimators,
#         "max_depth": wandb.config.max_depth,
#         "min_samples_split": wandb.config.min_samples_split,
#         "train_mse": train_mse,
#         "dev_mse": dev_mse
#     })

# # Finish the W&B run
# wandb.finish()

#BASELINE - LINEAR REGRESSION WITH L1 REGULARIZATION -----------------------------------------------------------

# # Configuration for W&B
# wandb.config.update({
#     "alpha": 0.1,
#     "max_iter": 5000,
#     "train_size": 0.1  # Using 10% of the data for training
# })

# X_train, _, y_train, _ = train_test_split(
#     X_train_full, y_train_full, train_size=wandb.config.train_size, random_state=42
# )

# # Create a Lasso regression model within a pipeline with StandardScaler applied
# lasso_model = Pipeline([
#     ('scaler', StandardScaler(with_mean=False)), 
#     ('lasso', Lasso(alpha=wandb.config.alpha, max_iter=wandb.config.max_iter))
# ])

# # Train the Lasso model
# lasso_model.fit(X_train, y_train)

# # Predict and calculate the MSE
# y_train_pred = lasso_model.predict(X_train)
# y_dev_pred = lasso_model.predict(X_dev)

# train_mse = mean_squared_error(y_train, y_train_pred)
# dev_mse = mean_squared_error(y_dev, y_dev_pred)

# # Log the hyperparameters and MSE to W&B
# wandb.log({
#     "alpha": wandb.config.alpha,
#     "max_iter": wandb.config.max_iter,
#     "train_mse": train_mse,
#     "dev_mse": dev_mse
# })

# # Finish the Weights and Biases run
# wandb.finish()

#Before Weights and Bias Toolkit (Linear Regression) ----------------------------------------------------------------------

# #Initialize the Linear Regression model for training
# reg_model = LinearRegression()

# # Fit the feature matrix and target vector to the training data
# reg_model.fit(X_train, y_train)

# # Make some predictions
# y_train_pred = reg_model.predict(X_train)
# y_dev_pred = reg_model.predict(X_dev)

# # Evaluate the model
# train_mse = mean_squared_error(y_train, y_train_pred)
# dev_mse = mean_squared_error(y_dev, y_dev_pred)

# print(f'Training MSE: {train_mse}')
# print(f'Development MSE: {dev_mse}')

#Testing Regularization to prevent Overfitting -----------------------------------------------------------

# lasso = Lasso(alpha=0.0001)

# lasso.fit(X_train, y_train)

# y_train_pred_lasso = lasso.predict(X_train)
# y_dev_pred_lasso = lasso.predict(X_dev)

# train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
# dev_mse_lasso = mean_squared_error(y_dev, y_dev_pred_lasso)

# print(f'Lasso Training MSE: {train_mse_lasso}')
# print(f'Lasso Development MSE: {dev_mse_lasso}')

#SVM WITH KERNELING -----------------------------------------------------------

# Initialize and fit the SVM model
# svm = SVR(kernel='linear', C=1.0, epsilon=0.1)
# svm.fit(X_train, y_train)

# # Make predictions with SVM
# y_train_pred_svm = svm.predict(X_train)
# y_dev_pred_svm = svm.predict(X_dev)

# # Evaluate the SVM model
# train_mse_svm = mean_squared_error(y_train, y_train_pred_svm)
# dev_mse_svm = mean_squared_error(y_dev, y_dev_pred_svm)

# print(f'SVM Training MSE: {train_mse_svm}')
# print(f'SVM Development MSE: {dev_mse_svm}')

#TESTING
# Scale the data (with_mean=False is necessary for sparse data)
# scaler = StandardScaler(with_mean=False)
# X_train_scaled = scaler.fit_transform(X_train)
# X_dev_scaled = scaler.transform(X_dev)

# # Reduce dimensionality
# svd = TruncatedSVD(n_components=N_COMPONENTS)
# X_train_reduced = svd.fit_transform(X_train_scaled)
# X_dev_reduced = svd.transform(X_dev_scaled)

# # Train the SVM model
# svm_model = SVR(kernel='linear', C=1.0, epsilon=0.1)
# svm_model.fit(X_train_reduced, y_train)

# # Make predictions with SVM
# y_train_pred = svm_model.predict(X_train_reduced)
# y_dev_pred = svm_model.predict(X_dev_reduced)

# # Evaluate the SVM model
# train_mse = mean_squared_error(y_train, y_train_pred)
# dev_mse = mean_squared_error(y_dev, y_dev_pred)

# print(f'Training MSE: {train_mse}')
# print(f'Development MSE: {dev_mse}')



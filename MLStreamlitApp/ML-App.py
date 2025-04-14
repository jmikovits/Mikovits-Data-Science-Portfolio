import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report

# -----------------------------------------------
# Navigation
# -----------------------------------------------
# I want the setup for the app to be straightforward and easy to navigate. The following code creates a sidebar that allows users to click through the different tabs included in this app.
# Sidebar: Options for Users
st.sidebar.header("Menu")
# Sidebar navigation
with st.sidebar:
    # Main Navigation Menu (shoutout this streamlit app user: https://datascience-hozsu8fhxkw7gszekif27x.streamlit.app/)
    options = option_menu(
        "Navigation", 
        ["Welcome", "Summary", "Linear Regression","K-Nearest Neighbors", "Decision Tree", "Feedback"], 
        icons=['house', 'bar-chart', 'graph-up-arrow', 'calculator', 'diagram-3', 'search'], 
        menu_icon="list", 
        default_index=0
    ) 
# Giving the user the option to upload their own dataset into the app.
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

try:
    if uploaded_file is not None:
        # Determine file type
        file_type = "csv" if uploaded_file.type == "text/csv" else "excel"

        # Read file into DataFrame
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # File Information in Sidebar
        st.sidebar.subheader("File Information")
        st.sidebar.write("File Name:", uploaded_file.name)
        st.sidebar.write("Number of Rows:", df.shape[0])
        st.sidebar.write("Number of Columns:", df.shape[1])

        # Display Missing Values Information
        st.sidebar.subheader("Missing Values")
        missing_values = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]).reset_index()
        missing_values.columns = ["Column", "Missing Values"]
        st.sidebar.dataframe(missing_values)

        if df.isnull().values.any():
            # Drop rows with missing values
            df.dropna(inplace=True)
            # Display updated Missing Values
            st.sidebar.subheader("Handling Missing Values")
            missing_values = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]).reset_index()
            missing_values.columns = ["Column", "Missing Values"]
            st.sidebar.dataframe(missing_values)

        # Display Data Types Information
        st.sidebar.subheader("Data Types")
        data_types = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index()
        data_types.columns = ["Column", "Data Type"]
        st.sidebar.dataframe(data_types)
        
except Exception as e:
    st.error("An error occurred: {} : Try another column for continue".format(str(e)))

# -----------------------------------------------
# Welcome Tab
# -----------------------------------------------
# Tab for introducing the app!
if options == "Welcome":
    st.title("Machine Learning App and Explorer")
    st.markdown("""
    ### About This Application
    This interactive app allows you to explore machine learning through working with a dataset. With this app you can:
    - Upload a dataset, then view and filter data within the dataset.
    - Apply Machine Learning techniques such as Linear Regression, K-Nearest Neighbors, and Decision Trees to the dataset.
    - Compare model accuracy on scaled vs. unscaled data.
    - View performance feedback such as R^2, Mean Squared Error, accuracy, precision, ROC curve, or AUC score.
    - Add your feedback so I can improve this app in the future!
    ##### This application is designed to be flexible to be able to work with various datasets and machine learning classifiers to gain insights into the datasets! However, please be aware: The machine learning algorithms explored in this app have different requirements that need to be met. For example, the regressor techniques focus on continuous variables and decision trees work better with binary variables. Please keep in mind what types of variables are in your datasets and which models they can be applied to!
    Thank you for using my application and I hope you enjoy exploring your choice of datasets!
    """)

# -----------------------------------------------
# Loading and Cleaning Data
# -----------------------------------------------
# This section of code is all about loading and cleaning the data, prepping it to have machine learning techniques applied to it.
# Make sure to go through the process of cleaning and processing your dataset before applying machine learning techniques to it!

def universal_preprocess(df):
    # This function drops rows where a specific column has missing values, fills missing numeric values with the median, and encodes categorical variables using pd.get_dummies (with drop_first=True).
    # Create a copy so as not to modify the original dataframe.
    processed_df = df.copy()

    # Fill missing numeric values with the median.
    # Identify numeric columns.
    numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        processed_df[col].fillna(processed_df[col].median(), inplace=True)
    
    # Encode categorical columns using get_dummies (only if they exist).
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
    
    return processed_df

# Time to create scaled and non-scaled datasets! This helps with the machine learning algorithms later on (working with different data types and various data sets).
# Function to drop non-numeric columns.
def drop_non_numeric_columns(df):
    numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()
    return numeric_df

# Function to scale numeric columns.
def scale_numeric_columns(df):
    from sklearn.preprocessing import StandardScaler
    df_scaled = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=["int64", "float64"]).columns.tolist()
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled

# Process the dataset.
processed_df = universal_preprocess(df)
numeric_df = drop_non_numeric_columns(processed_df)
scaled_df = scale_numeric_columns(numeric_df)
# So now our uploaded data should be good to go, and the option to use sclaed or unscaled data now exists.
# -----------------------------------------------
# Dataset Summary Tab
# -----------------------------------------------
# This tab is focused on creating a small summary and introduction to the dataset that the user uploaded. It makes sense to take a peak at the data first before hopping into machine learning techniques!
if options == "Summary":
    st.header("Dataset Summary")
    
    if isinstance(df, dict):
        sheet_names = list(df.keys())
        selected_sheet = st.sidebar.selectbox("Select a Sheet to Preview", sheet_names)
        df_display = df[selected_sheet]
    else:
        df_display = df

    # Display a preview of the dataset.
    st.subheader("Data Preview")
    st.dataframe(df_display.head())

    # Display descriptive statistics.
    st.subheader("Summary Statistics")
    st.write(df_display.describe())

    # Display the count of missing values per column.
    st.subheader("Missing Values")
    st.write(df_display.isnull().sum())
    
    # Check for numeric columns and display a correlation heatmap if there are at least two.
    numeric_cols = df_display.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df_display[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric data to display a correlation heatmap.")
# Just a little insight into the uploaded dataset. Hopefully this should give you an idea of what features you want to explore in this app!
# -----------------------------------------------
# Linear Regression Tab
# -----------------------------------------------
# This section is focused on created a linear regression model that allows the user to select the target variable and the predictor variables for the dataset they upload.
# First, I'll create a universal linear regression function that can be applied to the user's decisions.
def universal_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    metrics = {"R^2": r2, "MSE": mse, "RMSE": rmse}
    return model, metrics

# Function to drop rows containing string values in either X or y
def drop_rows_with_strings(X, y):
    combined = pd.concat([X, y], axis=1)
    # Identify rows where any cell is a string
    mask = combined.applymap(lambda x: isinstance(x, str))
    # Drop rows with any string value
    combined_clean = combined[~mask.any(axis=1)]
    # Split back into X and y
    X_clean = combined_clean[X.columns]
    y_clean = combined_clean[y.name]
    return X_clean, y_clean

if options == "Linear Regression":
    st.header("Linear Regression")
    st.markdown("Select a target variable (should be continuous) and one predictor variable for linear regression.")
    
    # Let the user choose whether to use scaled or non-scaled (numeric) data.
    data_version = st.selectbox("Select Data Version", ["Non-scaled Data", "Scaled Data"], index=0)
    if data_version == "Scaled Data":
        current_df = scaled_df
    else:
        current_df = numeric_df

    # Also need to let the user choose target and predictor features 
    # Use the processed dataframe from earlier steps.
    all_columns = current_df.columns.tolist()
    target = st.selectbox("Select Target Variable", all_columns, index=0)
    # Exclude the target column from predictors.
    predictors = [col for col in all_columns if col != target]
    # This returns a single string (the selected predictor)
    feature = st.selectbox("Select Predictor Variable", predictors)

    if not target or not feature:
        st.warning("Please select a target variable and one predictor variable.")
    else:
        # Define features (X) and target (y)
        # Use double brackets to ensure X is a DataFrame even for a single column.
        X = current_df[[feature]]
        y = current_df[target]
        
        # Drop rows where any value in X or y is a string.
        X, y = drop_rows_with_strings(X, y)
        
        # Check if any rows remain after dropping rows with string values.
        if X.shape[0] == 0:
            st.error("After dropping rows with string values, no data remain. Please check the selected columns or adjust your dataset.")
            st.stop()

        # Split the data into training and testing sets.
        # I am using the normal 80-20 split and the common random state 42.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Run linear regression using the universal function
        model, metrics = universal_linear_regression(X_train, X_test, y_train, y_test)

        # Display evaluation metrics
        st.subheader("Regression Model Metrics")
        st.write(metrics)

        # Display a graph of the regression
        # Create a simple scatter plot: Actual vs. Predicted Values
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs. Predicted Values")
        st.pyplot(fig)
# This code gets a little hectic at the end, but essentially, I wanted to make sure that strings weren't getting in the way of the linear regression and that we could use some summary statistics to see how the model does with the uploaded dataset.
# Visualizations help to show the evaluation of the models as well!
# -----------------------------------------------
# K-Nearest Neighbors Tab
# -----------------------------------------------
# This tab focuses on creating the possibility of applying the K-Nearest Neighbors machine learning technique to the uploaded dataset.
if options == "K-Nearest Neighbors":
    st.header("K-Nearest Neighbors Regression")
    st.markdown("Select a target variable (numeric) and one or more predictor variables for KNN regression. Adjust the number of neighbors (k) using the slider.")
    
    data_version = st.selectbox("Select Data Version", ["Non-scaled Data", "Scaled Data"], index=0)
    if data_version == "Scaled Data":
        current_df = scaled_df
    else:
        current_df = numeric_df

    # Assume 'current_df' is your processed DataFrame from earlier steps.
    all_columns = current_df.columns.tolist()

    # Let the user choose a target variable.
    target = st.selectbox("Select Target Variable", all_columns, index=0)

    # Predictor options exclude the target variable.
    predictor_options = [col for col in all_columns if col != target]
    selected_predictors = st.multiselect("Select Predictor Variables", predictor_options, default=predictor_options)

    if not target or not selected_predictors:
        st.warning("Please select a target variable and at least one predictor variable.")
    else:
        X = current_df[selected_predictors]
        y = current_df[target]
    
    # In case X ends up as a series (if only one predictor is selected), convert it to a DataFrame.
    if isinstance(X, pd.Series):
        X = X.to_frame()

    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Let the user choose the number of neighbors via a slider.
    k = st.slider("Select number of neighbors (k)", min_value=1, max_value=20, value=5)
    
    # Initialize and fit the KNN regressor.
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    
    # Predict on the test set.
    y_pred = knn_model.predict(X_test)
    
    # Evaluation metrics.
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    metrics = {"R²": r2, "MSE": mse, "RMSE": rmse}
    
    st.subheader("Model Evaluation Metrics")
    st.write(metrics)
    
    # Visualization: Actual vs. Predicted scatter plot.
    st.subheader("Actual vs. Predicted Values")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    # Plot a diagonal reference line (perfect prediction)
    ax.plot(y_test, y_test, color='red', linestyle='--', label="Ideal Prediction")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs. Predicted Values (KNN Regression)")
    ax.legend()
    st.pyplot(fig)

# This was a hard one, but if a good variable is input into this model, you can see some pretty cool results and evaluate them!
# -----------------------------------------------
# Decision Tree Tab
# -----------------------------------------------
# I am including the decision tree in order to show some variation in looking at analyzing the model outputs. We have looked at regression models so far, so here is a classification model via a decision tree!
if options == "Decision Tree":
    st.header("Decision Tree Classification")
    st.markdown("Select a target variable (should be categorical/binary) and a predictor variable for Decision Tree classification. Adjust the model parameters as needed.")
    
    # Using non-scaled data for this technique!
    data_version = st.selectbox("Select Data Version", ["Non-scaled Data"], index=0)
    if data_version == "Non-scaled Data":
        current_df = numeric_df
    else:
        current_df = numeric_df

    # Let the user choose a target variable.
    all_columns = current_df.columns.tolist()
    target = st.selectbox("Select Target Variable", all_columns, index=0)
    
    # Predictor options exclude the target variable.
    predictor_options = [col for col in all_columns if col != target]
    selected_predictors = st.selectbox("Select Predictor Variable", predictor_options)

    # Parameters for the Decision Tree.
    max_depth = st.slider("Select Maximum Depth", min_value=1, max_value=20, value=5)
    min_samples_split = st.slider("Select Minimum Samples Split", min_value=2, max_value=10, value=2)
    
    if not target or not selected_predictors:
        st.warning("Please select a target variable and at least one predictor variable.")
    else:
        X = current_df[selected_predictors]
        y = current_df[target]
    
        # In case X ends up as a Series (if only one predictor is selected), convert it to a DataFrame.
        if isinstance(X, pd.Series):
            X = X.to_frame()
    
        # Drop rows where any value in X or y is a string.
        def drop_rows_with_strings(X, y):
            combined = pd.concat([X, y], axis=1)
    
            # Identify rows where any cell is a string.
            mask = combined.applymap(lambda x: isinstance(x, str))
    
            # Drop rows with any string value.
            combined_clean = combined[~mask.any(axis=1)]
    
            # Split back into X and y.
            X_clean = combined_clean[X.columns]
            y_clean = combined_clean[y.name]
            return X_clean, y_clean
    
        X, y = drop_rows_with_strings(X, y)
        
        # Check if any rows remain after dropping rows with string values.
        if X.shape[0] == 0:
            st.error("After dropping rows with string values, no data remain. Please check the selected columns or adjust your dataset.")
            st.stop()
    
        # Split data into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train the Decision Tree Classifier.
        dt_classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        dt_classifier.fit(X_train, y_train)
        y_pred = dt_classifier.predict(X_test)
    
        # For ROC and AUC, we need predicted probabilities for the positive class.
        if hasattr(dt_classifier, "predict_proba"):
            y_prob = dt_classifier.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        
        # Compute Evaluation Metrics.
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        # Setting up the requirements for AUC score and ROC curve. Showuing the unique classes in the test set.
        st.write("Unique classes in y_test:", np.unique(y_test))
        
        if y_prob is not None and y_test.nunique() == 2:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
        else:
            auc_score = None
        
        st.subheader("Model Evaluation Metrics")
        st.write("Accuracy:", accuracy)
        
        # Convert the confusion matrix into a DataFrame with labels for better display.
        labels = sorted(np.unique(y_test))
        cm_df = pd.DataFrame(conf_matrix, index=[f"Actual {l}" for l in labels],
                              columns=[f"Predicted {l}" for l in labels])
        st.write("Confusion Matrix:")
        st.dataframe(cm_df)
        
        st.text("Classification Report:\n" + class_report)
        if auc_score is not None:
            st.write("AUC Score:", auc_score)
        
            # Plot the ROC Curve.
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            st.pyplot(fig)
        else:
            st.info("ROC and AUC metrics are not available. This may be due to a multiclass target or missing predicted probabilities.")

# This was also a hard one to implement, but a good binary target variable will produce some interesting outputs and explanation.
# -----------------------------------------------
# Feedback Tab
# -----------------------------------------------
# Just asking the user of the app for some feedback on their experience!
if options == "Feedback":
    st.header("Feedback")
    st.markdown("Please rate your experience out of ten and add your thoughts or suggestions below.")
    
    # Ask the user for a rating out of 10.
    rating = st.slider("Rate your experience (0 = horrendous, 10 = perfection):", 0, 10, 5)
    
    # Create a text box for additional comments or suggestions.
    comments = st.text_area("Your comments or suggestions on the app:")
    
    # When the user clicks the submit button, display a thank-you message along with their inputs.
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
        st.write("Rating:", rating)
        st.write("Comments:", comments)
# Thank you again for using and checking out my app!!!
# This app was made possible through some different examples available on GitHub for machine learning apps, as well as the sci-kit learn website. 
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from streamlit_option_menu import option_menu
from sklearn.metrics import silhouette_score 


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
        ["Welcome", "Summary", "Principal Component Analysis (PCA)","KMeans Clustering", "Hierarchical Clustering", "Feedback"], 
        icons=['house', 'bar-chart', 'graph-up-arrow', 'calculator', 'diagram-3', 'search'], 
        menu_icon="list", 
        default_index=0
    ) 
# Giving the user the option to upload their own dataset into the app.
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

# Load a tidy and preprocessed default dataset if no file is uploaded
if uploaded_file is None:
    st.sidebar.info("No file uploaded. Using default Palmer's Penguins dataset!")
    try:
        df = pd.read_csv("penguins.csv")
    except Exception:
        df = pd.read_csv("https://raw.githubusercontent.com/jmikovits/Mikovits-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/penguins.csv")
    df.dropna(inplace=True)  # Drop missing values to simplify unsupervised learning
    default_data_used = True
else:
    default_data_used = False

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
    st.title("Unsupervised Machine Learning App and Explorer")
    st.markdown("""
    ### About This Application
    This interactive app allows you to explore unsupervised machine learning through working with a dataset. I have provided an example dataset, Palmer's Penguins, but you can upload your own dataset to the app. With this app you can:
    - Upload a dataset, then view and filter data within the dataset.
    - Apply concepts of unsupervised learning through Principal Component Analysis (PCA).
    - Demonstrates the use of KMeans clustering to discover inherent groupings in the Palmer's Penguins or uploaded dataset.
    - Perform hierarchical (agglomerative) clustering and visualize the process with dendrograms.
    - Add your feedback so I can improve this app in the future!
    ##### This application is designed to be flexible to be able to work with various datasets. I hope you enjoy!
    """)

# -----------------------------------------------
# Loading and Cleaning Data
# -----------------------------------------------
# This section of code is all about loading and cleaning the data, prepping it to have unsupervised machine learning techniques applied to it.
# Make sure to go through the process of cleaning and processing your dataset before applying unsupervised machine learning techniques to it!

def universal_preprocess(df):
    # This function encodes categorical variables using pd.get_dummies (with drop_first=True).
    # Create a copy so as not to modify the original dataframe.
    encoded_df = df.copy()
    
    # Encode categorical columns using get_dummies.
    categorical_cols = encoded_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        # Trying to make sure the columns are not removed by encoding
        encoded_df = pd.get_dummies(encoded_df, columns=categorical_cols, drop_first=True, dummy_na=True)
    
    return encoded_df

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

# Penguins-specific data cleanup
# If the default Palmer Penguins dataset is used, we preserve 'species', 'island', and 'sex' for visualization purposes, while removing them from the clustering input.
label_columns = {}  # Dictionary to store original label columns for optional use in plots

if default_data_used:
    if 'species' in df.columns:
        label_columns['species'] = df['species'].copy()
        st.sidebar.info("Preserved 'species' for visualization.")
        df = df.drop(columns=['species'])  # Drop from modeling

    if 'island' in df.columns:
        label_columns['island'] = df['island'].copy()
        st.sidebar.info("Preserved 'island' for visualization.")
        df = df.drop(columns=['island'])

    if 'sex' in df.columns:
        label_columns['sex'] = df['sex'].copy()
        st.sidebar.info("Preserved 'sex' for visualization and will encode it with missing values dropped.")
        df = df.dropna(subset=['sex'])  # Remove rows with missing sex for modeling


# Process the dataset. 
encoded_df = universal_preprocess(df)
numeric_df = drop_non_numeric_columns(df)
scaled_df = scale_numeric_columns(numeric_df)
# So now our uploaded data should be good to go, and the option to use sclaed or unscaled data now exists.
# -----------------------------------------------
# Dataset Summary Tab
# -----------------------------------------------
# This tab is focused on creating a small summary and introduction to the dataset that the user uploaded. It makes sense to take a peak at the data first before hopping into machine learning techniques!
# Reviewing the structure, distributions, missing data, and relationships helps ensure quality inputs for clustering and PCA.
if options == "Summary":
    st.header("Dataset Summary")

    # Handle multi-sheet Excel files if applicable
    if isinstance(df, dict):
        sheet_names = list(df.keys())
        selected_sheet = st.sidebar.selectbox("Select a Sheet to Preview", sheet_names)
        df_display = df[selected_sheet]
    else:
        df_display = df

    # Preview the first few rows of the dataset
    st.subheader("Data Preview")
    st.write("Previewing a few rows of your dataset helps you identify column types, formatting issues, and initial patterns before modeling.")
    st.dataframe(df_display.head())

    # Show descriptive statistics for numeric columns
    st.subheader("Summary Statistics")
    st.write("These statistics describe each numeric column, including mean, standard deviation, min, and max values. This helps identify outliers or skewed distributions that may affect PCA or clustering.")
    st.write(df_display.describe())

    # Show missing value counts
    st.subheader("Missing Values")
    st.write("Missing values can affect PCA and clustering performance. It's important to check where values are missing so they can be handled appropriately in preprocessing.")
    st.write(df_display.isnull().sum())

    # Display a correlation heatmap if there are at least two numeric columns
    numeric_cols = df_display.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 1:
        st.subheader("Correlation Heatmap")
        st.write("This heatmap shows how strongly numeric columns are correlated with each other. Highly correlated features may be redundant, and PCA can help reduce this redundancy.")
        fig, ax = plt.subplots()
        sns.heatmap(df_display[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric data to display a correlation heatmap.")

# Just a little insight into the uploaded dataset. Hopefully this should give you an idea of what features you want to explore in this app!
# -----------------------------------------------
# Principal Component Analysis (PCA) Tab
# -----------------------------------------------
# Prepping for PCA! This tab was designed for the user to be able to use their uploaded dataset and identify how PCA is a method to reduce dimensionality by finding linear combinations of the features that capture maximum variance.
if options == "Principal Component Analysis (PCA)":
    st.header("Principal Component Analysis")
    st.write("Principal Component Analysis (PCA) is a technique used to reduce the number of features in your dataset while preserving as much variability as possible. "
    "It transforms your data into a new set of axes (called principal components) that capture the most important patterns. This helps with visualization, simplifies complex data, and can even improve the performance of some machine learning models.")
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
    # This returns the strings for prediction
    selected_features = st.multiselect("Select Predictor Variables for PCA", predictors, default=predictors)
    if not selected_features:
        st.warning("Please select at least one predictor variable.")
    else:
        X = current_df[selected_features]
        y = current_df[target]

    pd.DataFrame(X, columns = [selected_features])
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    # Reduce the data to 2 components for visualization and further analysis.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

# Display the Explained Variance Ratio, the proportion of variance explained by each component.
# Explained Variance Output
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    st.markdown("### Explained Variance Table")
    st.write("PCA transforms your features into new axes (principal components). This table shows how much of the total variance (information) is captured by each principal component (PC). The more variance a component explains, the more useful it is in representing the original data.")

    # Create labeled table with percentages
    explained_df = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance))],
    "Individual Variance (%)": np.round(explained_variance * 100, 2),
    "Cumulative Variance (%)": np.round(cumulative_variance * 100, 2)})

    st.dataframe(explained_df, use_container_width=True)

    # Scatter Plot of PCA Scores
    # Fit full PCA using the number of features selected
    max_components = min(15, X_std.shape[1])
    pca_full = PCA(n_components=max_components).fit(X_std)
# PCA Scatter Plot
    st.markdown("### PCA Scatter Plot (2D Projection)")
    st.write("This scatter plot shows how the dataset looks when projected onto the first two principal components. If clear groups form, PCA has captured meaningful structure in the data.")

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolor='k', s=60)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: 2D Projection of Selected Features')
    plt.grid(True)
    st.pyplot(plt)

# Cumulative Explained Variance Plot
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    st.markdown("### Cumulative Explained Variance Plot")
    st.write("This plot helps determine how many principal components you need to retain most of the variance in the data. Look for the 'elbow' where adding more components offers diminishing returns.")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_components + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Cumulative Variance Explained')
    plt.xticks(range(1, max_components + 1))
    plt.grid(True)
    st.pyplot(plt)

# Bar Plot: Variance by Component
    st.markdown("### Bar Plot of Variance Explained")
    st.write("This bar plot shows the percentage of variance explained by each individual principal component. It helps identify which components are most influential.")

    plt.figure(figsize=(8, 6))
    components = range(1, max_components + 1)
    plt.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='teal')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('Variance Explained by Each Principal Component')
    plt.xticks(components)
    plt.grid(True, axis='y')
    st.pyplot(plt)

# Combined Bar + Line Plot 
    st.markdown("### Combined Variance Explained Plot")
    st.write("This plot combines two insights: the blue bars show how much variance each component explains individually, while the red line shows how much total variance is explained as you add more components.")
    
    # Get individual and cumulative variance explained by each principal component
    explained = pca_full.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    # Create plot with twin axes
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Bar plot for individual variance explained
    bar_color = 'steelblue'
    ax1.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Variance Explained (%)', color=bar_color)
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax1.set_xticks(components)
    ax1.set_xticklabels([f"PC{i}" for i in components])

    # Add value labels above each bar
    for i, v in enumerate(explained):
        ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

    # Create second y-axis to show cumulative variance explained
    ax2 = ax1.twinx()
    line_color = 'crimson'
    ax2.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax2.set_ylim(0, 100)

    # Remove grid lines
    ax1.grid(False)
    ax2.grid(False)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

    # Finishing and showing the plot in Streamlit and closing it
    plt.title('PCA: Variance Explained', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig) 

# This should wrap up our section on PCA! We should have a lot of good visualizations being produced from this tab on PCA and hopefully a good understanding of what everything means!
# -----------------------------------------------
# KMeans Clustering Tab
# -----------------------------------------------
# This section will focus on running KMeans Clustering on the standardized data and visualizing the clustering results in 2D using PCA for dimensionality reduction.
if options == "KMeans Clustering":
    st.header("KMeans Clustering")
    st.write("KMeans clustering is an unsupervised learning technique that groups data points into a specified number of clusters (k) based on feature similarity. The algorithm works by placing k centroids, assigning points to the nearest centroid, and updating the centroids based on those assignments."
    "This process repeats until the clusters stabilize. Use the options below to explore how your data can be grouped based on the features you select.")
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
    # This returns the strings for prediction and adds a selectbox for which predictor variables the user would like to use
    selected_features = st.multiselect("Select Predictor Variables for KMeans Clustering", predictors, default=predictors)
    if not selected_features:
        st.warning("Please select at least one predictor variable.")
    else:
        X = current_df[selected_features]
        y = current_df[target]

    # Prepare the data for clustering
    pd.DataFrame(X, columns=selected_features)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Run KMeans with a user input k to allow for fun interaction and exploration!
    k = st.slider("Select number of clusters (k) for initial KMeans visualization:", min_value=2, max_value=10, value=2, step=1)
    st.write(f"You've selected **{k} clusters**. In KMeans, increasing the number of clusters creates more, smaller groups that may capture finer patterns in the data, while decreasing it groups the data into broader categories. Try different values to see how the clustering pattern changes!")
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_std)

    # Reduce data to 2D with PCA for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    # Visualize the KMeans clusters in 2D PCA space
    st.subheader("KMeans Cluster Visualization")
    st.write("This plot shows the result of clustering your data into {k} groups using KMeans. It uses PCA to reduce dimensionality for visualization.")

    # Need to create a dynamic scatter plot that aligns with the user's choice of the number of clusters.
    colors = plt.cm.get_cmap('tab10', k)
    plt.figure(figsize=(8, 6))
    for cluster_id in range(k):
        plt.scatter(X_pca[clusters == cluster_id, 0], X_pca[clusters == cluster_id, 1],
                    color=colors(cluster_id), alpha=0.7, edgecolor='k', s=60, label=f'Cluster {cluster_id}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering: 2D PCA Projection')
    plt.legend(loc='best')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

    # Let user select a preserved label to color by (e.g., species, island, sex)
    if label_columns:
        st.subheader("True Labels in 2D PCA Projection")
        label_option = st.selectbox("Select a label to color points by:", list(label_columns.keys()))
        label_data = label_columns[label_option]
        st.write(f"This plot uses the same PCA projection and colors the points by the selected original label column: **{label_option}**.")

    unique_labels = pd.Series(label_data).dropna().unique()
    if len(unique_labels) <= 20:
        plt.figure(figsize=(8, 6))
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = label_data.values == label  # Fix for Series comparison
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=cmap(i), alpha=0.7, edgecolor='k', s=60, label=str(label))
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'2D PCA Projection Colored by {label_option}')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        st.pyplot(plt)
        plt.close()
    else:
        st.info(f"There are {len(unique_labels)} unique values in '{label_option}', which is too many to display clearly.")

    # Calculate and show the silhouette score for current clustering
    st.subheader("Silhouette Score")
    st.write("The silhouette score measures how well each point fits within its cluster. Higher values (closer to 1) are better. A high silhouette score means clusters are distinct and well-formed.")
    st.write(f"Silhouette Score for k={k}: {silhouette_score(X_std, clusters):.2f}")

    # Evaluate clustering quality over a range of k values
    st.subheader("KMeans Performance Across Multiple k Values")
    st.write("Try different numbers of clusters to see how clustering quality changes. These plots help you choose a good number of clusters.")

    # Initialize containers for WCSS and silhouette scores
    ks = range(2, 11)
    wcss = []
    silhouette_scores = []

    # Run KMeans for each k and collect metrics
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_std)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X_std, labels))

    # Letting the user know what the WCSS is and what it does!
    st.write("The plots below help you decide how many clusters (k) to use in KMeans clustering:")
    st.write("A lower **Within-Cluster Sum of Squares (WCSS)** value (e.g., 500) means that the points within each cluster are close to their cluster's center, indicating more compact groupings. However, lower isn't always better. If you keep adding clusters, WCSS will decrease. The goal is to find the 'elbow' point where adding more clusters gives diminishing returns.")

    # Elbow Plot: Shows how WCSS changes as k increases
    st.subheader("Elbow Method for Optimal k")
    fig_elbow, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(ks, wcss, marker='o')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.set_title('Elbow Plot')
    ax1.grid(True)
    st.pyplot(fig_elbow)
    plt.close(fig_elbow)

    # Silhouette Score Plot: Evaluates how well-separated the clusters are
    st.subheader("Silhouette Score for Optimal k")
    fig_silhouette, ax2 = plt.subplots(figsize=(9, 6))
    ax2.plot(ks, silhouette_scores, marker='o', color='green')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Plot')
    ax2.grid(True)
    st.pyplot(fig_silhouette)
    plt.close(fig_silhouette)

# There were a lot of cool visualizations and interactions in this tab for Kmeans! I hope you agree and learned something from this section.
# -----------------------------------------------
# Hierarchical Clustering Tab
# -----------------------------------------------
#  In this tab, we'll be tackling Hierarchical clustering and what it is, what it does, and various methods attached to it.
if options == "Hierarchical Clustering":
    st.header("Hierarchical Clustering")
    st.write("Hierarchical clustering is a way to group data points based on their similarity without specifying centroids. It builds a tree-like structure called a dendrogram to show which points or clusters merge together as distance increases.")
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
    # This returns the strings for prediction and adds a selectbox for which predictor variables the user would like to use
    selected_features = st.multiselect("Select Predictor Variables for Hierarchical Clustering", predictors, default=predictors)
    if not selected_features:
        st.warning("Please select at least one predictor variable.")
    else:
        X = current_df[selected_features]
        y = current_df[target]

    # Prepare the data for clustering
    pd.DataFrame(X, columns=selected_features)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Generate the dendrogram
    st.subheader("Dendrogram")
    st.write("The dendrogram below shows how the clustering algorithm merges individual points into larger clusters. "
        "Each block shape indicates a merge, and taller merges mean the combined clusters were less similar. "
        "This helps you visually determine a good number of clusters by looking for big jumps in merge height.")
    # Creating the dendrogram using Ward's method
    Z = linkage(X_std, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, truncate_mode=None, no_labels=True, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance")
    st.pyplot(fig)
    plt.close(fig)

    # Let the user select how many clusters to extract from the dendrogram
    k = st.slider("Select number of clusters to extract:", min_value=2, max_value=10, value=4)
    st.write(f"You selected **{k} clusters**. The dendrogram is 'cut' at this level to divide your data into {k} groups. This process is based purely on distances, not predefined centers.")

    # Run agglomerative clustering to assign cluster labels
    cluster_model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    cluster_labels = cluster_model.fit_predict(X_std)
   # Create a DataFrame to display cluster assignments
    cluster_df = X.copy()  # This contains the selected (scaled) features used for clustering
    cluster_df["Cluster"] = cluster_labels  # Add a new column showing which cluster each row belongs to

    # Show a sample of the clustered data
    st.subheader("Cluster Assignments")
    st.write("The rows below shows some of the data values for various observations and the cluster number they were assigned to. Clusters are formed by grouping the z scores of data points that are similar to each other based on the features you selected above.")
    st.dataframe(cluster_df.head())

    # Display the number of data points in each cluster
    st.write("**Cluster Sizes**")
    st.write("This summary shows how many data points were assigned to each cluster. Large imbalances may suggest natural groupings, noise, or that some clusters are more compact than others.")
    cluster_counts = cluster_df["Cluster"].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        st.write(f"Cluster {cluster_id}: {count} points")

    # Use PCA to reduce data to 2D for visualization
    X_pca = PCA(n_components=2).fit_transform(X_std)

    # Create and display scatterplot of PCA-reduced clusters
    st.subheader("Cluster Visualization via PCA")
    st.write("This scatter plot reduces your selected features into two principal components to help visualize the clusters. Although PCA is not used during clustering, it gives you a sense of how well-separated the groups are.")
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', k)
    for cluster_id in range(k):
        plt.scatter(X_pca[cluster_labels == cluster_id, 0], X_pca[cluster_labels == cluster_id, 1],
                    color=colors(cluster_id), alpha=0.7, edgecolor='k', s=60, label=f'Cluster {cluster_id}')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Agglomerative Clustering Visualization (PCA 2D)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

# Wow. It's been a journey to get here but we did! I hope you enjoyed this tab as much as the others and was able to learn something new about dendrograms or Hierarchical Clustering!
# -----------------------------------------------
# Feedback Tab
# -----------------------------------------------
# Just asking the user of the app for some feedback on their experience!
if options == "Feedback":
    st.header("Feedback")
    st.markdown("Please rate your experience out of ten and add your thoughts or suggestions below.")
    
    # Ask the user for a rating out of 10.
    rating = st.slider("Rate your experience (0 = horrendous, 10 = perfect):", 0, 10, 5)
    
    # Create a text box for additional comments or suggestions.
    comments = st.text_area("Your comments or suggestions on the app:")
    
    # When the user clicks the submit button, display a thank-you message along with their inputs.
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
        st.write("Rating:", rating)
        st.write("Comments:", comments)
# Thank you again for using and checking out my Unsupervised Machine Learning App!!!
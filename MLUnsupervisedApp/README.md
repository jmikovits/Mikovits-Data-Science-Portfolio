### Unsupervised ML App Project

### Project Overview
This project is an interactive Unsupervised Machine Learning App that allows users to upload their own datasets or use a built-in one to explore clustering and dimensionality reduction techniques. The app includes Principal Component Analysis (PCA), KMeans Clustering, and Hierarchical Clustering, with interactive controls for selecting features and tuning hyperparameters like the number of clusters. Users can view dataset summaries, examine dendrograms, interpret silhouette scores and elbow plots, and visualize results in 2D using PCA projections. The goal is to help users experiment with unsupervised learning concepts and uncover structure in their data through clear, interactive visualizations.

### Instructions

#### Running the App Locally
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jmikovits/MLUnsupervisedApp.git
   cd MLUnsupervisedApp
2. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   matplotlib==3.10.1
   numpy==2.2.4
   pandas==2.2.3
   scikit_learn==1.6.1
   seaborn==0.13.2
   streamlit==1.37.1
   streamlit-option-menu==0.3.0
   scipy>=1.11.0
3. **Run the App:**
  ```bash
    streamlit run ML-Unsupervised-App.py
```
4. **Access the App:**
   ***Open your browser and navigate to the URL provided in your terminal (usually http://localhost:8501).***

### Deployed Version
This app is also deployed online. Please visit here to access it: [ML-Unsupervised-App](https://mikovits-data-science-portfolio-3cqemgud4yuek86idfwt5p.streamlit.app/).

### App Features
- **Dataset Upload & Summary:**
  - Provided with the Palmer's Penguins dataset as an example.
  - Upload CSV or Excel files.
  - View data preview, summary statistics, missing values information, and correlation heatmaps.
  
- **Model Selection & Hyperparameter Tuning:**
  - **Principal Component Analysis (PCA):** Reduce dimensionality by selecting predictor variables, view variance explained, and visualize data in a 2D PCA projection.
  - **KMeans Clustering:** Choose the number of clusters (k), select input features, and visualize cluster assignments using PCA. Includes elbow plot and silhouette score to evaluate clustering quality.
  - **Hierarchical Clustering:** Build a dendrogram to explore hierarchical relationships, choose cluster cut points, and view PCA-based cluster visualizations alongside cluster size summaries.
  
- **Feedback Collection:**
  - Users can rate their experience and make comments on the app.
  
- **Data Preprocessing Options:**
  - Access to preprocessed and cleaned Palmer's Penguins dataset. 
  - Choose between scaled and non-scaled data.
  - Automatic cleaning such as dropping rows with string data in numeric analyses and encoding categorical variables.

### References
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Scikit-Learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)

### Visual Examples
Below are some example screenshots of the app’s key functionalities:

- **Dataset Summary:**  
  ![Dataset Summary Screenshot](https://github.com/user-attachments/assets/abf9a895-35fe-4226-82db-3639547fd878)
  
- **PCA Combined Variance Explanied Output:**  
  ![PCA Screenshot](https://github.com/user-attachments/assets/26135df2-71f3-4474-9305-92ecdf93620e)
  
- **KMeans Clustering Visualization:**  
  ![KMeans Clustering Output Screenshot](https://github.com/user-attachments/assets/3fa154bf-9be1-4181-8e1c-3d2fa0277fdb)
  
- **Hierarchical Clustering Dendrogram:**  
  ![Dendrogram Screenshot](https://github.com/user-attachments/assets/27fcce2e-893d-422a-9c00-11813f4be257)

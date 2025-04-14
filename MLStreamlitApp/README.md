### ML App Project Repository

### Project Overview
This project is an interactive Machine Learning App that enables users to upload their own datasets and explore various supervised learning models. The app supports multiple regression and classification techniques, including Linear Regression, K-Nearest Neighbors, and Decision Trees. Users can view dataset summaries, adjust hyperparameters interactively, and receive detailed evaluation metrics (e.g., R², MSE, RMSE for regression and accuracy, confusion matrix, ROC curve, and AUC for classification). The goal of the project is to make it easy for users to experiment with machine learning concepts, analyze model outputs, and gain insights into data through interactive visualizations.

### Instructions

#### Running the App Locally
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/ml-app-project.git
   cd ml-app-project
2. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   streamlit==1.12.0 
   pandas==1.5.0
   numpy==1.21.0
   seaborn==0.11.0
   matplotlib==3.5.1
   scikit-learn==1.1.3
   streamlit-option-menu==0.3.0
3. **Run the App:**
  ```bash
    streamlit run app.py
```
4. **Access the App:**
   ***Open your browser and navigate to the URL provided in your terminal (usually http://localhost:8501).***

### Deployed Version
This app is also deployed online. Please visit here to access it: [ML-App](https://mikovits-data-science-portfolio-as42mw2534kdhztxv7xcwg.streamlit.app/).

### App Features
- **Dataset Upload & Summary:**
  - Upload CSV or Excel files.
  - View data preview, summary statistics, missing values information, and correlation heatmaps.
  
- **Model Selection & Hyperparameter Tuning:**
  - **Linear Regression:** Choose target and predictor variables, view regression metrics (R², MSE, RMSE), and visualize Actual vs. Predicted values.
  - **K-Nearest Neighbors Regression:** Select predictor variables, adjust the number of neighbors, and evaluate with regression metrics.
  - **Decision Tree Classification:** For binary classification tasks, select variables, tweak parameters (max depth and min samples split), and view classification outputs including accuracy, confusion matrix, ROC curve, and AUC score.
  
- **Feedback Collection:**
  - Users can rate their experience and make comments on the app.
  
- **Data Preprocessing Options:**
  - Choose between scaled and non-scaled data.
  - Automatic cleaning such as dropping rows with string data in numeric analyses and encoding categorical variables.

### References
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- Various GitHub repositories on interactive ML apps such as [link](https://datascience-hozsu8fhxkw7gszekif27x.streamlit.app/).

### Visual Examples
Below are some example screenshots of the app’s key functionalities:

- **Dataset Summary:**  
  ![Dataset Summary Screenshot](![image](https://github.com/user-attachments/assets/61467123-736b-4ab9-9fd3-730018c603b9))
  
- **Linear Regression Output:**  
  ![Linear Regression Screenshot](![image](https://github.com/user-attachments/assets/56fad4ff-24fa-4941-a413-c6863fc67089))
  
- **K-Nearest Neighbors Visualization:**  
  ![KNN Output Screenshot](![image](https://github.com/user-attachments/assets/223b822c-8c27-4546-bd93-acb9683591c9))
  
- **Decision Tree Classification Metrics:**  
  ![Decision Tree Classification Screenshot](![image](https://github.com/user-attachments/assets/03e3cc2c-47bf-415c-b75d-56560d6254f8))

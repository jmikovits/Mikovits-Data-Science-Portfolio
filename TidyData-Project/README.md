### Tidy Data Project: 2008 Olympic Medalists Analysis

### Project Overview
This repository contains a Jupyter Notebook designed to clean, transform, and analyze an Olympic medalist dataset following **tidy data principles**. The dataset contains 2008 Olympic medalist data across multiple sports. Originally, it was in wide format, where each sport and gender had separate columns for medals. The dataset required cleaning and restructuring to align with tidy data principles. The project ensures that each variable is stored in a column, each observation in a row, and each observational unit in a separate table. The cleaned dataset is then able to be used for aggregation, visualization, and statistical analysis.

### Instructions

#### Cloning and Running the Jupyter Notebook
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jmikovits/TidyData-Project.git
   cd TidyData-Project.ipynb
2. **Open a terminal or command prompt and navigate to the project folder.**
3. **Install Required Dependencies/Libraries:**
   ```bash
   pip install -r requirements.txt
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.tree import export_text
   from sklearn.metrics import precision_score, recall_score, f1_score
   from sklearn.metrics import mean_squared_error, mean_absolute_error
4. **Run the App:**
  Start Jupyter Notebook by running TidyData-Project.ipynb and run all cells!

### App Features: Data Cleaning and Transformation
- **Reshape Data:**
  - Used pd.melt() to convert wide-format columns into a long-format dataset.
  
- **Standardized Column Names:**
  - Cleaned column names using str.replace() and str.split().
  
- **Handled Missing Values:**
  - Used dropna() and fillna() to remove or fill NaN values.
 
- **Pivot Table Aggregation:**
  - Created a pivot table to summarize medal counts by sport and gender.
 
- **Generated Visualizations:**
  - Used matplotlib and seaborn to create bar charts and heatmaps for insights.
 
- **Feedback Collection:**
  - Users can rate their experience and make comments on the app.

### References
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Tidy Data Principles by Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)


### Visual Example
Below is an example screenshot of the Notebook's utilization of Tidy Data:
**Heatmap:**  
  ![image](https://github.com/user-attachments/assets/c5321544-616e-4894-9227-2319b97c3c7a)
  




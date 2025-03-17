# Tidy Data Project: Olympic Medalists Analysis  

### This repository contains an interactive Jupyter Notebook designed to clean, transform, and analyze an Olympic medalist dataset following **tidy data principles**. The project ensures that each variable is stored in a column, each observation in a row, and each observational unit in a separate table. The cleaned dataset is then used for aggregation, visualization, and statistical analysis.  

### 📊 Dataset Description
#### The dataset contains Olympic medalist data across multiple sports. Originally, it was in wide format, where each sport and gender had separate columns for medals. The dataset required cleaning and restructuring to align with tidy data principles.

### 🔄 Data Cleaning & Transformation
#### Reshaped Data: Used pd.melt() to convert wide-format columns into a long-format dataset.
#### Standardized Column Names: Cleaned column names using str.replace() and str.split().
#### Handled Missing Values: Used dropna() and fillna() to remove or fill NaN values.
#### Pivot Table Aggregation: Created a pivot table to summarize medal counts by sport and gender.
#### Generated Visualizations: Used matplotlib and seaborn to create bar charts and heatmaps for insights.

### 📌 Key Findings
#### ✅ Athletics and Swimming have the highest number of medals awarded across both genders.
#### ✅ Some sports are gender-exclusive (e.g., rhythmic gymnastics for women, greco-roman wrestling for men).
#### ✅ Basketball and Water Polo have nearly equal medal distributions between male and female athletes.

---

## How to Run the Notebook  
### 🚀 Run the Notebook
1. Clone this repository or download the notebook file.
2. Open a terminal or command prompt and navigate to the project folder.
3. Start Jupyter Notebook by running TidyData-Project.ipynb
4. Run all cells. 


### Install Required Dependencies  

Python must be installed with the following required packages:  

```bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns






import streamlit as st
import pandas as pd
#Importing the necessary libraries and the Palmer's Penguins Dataset

df= pd.read_csv("data/penguins.csv")
st.title("Let's look at the Palmer's Penguins Dataset!")

#Short description of the app
st.write("### This app aims to utilize Streamlit to create interactive widgets that provide insights and fun interaction with the Palmer's Penguins Dataset!")

st.write("Here's a sneak peak into the overall dataset:")
st.dataframe(df)

# Adding User Interaction with Widgets 

# Displaying a short summary of the statistics of the dataset
st.write("### Summary Statistics")
if st.checkbox("Show summary statistics"):
    st.write(df.describe())

# Adding a selectbox to allow users to filter data by penguin species
st.write("### Select a Species to see Characteristics of Specific Penguins in that Species.")
species = st.selectbox("Select a species", df["species"].unique())
# Filtering the dataset 
filtered_df = df[df["species"] == species]
# Displaying the filtered results
st.write(f"Penguins in {species}:")
st.dataframe(filtered_df)

# Using a selectbox to allow users to filter data by flipper length
st.write("### Use the Slider to Identify Penguins with Flipper Lengths Below the Value of the Slider.")
# Creating the slider and the bounds of the sliders as the min and max values contained in the dataset.
flip_length = st.slider("Choose a flipper length range:", 
                   min_value = df["flipper_length_mm"].min(),
                   max_value = df["flipper_length_mm"].max())
st.write(f"Penguins with flipper lengths under {flip_length}:")
# Displaying the data based on the dataset values under or equal to the user-inputed slider value
st.dataframe(df[df['flipper_length_mm'] <= flip_length])

# Using a dropdown widget to allow users to filter data by column
st.write("### Select a Column to View Unique Values in the Dataset.")
column = st.selectbox("Choose a column:", df.columns)
st.write(f"Unique values in {column}:")
# Displaying the unique data entries for the user-inputed column
st.write(df[column].unique())

# Using a widget to allow the user to reflect on their experience with the dataset and anything that they noticed during the experience.
st.write("### Add Your Observations")
user_note = st.text_area("You can enter any of your observations or notes about the dataset:")
if user_note:
    st.write("Thanks for writing your observations! You wrote:")
    st.write(user_note)

# Using a fun interactive widget to see how users liked or disliked the app
st.write("#### How did you like the app?")
st.feedback("stars")
st.write("Thank you for using my basic Streamlit app!")
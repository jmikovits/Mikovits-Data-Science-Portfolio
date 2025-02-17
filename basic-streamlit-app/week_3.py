# Import the Streamlit library
import streamlit as st

# Navigate
#ls
#cd, lets you choose something in the directory to go to

# Display a simple text message
st.title("Hello, streamlit!")


# Display a large title on the app
st.write("This is my first time using streamlit!")

#Using markdown
st.markdown('#### Excited to get started!')

# ------------------------
# INTERACTIVE BUTTON
# ------------------------

# Create a button that users can click.
# If the button is clicked, the message changes.
if st.button("Click me!"):
    st.write("You clicked the button! Nice work!")
else:
    st.write("Go ahead... click the button. I dare you.")

#Interactive Widgets

st.slider('Slide me :)', min_value=0, max_value=10)

st.radio("Who's the GOAT?", ['LeBron','Jordan'])

if st.selectbox('Select Your Favorite Artist', ['Troye Sivan',"Chappell Roan","Taylor Swift", "Sabrina Carpenter", "The Weeknd", "Ariana Grande", "Tate McRae"]):
    st.write("Nice Pick!") 

# ------------------------
# COLOR PICKER WIDGET
# ------------------------

# Creates an interactive color picker where users can choose a color.
# The selected color is stored in the variable 'color'.
color = st.color_picker("Pick a color", "#00f600")

# Display the chosen color value
st.write(f"You picked: {color}")

# ------------------------
# ADDING DATA TO STREAMLIT
# ------------------------

# Import pandas for handling tabular data
import pandas as pd

# Display a section title
st.subheader("Now, let's look at some data!")

# Create a simple Pandas DataFrame with sample data
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],  # Column of names
    'Age': [25, 30, 35, 40],  # Column of ages
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']  # Column of cities
})

# Display a descriptive message
st.write("Here's a simple table:")

# Display the dataframe in an interactive table.
# Users can scroll and sort the data within the table.
st.dataframe(df)

# ------------------------
# INTERACTIVE DATA FILTERING
# ------------------------

# Create a dropdown (selectbox) for filtering the DataFrame by city.
# The user selects a city from the unique values in the "City" column.
city = st.selectbox("Select a city", df["City"].unique())

# Create a filtered DataFrame that only includes rows matching the selected city.
filtered_df = df[df["City"] == city]

# Display the filtered results with an appropriate heading.
st.write(f"People in {city}:")
st.dataframe(filtered_df)  # Show the filtered table

# ------------------------
# NEXT STEPS & CHALLENGE
# ------------------------

# Play around with more Streamlit widgets or elements by checking the documentation:
# https://docs.streamlit.io/develop/api-reference
# Use the cheat sheet for quick reference:
# https://cheat-sheet.streamlit.app/

### Challenge:
# 1️⃣ Modify the dataframe (add new columns or different data).
# 2️⃣ Add an input box for users to type names and filter results.
# 3️⃣ Make a simple chart using st.bar_chart().
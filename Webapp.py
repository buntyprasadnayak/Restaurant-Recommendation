import streamlit as st
import pandas as pd
import pickle


# Load the pre-trained model (cosine_similarities matrix)
model_file_path = 'trained_model.sav'

try:
    with open(model_file_path, 'rb') as model_file:
        cosine_similarities = pickle.load(model_file)
    print("Model loaded successfully.")
    
except Exception as e:
    print(f"Error loading the model: {e}")
    cosine_similarities = None  # Set to None or another default value if loading fails

# Load the DataFrame df
df_file_path = '/Users/bunty/Code/MachineLearning/Restaurant_Recommendation/Merge_data.csv'

try:
    df = pd.read_csv(df_file_path, usecols=['Cuisines', 'Rating', 'Cost', 'Timings'])
    print("DataFrame loaded successfully.")
    
except Exception as e:
    print(f"Error loading the DataFrame: {e}")
    
    df = pd.DataFrame()  # Set to an empty DataFrame or handle the failure accordingly


# Load the pre-trained model (cosine_similarities matrix)
with open('trained_model.sav', 'rb') as model_file:
    cosine_similarities = pickle.load(model_file)

# Load the dataframe df
# Note: Make sure to have df available, either by loading it from the original source or saving it to a file and loading it here.
df = pd.read_csv('/Users/bunty/Code/MachineLearning/Restaurant_Recommendation/NewMerge.csv')
# Your recommendation function
def recommend(name, cosine_similarities=cosine_similarities, df=df):
    # Strip whitespace from the user input
    name = name.strip()

    # Check if the restaurant name exists in the DataFrame
    if name not in df.index:
        raise KeyError("Restaurant not found in the dataset.")

    # Create a list to put top 10 restaurants
    recommend_restaurant = []

    # Find the index of the restaurant entered
    idx = df.index.get_loc(name)

    # Find the restaurants with a similar cosine-sim value and order them from biggest number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)

    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)

    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(df.index[each])

    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['Cuisines', 'Rating', 'Cost', 'Timings'])

    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df[['Cuisines', 'Rating', 'Cost', 'Timings']][df.index == each].sample()))

    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['Cuisines', 'Rating', 'Cost'], keep=False)
    df_new = df_new.sort_values(by='Rating', ascending=False).head(7)

    return df_new

# Streamlit UI
def main():
    st.title("Restaurant Recommendation System")
    user_input = st.text_input("Enter a restaurant name:")

    if st.button("Make Prediction"):
        if cosine_similarities is not None and not df.empty:
            try:
                prediction_result = recommend(user_input)
                st.write("Top Recommendations:")
                st.write(prediction_result)
            except KeyError:
                st.write("Restaurant not found. Please enter a valid restaurant name.")
        else:
            if cosine_similarities is None:
                st.write("Model loading failed. Please check the logs for details.")
            if df.empty:
                st.write("DataFrame loading failed. Please check the logs for details.")

if __name__ == "__main__":
    main()

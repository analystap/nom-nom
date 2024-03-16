#!/usr/bin/env python
# coding: utf-8

# In[14]:


# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 02:20:31 2024

@author: Anurag Patil, Dhanashree Badhe, Nicole Chung, Sai Kulkarni, Sydney Walker
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 02:20:31 2024

@author: Anurag Patil, Dhanashree Badhe, Nicole Chung, Sai Kulkarni, Sydney Walker
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string # Add this import statement
from IPython.display import display, HTML
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


#app=Flask(__name__)
#Swagger(app)

stemmer = nltk.stem.PorterStemmer()
ENGLISH_STOP_WORDS = stopwords.words('english')
additional_stop_words = [
    'make', 'dish', 'suggest', 'recipe', 'includes', 'looking', 'cook', 'something', 
    'wish', 'eat', 'help', 'find', 'mood', 'show', 'recipes', 'try', 'thinking', 
    'cooking', 'suggestions', 'recommend', 'good', 'help', 'ideas', 'inspiration',
    '.', ',', '!', '?', '-', ':', ';', '"', "'", "’", '“', '”', '(', ')'
]
STOP_words = ENGLISH_STOP_WORDS + additional_stop_words

def recipe_tokenizer(sentence):
    # remove punctuation and set to lowercase
    for punctuation_mark in string.punctuation:
        sentence = sentence.replace(punctuation_mark, '').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    listofstemmed_words = []

    # remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in STOP_words) and (word != ''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            listofstemmed_words.append(stemmed_word)

    return listofstemmed_words



pickle_in_vectorizer = open("vectorizer.pkl","rb")
vectorizer=pickle.load(pickle_in_vectorizer)

pickle_in_recipe_data = open("recipe_data.pkl","rb")
recipe_data=pickle.load(pickle_in_recipe_data)

pickle_in_vectorized_data = open("vectorized_data.pkl","rb")
vectorized_data=pickle.load(pickle_in_vectorized_data)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
# def recipe_recommendation_engine(user_prompt):
    
#     """Here's your recipe! 
#     This is using docstrings for specifications.
#     ---
#     parameters:  
#       - name: User Input
#         in: query
#         type: text
#         required: true
#     responses:
#         200:
#             description: The output values
        
#     """
    
    
#     # Process and vectorize the input
#     input_vectorized = vectorizer.transform([user_prompt])
    
#     # Compute cosine similarity
#     cosine_sim_matrix = cosine_similarity(input_vectorized, vectorized_data)
    
#     # Get the index of the most similar recipe
#     most_similar_recipe_index = np.argmax(cosine_sim_matrix)

#     # Output the most similar recipe
#     print(user_prompt)
    
#     # Sort the recipes by their similarity score
#     top_indices = np.argsort(cosine_sim_matrix.flatten())[-1:][::-1]
#     print("Here is your recipe:")
#     for index in top_indices:
#         title = recipe_data.iloc[index]['title']
#         ingredients = recipe_data.iloc[index]['ingredients_str']
#         instructions = recipe_data.iloc[index]['instructions_str']
#         similarity_score = f"{cosine_sim_matrix[0, index]:.2f}"
    
#         # Use HTML to format the ingredients and instructions in bold
#         display(HTML(f"<div><strong>Title:</strong> {title}</div>"))
#         display(HTML(f"<div><strong>Ingredients:</strong> {ingredients}</div>"))
#         display(HTML(f"<div><strong>Instructions:</strong> {instructions}</div>"))
#         display(HTML(f"<div><strong>Similarity Score:<strong> {similarity_score}</div>"))


def recipe_recommendation_engine(user_prompt):
    # Process and vectorize the input
    input_vectorized = vectorizer.transform([user_prompt])
    
    # Compute cosine similarity
    cosine_sim_matrix = cosine_similarity(input_vectorized, vectorized_data)
    
    # Sort the recipes by their similarity score
    top_indices = np.argsort(cosine_sim_matrix.flatten())[-1:][::-1]
    
    recipes_html = ""
    for index in top_indices:
        title = recipe_data.iloc[index]['title']
        ingredients = recipe_data.iloc[index]['ingredients_str']
        instructions = recipe_data.iloc[index]['instructions_str']
        similarity_score = f"{cosine_sim_matrix[0, index]:.2f}"
    
        # Construct HTML content
        recipe_html = f"""
        <div>
            <strong>Title:</strong> {title}<br>
            <strong>Ingredients:</strong> {ingredients}<br>
            <strong>Instructions:</strong> {instructions}<br>
            <strong>Similarity Score:</strong> {similarity_score}
        </div>
        <br>
        """
        recipes_html += recipe_html
    
    return recipes_html



def main():
    st.title("NOM NOM")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">NOM NOM Recipe Recommendation </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    user_prompt = st.text_input("What's Cookin'","Type Here and click on Recipe button below")
    result=""
    if st.button("Recipe!"):
        result=recipe_recommendation_engine(user_prompt)
    if result:
        st.markdown(result, unsafe_allow_html=True)

    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
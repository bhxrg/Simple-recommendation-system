#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = {
    'id': [1, 2, 3, 4, 5],
    'gender': ['male', 'female', 'male', 'female', 'male'],
    'stream': ['science', 'commerce', 'humanities', 'commerce', 'science'],
    'subject': ['physics', 'maths', 'history', 'economics', 'chemistry'],
    'marks': [78, 56, 79, 88, 79],
    'course': ['btech', 'bsc', 'ba', 'bcom', 'Esc'],
    'specialization': ['civil', 'maths', 'history', 'hons', 'biology']
}

df = pd.DataFrame(data)

# Feature Engineering
features = ['stream', 'subject', 'marks']

# Recommendation Algorithm
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df[features].astype(str))

# Computing Cosine Similarity between courses
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Making Recommendations
def recommend_courses(new_student_profile, cosine_sim=cosine_sim):
    # Calculate TF-IDF vector for new student profile
    new_student_tfidf = tfidf_vectorizer.transform([str(val) for val in new_student_profile.values()])

    # Calculate cosine similarity between new student profile and courses
    sim_scores = cosine_similarity(new_student_tfidf, tfidf_matrix)
    sim_scores = sim_scores[0]  # Flatten the array

    # Get indices of courses sorted by similarity score (descending order)
    course_indices = sim_scores.argsort()[::-1]

    # Filter out the courses already taken by the student
    taken_courses = set(df['course'][df['id'] == new_student_profile['id']])
    recommended_courses = [(df.iloc[idx]['course'], df.iloc[idx]['specialization']) for idx in course_indices if df.iloc[idx]['course'] not in taken_courses]

    return recommended_courses

# Example: New student profile
new_student_profile = {
    'id': 6,
    'stream': 'commerce',
    'subject': 'economics',
    'marks': 85
}

# Make recommendations for the new student
recommendations = recommend_courses(new_student_profile)
print("Recommended courses:")
for course, specialization in recommendations:
    print(f"- {course} ({specialization})")


# In[ ]:





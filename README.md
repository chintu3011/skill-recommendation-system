# Skill-Based User Recommendation System

The recommendation system utilizes the **TfidfVectorizer** to tokenize and analyze skills, constructing a cosine similarity matrix. This matrix effectively captures the relationships between the required skills and the user's skills, facilitating accurate and relevant recommendations. 


## Description

The Skill-Based User Recommendation System is designed to identify and recommend users whose skills most closely align with a given set of required skills. Utilizing the Term Frequency-Inverse Document Frequency (TF-IDF) technique from the sklearn library, this system tokenizes the provided skill data and creates a cosine similarity matrix to determine the degree of relevance between users' skills and the required skills.

This recommendation system is particularly useful for recruiters, project managers, or anyone seeking to match individuals to roles, projects, or opportunities based on specific skill requirements.

## Installation


```bash
# Example command to install dependencies
pip install -r requirements.txt

# Run Fastapi server
unicorn predict:app --reload
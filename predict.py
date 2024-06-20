import pickle
import ast
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
tfidf=TfidfVectorizer(tokenizer=lambda x: x.split(' '))

# loading dataframe
with open('dataFrame.pkl','rb') as d:
    df=pickle.load(d)

print(df.columns)
# loading cosing matrix
with open('cosineMatrix.pkl','rb') as c:
    cosine=pickle.load(c)

tfidf_matrix = tfidf.fit_transform(df['Text'])
def recommend_profiles(skills, df=df, tfidf=tfidf, cosine_sim=cosine, top_n=5):
    # Preprocess the input skills
    skills = skills.lower()
    skills_vector = tfidf.transform([skills])
    
    # Compute similarity with the input skills
    sim_scores = cosine_similarity(skills_vector, tfidf_matrix).flatten()
        
    # Get top N profiles
    sim_scores_indices = sim_scores.argsort()[-top_n:][::-1]
    recommended_profiles = df.iloc[sim_scores_indices]

    return recommended_profiles

print(df.columns)

@app.get('/')
async def home():
    return {'details':'Home Page'}

@app.post('/predict/')
async def predict(skills:str='developer',quantity:int=5):
    profiles = recommend_profiles(skills,top_n=quantity)
    profiles=profiles[['Full Name','Company Name','School Name','URL','Headline','Skills','Text']]
    profiles['Headline']=profiles['Headline'].apply(lambda x:ast.literal_eval(x))
    profiles['Headline']=profiles['Headline'].apply(lambda x:' '.join(x))
    profiles=profiles.to_dict(orient='records')
    updated_profiles = [
    {**item, 'Status': 'Fresher' if not item['Company Name'] else 'Experienced'}
for item in profiles
]
    return updated_profiles
import pandas as pd
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance

from transformers import pipeline
# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

import keyBERTinspired_MMR_clusters
from keyBERTinspired_MMR_clusters import *

import glob
daily_news_folder = "news_clustering/news_by_date"

# Step 4: Use a Better Embedding Model for News
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

print(glob.glob(daily_news_folder+"/*"))

# get 1st df 
#get cluster in  topic model , ketbert mmr , get KB topic 0 
# parse name of csv for date - save  topic 1 - topic , count m representation  represention doc 
# get sentiment for topic 1 Sentiment Summary: {'positive': 17, 'negative': 34, 'neutral': 0}

# Define a function to remove stopwords
def remove_stopwords(line, stop_words):
    # Split the sentence into words
    words = line.split()  
    # Filter out stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Join the filtered words back into a single string
    return " ".join(filtered_words)


def clean_text(news_df):
    news_df = news_df.fillna("")  # Handle missing values
    news_corpus = [", ".join(map(str, row))  # Combine 'headline', 'lead_paragraph', 'snippet'
                    for row in news_df[['headline', 'lead_paragraph', 'snippet']].values
                    ]
    # Apply stopword removal
    titles_cleaned = [remove_stopwords(line, stop_words) for line in news_corpus]
    return titles_cleaned


def initiate_clusters():
    # Step 1: UMAP for Dimensionality Reduction (Clustering)
    umap_model_15D = UMAP(
        n_neighbors=15,  # Higher to capture more context
        n_components=2,  # Higher dimensionality preserves more structure for clustering
        min_dist=0.0,  # Avoid overly compact clusters
        metric="cosine",
        random_state=42,
    )

    # Step 2: UMAP for Visualization (2D)
    umap_2d_embeddings = UMAP(
        n_neighbors=15,
        n_components=2,  # 2D for visualization
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # Step 3: HDBSCAN Configuration
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,  # Increase for more stable clusters
        min_samples=5,  # Increase sensitivity
        cluster_selection_epsilon=0.05,  # Tighter clusters
        metric="euclidean",
        cluster_selection_method="eom",  # Excess of Mass for robust selection
    )
    
  
    return umap_model_15D , umap_2d_embeddings,hdbscan_model


def get_topic_model(embedding_model, embeddings_corpus, umap_model_15D , hdbscan_model,titles_cleaned):
    # Step 1: Fit BERTopic with Updated Components
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model_15D,
        hdbscan_model=hdbscan_model,)
    
    topics, probs = topic_model.fit_transform(titles_cleaned)
    
    # Step 2: Fit KeyBert Inspired
    representation_model = KeyBERTInspired()
    topic_model.update_topics(titles_cleaned, representation_model=representation_model)
    
    return topic_model

def getcorpus_sentiment(cluster_text, sentiment_model):
    if cluster_text and isinstance(cluster_text, list) and len(cluster_text) >= 1:
        day_sentiments = sentiment_model(cluster_text)  # Pass to sentiment analyzer
        day_sentiments_summary = {
            "positive": sum(1 for s in day_sentiments if s['label'] == 'POSITIVE'),
            "negative": sum(1 for s in day_sentiments if s['label'] == 'NEGATIVE'),
            "neutral": sum(1 for s in day_sentiments if s['label'] == 'NEUTRAL'),
        }

        positive = day_sentiments_summary["positive"]
        negative = day_sentiments_summary["negative"]
        neutral = day_sentiments_summary["neutral"]
    else:
        positive, negative, neutral = 0, 0, 0

    return positive, negative, neutral

    
if __name__== "__main__":
    csv_list= glob.glob(daily_news_folder+"/*")
    csv_list.sort(key=lambda x: x.split('/')[-1].split('_')[0])
    news_dates= [file_csv.split("/")[-1].split("_")[0] for file_csv in csv_list]
    
    compare_news_df =[]
    i=0
    for daily_csv in csv_list:
        
        news_df = pd.read_csv(daily_csv)
        titles_cleaned =clean_text(news_df)
        umap_model_15D , umap_2d_embeddings, hdbscan_model = initiate_clusters()
        embeddings_corpus = embedding_model.encode(titles_cleaned, show_progress_bar=True)
        topic_model= get_topic_model(embedding_model, embeddings_corpus, umap_model_15D , hdbscan_model,titles_cleaned)
        print (topic_model.get_topic_info())
        cluster_text= topic_model.get_representative_docs(0) 
        positive, negative, neutral = getcorpus_sentiment(cluster_text, sentiment_analyzer)
        try:
            compare_news_df.append({
                'Date': news_dates[i],
                'Count': topic_model.get_topic_info(0)["Count"],
                'Name': topic_model.get_topic_info(0)["Name"],
                'Representation': topic_model.get_topic_info(0)["Representation"],
                'Representative_Docs': topic_model.get_topic_info(0)["Representative_Docs"],
                'Positive': positive,
                'Negative': negative,
                'Neutral': neutral
            })
            

        except Exception as e:
            compare_news_df.append({
                'Date': news_dates[i],
                'Count': 0,
                'Name': "None",
                'Representation':"None",
                'Representative_Docs': "None",
                'Positive': 0,
                'Negative': 0,
                'Neutral': 0
            })
            
        
        print(daily_csv)
        #print("\n", titles_cleaned)
        print("\n\n", i)
        i+=1
        
    df = pd.DataFrame(compare_news_df)
    df.to_csv("compare_daily_cluster.csv", index=None)
    print(df)
        
     
    
   
    


import pandas as pd
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance

import keyBERTinspired_MMR_clusters
from keyBERTinspired_MMR_clusters import *

# Initialize stopwords list
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}


# Define a function to remove stopwords
def remove_stopwords(line, stop_words):
    # Split the sentence into words
    words = line.split()  
    # Filter out stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Join the filtered words back into a single string
    return " ".join(filtered_words)

# Function to create BERTopic model and get topics
def get_topics_prob(embedding_model, umap_model, hdbscan_model, titles_cleaned):
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
    )
    topics, probs = topic_model.fit_transform(titles_cleaned)
    return topics, probs, topic_model

def plot_umapp_cluster(topic_model, titles_cleaned ):
    
    # Step 1: Generate Embeddings
    embedding_model = SentenceTransformer("thenlper/gte-small")
    embeddings_corpus = embedding_model.encode(titles_cleaned, show_progress_bar=True)
    
    # Step 2: UMAP for Visualization (2D)
    umap_2d_embeddings = UMAP(
        n_neighbors=5,
        n_components=2,  # 2D for visualization
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings_corpus)

    # Step 3: Visualize Topics
    fig = topic_model.visualize_documents(
        news_corpus,
        reduced_embeddings=umap_2d_embeddings,  # Use precomputed 2D embeddings
        width=1200,
        hide_annotations=True
    )
    fig.show()

if __name__ == "__main__":
    # Load the news corpus
    news_data = pd.read_csv("november_news.csv")
    news_data = news_data.fillna("")  # Handle missing values
  
    news_corpus = [", ".join(map(str, row))  # Combine 'headline', 'lead_paragraph', 'snippet'
                    for row in news_data[['headline', 'lead_paragraph', 'snippet']].values
                    ]

    print(news_corpus[:3])
    # Apply stopword removal
    titles_cleaned = [remove_stopwords(line, stop_words) for line in news_corpus]
    print(titles_cleaned[:3])

    # Define embedding models and clustering parameters
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Step 1: UMAP for Dimensionality Reduction (Clustering)
    umap_model = UMAP(
        n_neighbors=15,  # Higher to capture more context
        n_components=5,  # Higher dimensionality preserves more structure for clustering
        min_dist=0.0,  # Avoid overly compact clusters
        metric="cosine",
        random_state=42,
    )
    
    # Step 2: HDBSCAN Configuration
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,  # Increase for more stable clusters
        min_samples=5,  # Increase sensitivity
        cluster_selection_epsilon=0.05,  # Tighter clusters
        metric="euclidean",
        cluster_selection_method="eom",  # Excess of Mass for robust selection
    )

    # Get topics and probabilities
    topics, probs, topic_model = get_topics_prob(embedding_model, umap_model, hdbscan_model, titles_cleaned)

    #Generate hierarchical topics
    hier_topics = topic_model.hierarchical_topics(titles_cleaned)
    topic_tree = topic_model.get_topic_tree(hier_topics)
    
    # Print or visualize the topic tree
    print(topic_tree)
    
    # Inspect topics
    print(topic_model.get_topic_info())
    
    # Get keyword topic clustering
    advanced_topics =  Key_Bert_MMR()
    topic_model_keybert = advanced_topics.key_BERT_inspired(topic_model,titles_cleaned )
    topic_model_mmr = advanced_topics.MMR(topic_model,titles_cleaned )
    
    print(topic_model_keybert.get_topic_info())
    print(topic_model_mmr.get_topic_info())
    
    
    #plot umap cluster
    plot_umapp_cluster(topic_model, titles_cleaned )
    plot_umapp_cluster(topic_model_keybert, titles_cleaned )
    plot_umapp_cluster(topic_model_mmr, titles_cleaned )
    
    #plot heatmap
    # Visualize relationships between topics
    fig1 = topic_model.visualize_heatmap(n_clusters=1)
    fig1.show()
    
    # Visualize the potential hierarchical structure of topics
    fig2 =topic_model.visualize_hierarchy()
    fig2.show()
    
    # Visualize the potential hierarchical structure of topics
    fig3 = topic_model.visualize_barchart()
    fig3.show()
    
    # Visualize the potential hierarchical structure of topics
    fig4 = topic_model_keybert.visualize_barchart()
    fig4.show()
      
    #print("")
    topic_model
    
    
    
    


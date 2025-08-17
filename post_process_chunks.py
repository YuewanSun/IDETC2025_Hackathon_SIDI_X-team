
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import json
import re

def cluster_terms(terms, dist_threshold=0.45):
    norm = [t for t in terms]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))
    X = vec.fit_transform(norm)  # sparse
    # cosine distance on L2-normalized TF-IDF is ~euclidean on normalized vectors
    # convert to dense if small; for big sets, compute pairwise cosine in chunks
    Xd = X.toarray()
    # Agglomerative on cosine distance via 1 - cosine_sim
    sim = (Xd @ Xd.T) / (np.linalg.norm(Xd, axis=1, keepdims=True) * np.linalg.norm(Xd, axis=1))
    dist = 1 - np.clip(sim, 0, 1)
    model = AgglomerativeClustering(
        metric="precomputed",                  # <- changed from affinity
        linkage="average",
        distance_threshold=dist_threshold,
        n_clusters=None,
    )
    labels = model.fit_predict(dist)
    groups = {}
    for lbl, original in zip(labels, terms):
        groups.setdefault(lbl, []).append(original)
    return list(groups.values())



def cluster_terms_main():
    # Read the CSV file
    csv_path = "rules_chunks_processed_2.csv"
    print(f"Reading keywords from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract keywords from all rows
    all_keywords = set()
    for keywords_json in df['keywords']:
        try:
            # Parse the JSON string to get the list of keywords
            keywords_list = json.loads(keywords_json)
            # Add each keyword to the set
            for keyword in keywords_list:
                all_keywords.add(keyword.strip())
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON: {keywords_json}")
    
    # Convert set to list for clustering
    all_keywords_list = list(all_keywords)
    print(f"Found {len(all_keywords_list)} unique keywords")
    
    # Use cluster_terms function to group similar keywords
    keyword_clusters = cluster_terms(all_keywords_list, dist_threshold=0.8)
    
    # Print the results
    print(f"Grouped into {len(keyword_clusters)} clusters")
    
    # Save clusters to a file
    with open("keyword_clusters.txt", "w", encoding="utf-8") as f:
        for i, cluster in enumerate(keyword_clusters):
            f.write(f"Cluster {i+1}: {', '.join(cluster)}\n")
    
    print(f"Clusters saved to keyword_clusters.txt")
    
    # Print first 10 clusters as example
    print("\nSample clusters:")
    for i, cluster in enumerate(keyword_clusters[:10]):
        print(f"Cluster {i+1}: {', '.join(cluster)}")
def read_keyword_clusters(file_path):
    """Read the keyword clusters file and create a mapping from keywords to cluster names."""
    keyword_to_cluster = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Extract cluster name and keywords
        match = re.match(r'(Cluster \d+): (.*)', line)
        if match:
            cluster_name = match.group(1)
            keywords = [k.strip() for k in match.group(2).split(',')]
            
            # Map each keyword to its cluster name
            for keyword in keywords:
                keyword_to_cluster[keyword] = cluster_name
    
    return keyword_to_cluster

def add_cluster_groups_column(input_csv, output_csv, keyword_clusters_file):
    """
    Add a column with cluster group references to the CSV file.
    
    Args:
        input_csv: Path to the input CSV file
        output_csv: Path to save the output CSV file
        keyword_clusters_file: Path to the keyword clusters file
    """
    # Read the keyword clusters
    keyword_to_cluster = read_keyword_clusters(keyword_clusters_file)
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Add new column for cluster groups
    df['cluster_group_list'] = None
    
    # Process each row
    for i, row in df.iterrows():
        try:
            # Parse the keywords JSON string to a list
            keywords = json.loads(row['keywords'])
            
            # Find the cluster name for each keyword
            cluster_groups = []
            for keyword in keywords:
                if keyword in keyword_to_cluster:
                    cluster_groups.append(keyword_to_cluster[keyword])
            
            # Deduplicate the list
            cluster_groups = list(dict.fromkeys(cluster_groups))
            
            # Store the new list as JSON string
            df.at[i, 'cluster_group_list'] = json.dumps(cluster_groups)
        except (json.JSONDecodeError, TypeError):
            # If there's an error parsing the JSON, use empty list
            df.at[i, 'cluster_group_list'] = json.dumps([])
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Processed CSV saved to {output_csv}")

if __name__ == "__main__":
    input_file = "rules_chunks_processed_2.csv"
    output_file = "rules_chunks_with_clusters.csv"
    clusters_file = "keyword_clusters.txt"

    add_cluster_groups_column(input_file, output_file, clusters_file)
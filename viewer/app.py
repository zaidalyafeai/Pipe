from flask import Flask, render_template, request, redirect, url_for
import json
import os
import collections
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time
from tqdm import tqdm
import concurrent.futures
import multiprocessing
from functools import lru_cache
from collections import Counter

app = Flask(__name__)

# Global variables to store cached data and timestamp
models_data_cache = {}
last_cache_update = 0
CACHE_VALIDITY_PERIOD = 1800  # Cache validity in seconds (30 minutes)

def process_single_file(file_path, max_num_requests=10000):
    """Process a single model file and return its data."""
    requests_data = []
    model_name = ""
    languages = []
    dataset_id = os.path.basename(file_path)
    responses_file = os.path.join(file_path, "results.jsonl")
    if not os.path.exists(responses_file):
        print(f"File {responses_file} does not exist")
        return None
        
    try:
        with open(responses_file, "r") as f:
            for line in tqdm(f, total=max_num_requests):
                if max_num_requests and len(requests_data) >= max_num_requests:
                    break
                if not line.strip():
                    continue
                    
                try:
                    json_data = json.loads(line)
                    model_name = json_data["model"]
                    language = json_data["language"]
                    languages.append(language)
                    score = -1
                    generated_text = ""
                    keywords = ""
                    score = json_data["score"]
                    reasoning = json_data["reasoning"]
                    input_text = json_data["input_text"]
                    if "generated_text" in json_data:
                        generated_text = json_data["generated_text"]
                    else:
                        generated_text = ""
                    thinking = json_data["thinking"]
                    keywords = json_data["keywords"]
                    requests_data.append({
                        "score": score,
                        "reasoning": reasoning,
                        "input_text": input_text,
                        "generated_text": generated_text,
                        "thinking": thinking,
                        "keywords": keywords
                    })
                except Exception:
                    continue
                    
        if not model_name or not requests_data:
            return None
            
        majority_language = max(set(languages), key=languages.count)
        scores = [request["score"] for request in requests_data]
        score_counts = collections.Counter(scores)
        
        score_distribution = {
            "labels": list(range(6)),
            "data": [score_counts.get(i, 0) for i in range(6)]
        }
        
        model_key = f"{model_name}-{majority_language}-{dataset_id}"
        # Get directory creation/modification time
        try:
            creation_ts = os.path.getctime(file_path)
        except Exception:
            creation_ts = 0
        return {
            model_key: {
                "num_requests": len(requests_data),
                "requests_data": requests_data,
                "score_distribution": score_distribution,
                "language": majority_language,
                "dataset_id": dataset_id,
                "model_name": model_name,
                "creation_ts": creation_ts,
                "thinking": thinking
            }
        }
    except Exception:
        return None

@lru_cache(maxsize=1)
def get_models_data(force_reload=False):
    """Process all model data and return a dictionary of models with their details.
    Uses caching and parallel processing to improve performance."""
    global models_data_cache, last_cache_update
    
    current_time = time.time()
    if not force_reload and models_data_cache and (current_time - last_cache_update) < CACHE_VALIDITY_PERIOD:
        return models_data_cache
    
    models_data = {}
    base_path = "/ibex/ai/home/alyafez/.cache/jql/"
    
    # Get list of directories to process
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) 
             if os.path.isdir(os.path.join(base_path, f))]
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_to_file = {executor.submit(process_single_file, file_path): file_path 
                         for file_path in files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                         total=len(files), 
                         desc="Processing model data"):
            result = future.result()
            if result:
                models_data.update(result)
    
    models_data_cache = models_data
    last_cache_update = current_time
    
    return models_data

def get_clusters(requests_data, num_clusters=20, highlight_score=None):
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    import random

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode([" ".join(request["keywords"]) for request in requests_data], show_progress_bar=True)
    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(embeddings)

    clustering = KMeans(n_clusters = num_clusters).fit(tsne_data)
    labels = clustering.labels_
    clusters = []
    
    # Default color scheme - all gray except highlighted score
    if highlight_score is None:
        color_hex = ["#000000", "#0000FF", "#00FF00", "#FF0000", "#FFFF00", "#FF00FF"] # different colors
    else:
        color_hex = ["#80808020"] * 6  # All gray by default
        color_hex[highlight_score] = "#0000FF"  # Highlight selected score in blue
    
    colors = [f"{color_hex[i]}" for i in range(6)]

    # Calculate most frequent keyword for each cluster
    cluster_keywords = {}
    for idx, label in enumerate(labels):
        if label not in cluster_keywords:
            cluster_keywords[label] = []
        cluster_keywords[label].extend(requests_data[idx]["keywords"])

    # Get most frequent keyword for each cluster
    cluster_top_keywords = {}
    for label, keywords in cluster_keywords.items():
        if keywords:
            counter = Counter(keywords)
            most_common = counter.most_common(1)
            if most_common:
                cluster_top_keywords[label] = most_common[0][0]

    for idx, request in enumerate(requests_data):
        cluster_label = labels[idx]
        top_keyword = cluster_top_keywords.get(cluster_label, "")
        clusters.append({
            "x": float(tsne_data[idx, 0]),
            "y": float(tsne_data[idx, 1]),
            "color": colors[int(request["score"]) % len(colors)],
            "reasoning": request["reasoning"],
            "thinking": request.get("thinking", ""),
            "score": request["score"],
            "keywords": request["keywords"],
            "input_text": request["input_text"],
            "cluster_label": int(cluster_label),
            "cluster_top_keyword": top_keyword
        })
    return clusters

def generate_score_graph(requests_data, model_name):
    """Generate a base64 encoded image of the score distribution."""
    scores = [request["score"] for request in requests_data]
    score_counts = collections.Counter(scores)
    
    plt.figure(figsize=(8, 6))
    plt.bar(score_counts.keys(), score_counts.values())
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(f"Score Distribution for {model_name} with {len(requests_data)} requests")
    plt.xlim(-1, 6)
    
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    """Home page showing all models and their request counts."""
    models_data = get_models_data()
    # Sort by creation_ts descending
    sorted_models = dict(
        sorted(models_data.items(), key=lambda item: item[1].get("creation_ts", 0), reverse=True)
    )
    return render_template('index.html', models_data=sorted_models)

@app.route('/model/<model_name>')
def model_details(model_name):
    models_data = get_models_data()
    if model_name in models_data:
        model_data = models_data[model_name]
        
        # Get page parameter from the query string, default to page 1
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        # Get sort parameter from query string, default to none
        sort_by = request.args.get('sort', None)
        sort_order = request.args.get('order', 'desc')
        
        # Get highlight score parameter
        highlight_score = request.args.get('highlight_score', type=int, default=None)
        
        # Create a deep copy of the requests data to avoid modifying the cached data
        requests_data = model_data['requests_data'].copy()
        
        if sort_by == 'score':
            for req in requests_data:
                if isinstance(req['score'], str):
                    try:
                        req['score'] = float(req['score'])
                    except ValueError:
                        req['score'] = 0
            
            requests_data.sort(key=lambda x: float(x['score']), reverse=(sort_order == 'desc'))
        
        total_requests = len(requests_data)
        total_pages = (total_requests + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_requests)
        
        current_page_requests = requests_data[start_idx:end_idx]
        
        # Generate cluster data with highlight parameter
        clusters = get_clusters(requests_data, highlight_score=highlight_score)
        
        return render_template('model_details.html', 
                              model_name=model_data['model_name'],
                              dataset_id=model_data['dataset_id'],
                              model_key=model_name,
                              model_data=model_data,
                              current_page_requests=current_page_requests,
                              page=page,
                              total_pages=total_pages,
                              sort_by=sort_by,
                              sort_order=sort_order,
                              clusters=clusters,
                              highlight_score=highlight_score)
    return redirect(url_for('index'))

@app.route('/refresh')
def refresh_data():
    """Endpoint to force refresh the cached data."""
    get_models_data(force_reload=True)
    return redirect(url_for('index'))

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M'):
    import datetime
    return datetime.datetime.fromtimestamp(value).strftime(format)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Changed from 127.0.0.1 to 0.0.0.0
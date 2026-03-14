import os
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

def parse_conllu(filepath):
    sentences = []
    current = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            fields = line.strip().split('\t')
            if '-' in fields[0] or '.' in fields[0]:
                continue
            current.append(fields)
    if current:
        sentences.append(current)
    return sentences

def get_parse_distance(head_indices, i, j):
    """Compute tree distance between words i and j using head indices."""
    def get_path_to_root(idx):
        path = []
        visited = set()
        while idx != 0 and idx not in visited:
            path.append(idx)
            visited.add(idx)
            idx = head_indices[idx - 1]
        path.append(0)
        return path
    
    path_i = get_path_to_root(i + 1)
    path_j = get_path_to_root(j + 1)
    set_i = {n: k for k, n in enumerate(path_i)}
    for k, node in enumerate(path_j):
        if node in set_i:
            return set_i[node] + k
    return len(path_i) + len(path_j)

def prims_mst(distance_matrix):
    """Compute minimum spanning tree using Prim's algorithm."""
    n = len(distance_matrix)
    if n == 1:
        return []
    in_mst = [False] * n
    min_edge = [float('inf')] * n
    parent = [-1] * n
    min_edge[0] = 0
    edges = []
    for _ in range(n):
        # Find minimum edge vertex not in MST
        u = min((v for v in range(n) if not in_mst[v]), 
                key=lambda v: min_edge[v])
        in_mst[u] = True
        if parent[u] != -1:
            edges.append((min(u, parent[u]), max(u, parent[u])))
        for v in range(n):
            if not in_mst[v] and distance_matrix[u][v] < min_edge[v]:
                min_edge[v] = distance_matrix[u][v]
                parent[v] = u
    return edges

def compute_linear_baseline(filepath):
    sentences = parse_conllu(filepath)
    
    uuas_correct = 0
    uuas_total = 0
    all_gold_distances = []
    all_pred_distances = []
    
    for sent in sentences:
        n = len(sent)
        if n < 2:
            continue
            
        head_indices = [int(fields[6]) for fields in sent]
        upos = [fields[3] for fields in sent]
        
        # Filter punctuation
        non_punct = [i for i in range(n) if upos[i] != 'PUNCT']
        # non_punct = list(range(n))  # include all tokens - non-filtering
        if len(non_punct) < 2:
            continue
        
        # Build gold distance matrix
        gold_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = get_parse_distance(head_indices, i, j)
                gold_matrix[i][j] = d
                gold_matrix[j][i] = d
        
        # Build linear distance matrix
        linear_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                linear_matrix[i][j] = abs(i - j)
        
        # Compute gold MST edges (non-punct only)
        non_punct_gold = gold_matrix[np.ix_(non_punct, non_punct)]
        non_punct_linear = linear_matrix[np.ix_(non_punct, non_punct)]
        
        gold_edges = set(map(tuple, prims_mst(non_punct_gold)))
        pred_edges = set(map(tuple, prims_mst(non_punct_linear)))
        
        uuas_correct += len(gold_edges & pred_edges)
        uuas_total += len(gold_edges)
        
        # Spearman for sentences of length 5-50
        if 5 <= len(non_punct) <= 50:
            for i in range(len(non_punct)):
                for j in range(i+1, len(non_punct)):
                    all_gold_distances.append(non_punct_gold[i][j])
                    all_pred_distances.append(non_punct_linear[i][j])
    
    uuas = uuas_correct / uuas_total if uuas_total > 0 else 0
    dspr, _ = spearmanr(all_gold_distances, all_pred_distances)
    
    return uuas, dspr

# Run on all languages
BASE = "/content/drive/MyDrive/5th Year/Winter Quarter, 2026/CS 224N/cs224n-structural-probe"

test_files = {
    'English': f"{BASE}/data/en/UD_English-EWT/en_ewt-ud-test.conllu",
    'French': f"{BASE}/data/fr/UD_French-GSD/fr_gsd-ud-test.conllu",
    'Spanish': f"{BASE}/data/es/UD_Spanish-AnCora/es_ancora-ud-test.conllu",
    'Italian': f"{BASE}/data/it/UD_Italian-ISDT/it_isdt-ud-test.conllu",
    'German': f"{BASE}/data/de/UD_German-GSD/de_gsd-ud-test.conllu",
    'Dutch': f"{BASE}/data/nl/UD_Dutch-Alpino/nl_alpino-ud-test.conllu",
    'Czech': f"{BASE}/data/cs/UD_Czech-PDT/cs_pdtc-ud-test-2k.conllu",
    'Finnish': f"{BASE}/data/fi/UD_Finnish-TDT/fi_tdt-ud-test.conllu",

}

print(f"{'Language':<12} {'UUAS':>8} {'DSpr.':>8}")
print("-" * 30)
for lang, filepath in test_files.items():
    if os.path.exists(filepath):
        uuas, dspr = compute_linear_baseline(filepath)
        print(f"{lang:<12} {uuas:>8.4f} {dspr:>8.4f}")
    else:
        print(f"{lang:<12} FILE NOT FOUND")
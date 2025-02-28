import numpy as np
import pandas as pd

def digitize_features(feature, bin_lenght = 0.1):
    bins = np.arange(0, 1, bin_lenght)  # Bins from 0 to 1 with step size 0.1
    feature_d = np.digitize(feature, bins) - 1
    return feature_d


def reverse_digitize_features(feature_d, bin_lenght = 0.1):
    bins = np.arange(0, 1+bin_lenght, bin_lenght)  # Bins from 0 to 1 with step size 0.1
    midpoints = bins[:-1] + 0.05  # Calculate midpoints of the bins
    feature_d = np.clip(feature_d, 0, len(midpoints) - 1)  # Ensure indices are within bounds
    return midpoints[feature_d]

def opinions_to_dataframe(workers_opinions, workers_confidence = [], bin_lenght = 0.1):
    bins = np.arange(0, 1, bin_lenght)  # Bins from 0 to 1 with step size 0.1
    data = []
    for worker_id, opinions in enumerate(workers_opinions):
        for task_id, opinion in enumerate(opinions):
            if bin_lenght != -1:
                opinion = np.digitize(opinion, bins) - 1
            if workers_confidence != []:
                confidence = workers_confidence[worker_id][task_id]
                if confidence != -1:
                    data.append([worker_id, task_id, opinion, confidence])
            else:
                data.append([worker_id, task_id, opinion])
    if workers_confidence != []:
        df = pd.DataFrame(data, columns=['worker', 'task', 'label', 'confidence'])
    else:
        df = pd.DataFrame(data, columns=['worker', 'task', 'label'])
    return df

def dataframe_to_opinions(df):
    n_workers = df['worker'].nunique()
    n_tasks = df['task'].nunique()
    workers_opinions = np.full((n_workers, n_tasks), -1, dtype=float)
    confidence = np.full((n_workers, n_tasks), 1, dtype=float)
    
    tasks = {}
    workers = {}
    task_id = 0
    worker_id = 0
    
    for index, row in df.iterrows():
        worker = row['worker']
        task = row['task']
        opinion = row['label']
    
        if worker not in workers:
            workers[worker] = worker_id
            worker_id += 1
    
        if task not in tasks:
            tasks[task] = task_id
            task_id += 1
    
        workers_opinions[workers[worker]][tasks[task]] = opinion

        if 'confidence' in row:
            confidence[workers[worker]][tasks[task]] = row['confidence']
    
    return workers_opinions, confidence

def calculate_accuracy(labels, ground_truth, threshold=0.05):
    total = 0 
    correct = 0
    for l, g in zip(labels, ground_truth):
        total += 1
        if abs(l - g) <= threshold:
            correct += 1
    return correct/total

def calculate_avg_error(labels, ground_truth):
    error = 0
    total  = 0
    for l, g in zip(labels, ground_truth):
        error += abs(l - g)
        total += 1
    return error/total
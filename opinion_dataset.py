import numpy as np
import random

def generate_dataset(n_tasks=10,n_workers=5,percetage_tasks_workers=0.9,prob_type_worker=[0,0,0,0,1,0],prob_worker=[[0.3,0.1],[0.3,0.1],[0.8,0.1],[0.8,0.1],[-1,-1],[-1,-1]],
                     confidence_worker=[[0.8,0.1],[0.3,0.1],[0.8,0.1],[0.3,0.1],[-1,0.1],[-1,-1]],hard_labels=False):
    

    probabilities = []
    labels = np.random.rand(n_tasks)
    if hard_labels:
        labels = [1 if l >=0.5 else 0 for l in labels]
    
    worker_type = []
    avg_error_workers = []
    workers_opinions = [[-1] * n_tasks for _ in range(n_workers)] #init opinions
    confidence_opinions = [[-1] * n_tasks for _ in range(n_workers)] #init confidences

    for i in range(n_workers):
        
        #define worker type #(0-Excellent/1-No_confident/2-Over_confident/3-No_accurate/4-Realistic/5-Random)
        rand_worker_type = random.random()
        sum_worker_type = 0
        for j, p in enumerate(prob_type_worker):
            sum_worker_type += p
            if rand_worker_type <= sum_worker_type:
                worker_type.append(j)
                break
        
        #define mean error of worker
        if prob_worker[worker_type[i]][0] != -1:
            avg_prob = prob_worker[worker_type[i]][0]
            std_prob = prob_worker[worker_type[i]][1]
            avg_error_worker = np.random.normal(avg_prob, std_prob, 1)[0]
            if avg_error_worker > 1:
                avg_error_worker = 1
            elif avg_error_worker < 0:
                avg_error_worker = 0

            #generate errors of worker w
            errors = np.random.normal(avg_error_worker, .2, n_tasks) 
            errors = [1 if i > 1 else i for i in errors]
            errors = [0 if i < 0 else i for i in errors]
            
        else:
            errors = np.random.rand(n_tasks)
    
        #check confidence of worker
        if confidence_worker[worker_type[i]][0] != -1: #Follow distribution provided
            avg_conf = confidence_worker[worker_type[i]][0]
            std_conf = confidence_worker[worker_type[i]][1]
            avg_confidence_worker = np.random.normal(avg_conf, std_conf, 1)[0]
            confidences = np.random.normal(avg_confidence_worker, std_conf, n_tasks) 
        elif confidence_worker[worker_type[i]][1] != -1: #Follow error
            confidences = []
            for error in errors:
                doubt = np.random.normal(error, confidence_worker[worker_type[i]][1], 1)[0]
                confidences.append(abs(1-doubt))
        else: #Random
            confidences = [random.random() for _ in range(n_tasks)]

        confidences = [1 if i > 1 else i for i in confidences]
        confidences = [0 if i < 0 else i for i in confidences]
        
        for j, (l, e) in enumerate(zip(labels, errors)): 
            if random.random() > percetage_tasks_workers: #decides if worker w gives his opinion
                workers_opinions[i][j] = -1
                confidence_opinions[i][j] = -1
            else:
                workers_opinions[i][j] = l + random.choice([e, -e]) #opinion o of worker w
                if workers_opinions[i][j] > 1:
                    workers_opinions[i][j] = 1
                elif workers_opinions[i][j] < 0:
                    workers_opinions[i][j] = 0
                confidence_opinions[i][j] = confidences[j]

    return labels, workers_opinions, confidence_opinions, probabilities
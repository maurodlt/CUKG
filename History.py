class History:
    def __init__(self, itt=100):
        self.iterations = itt
    
    def update_prediction(self, O, c_workers, familiarity):
        n_tasks = len(O[0])
        n_workers = len(O)
        l = [0]*n_tasks #infered labels
        for t in range(n_tasks):
            norm = 0
            l[t] = 0
            for w in range(n_workers):
                if O[w][t] != -1:
                    if O[w][t] >= 0.5:
                        l[t] += c_workers[w] * familiarity[w][t]
                    norm += c_workers[w] * familiarity[w][t]
            if norm != 0:
                l[t] /= norm
            else:
                l[t] = 0
    
        return l
    
    def update_workers(self, O, l):
        n_tasks = len(O[0])
        n_workers = len(O)
    
        c_workers = [1]*n_workers #confidence in workers
        for w in range(n_workers):
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for t in range(n_tasks):
                if O[w][t] != -1:
                    if O[w][t] >= 0.5:
                        tp += l[t]
                        fp += 1-l[t]
                    else:
                        tn += 1-l[t]
                        fn += l[t]
            c_workers[w] = (tp+tn) / (tp + tn + fp + fn)   
        return c_workers
    
    def run(self, O, familiarity):
        n_tasks = len(O[0])
        n_workers = len(O)
        c_workers = [1]*n_workers #confidence in workers
    
        if familiarity.size == 0:
            familiarity = [[0.5] * n_tasks] * n_workers
    
        l = []
        
        for i in range(self.iterations): # run until converges or maximum iteration count
            l = self.update_prediction(O, c_workers, familiarity)
            c_workers = self.update_workers(O, l)
    
        return l


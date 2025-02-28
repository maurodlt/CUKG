import numpy as np
from scipy.optimize import minimize

class CUKG():
    def errors_user(self, c,o,u,a):
        error = 0
        count = 0
        if a == []:
            for ci,oi,ui in zip(c,o,u):
                if ci < 0 or oi < 0:
                    continue
                error += ci*abs(oi-ui)
                count += 1
        else:
            for ci,oi,ui,ai in zip(c,o,u,a):
                if ci < 0 or oi < 0:
                    continue
                error += ci*abs(oi-ui) #*abs(oi-ai)
                count += 1
                
        return error, count
    
    def trust_user(self, c,o,u,a,beta):
        error, count = self.errors_user(c,o,u,a)
        return 1/((error/count)+beta)
    
    def trust_population(self, C, O, U, A, beta):
        if len(C) != len(O):
            return -1
    
        trust = 0
        for c, o in zip(C, O):
            trust += self.trust_user(c,o,U,A,beta)
            
        return trust
    
    def comb_prob(self, A, U):
        if len(A) != len(U):
            return -1
    
        combined_probability = 1
        for a, u in zip(A,U):
            combined_probability *= (2* a * u) - a - u + 1
    
        return combined_probability
    
    def quality(self, U,C,O,A,beta):
        if A == []:
            trust = self.trust_population(C,O,U,A,beta)
            return trust
        else:
            P = self.comb_prob(A,U)
            if P == 0:
                return 0
            trust = self.trust_population(C,O,U,A,beta)
            return trust*P
    
    def objective(self, U,C,O,A,beta): #U = initial guess, C = confidence_workers, O = opinion_workers, A = a priori prob., beta=hyperparameter
        return -self.quality(U,C,O,A,beta)
    
    def run(self, confidence_opinions, workers_opinions, probabilities=[], initial_guess=None, beta = 1):
        n_tasks = len(workers_opinions[0])
        bounds = [(0, 1)]*n_tasks

        if initial_guess==None:
            initial_guess = [0.5] * n_tasks
        
        return minimize(self.objective, x0=initial_guess, args=(confidence_opinions,workers_opinions,probabilities,beta), method='L-BFGS-B', bounds=bounds).x
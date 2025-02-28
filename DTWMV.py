from c45 import C45
import numpy as np
import copy

class DTWMV():
    def digitize_features(self, feature):
        bins = np.arange(0, 1, 0.1)  # Bins from 0 to 1 with step size 0.1
    
        feature_d = np.digitize(feature, bins) - 1
        return feature_d
    
    def reverse_digitize_features(self, feature_d):
        bins = np.arange(0, 1.1, 0.1)  # Bins from 0 to 1 with step size 0.1
        midpoints = bins[:-1] + 0.05  # Calculate midpoints of the bins
        feature_d = np.clip(feature_d, 0, len(midpoints) - 1)  # Ensure indices are within bounds
        return midpoints[feature_d]
    
    def mv(self, feature_digits, weights=[]):
        frequency = {}
        if len(weights) == 0:
            weights = [1]*len(feature_digits)
            
        for f, w in zip(feature_digits, weights):
            if f in frequency:
                frequency[f] += w
            else:
                frequency[f] = w
    
        max_value = 0
        y = 0
        for f in frequency:
            if frequency[f] > max_value:
                max_value = frequency[f]
                y = f
    
        return y, max_value
    
    def construct_tree(self, workers_opinions, attributes, Y, W):
        c2 = C45("", "")
        
        #fetch data
        data = []
        X = np.array(workers_opinions).T
        for x, y, w in zip(X, Y, W):
            d = []
            for i in x:
                d.append(i)
            d.append(str(int(y)))
            for i in range(w):
                data.append(d)
    
        c2.data = data
        c2.classes = ['0','1','2','3','4','5','6','7','8','9']
        c2.numAttributes =  len(attributes)
        c2.attributes = attributes
        c2.attrValues = {}
        for att in c2.attributes:
            c2.attrValues[att] = ['continuous']
    
        c2.preprocessData()
        c2.generateTree()
    
        return c2
    
    def predict(self, node, y, att):
        if not node.isLeaf:
            leftChild = node.children[0]
            rightChild = node.children[1]
            node_position = att.index(node.label)
            if y[node_position] <= node.threshold:
                return self.predict(leftChild, y, att)
            else:
                return self.predict(rightChild, y, att)
        else:
            return node.label
    
    def accuracy(self, c):
        total = 0
        correct = 0
        for x in c.data:
            y = x[-1]
            prediction = self.predict(c.tree, x, c.attributes)
            total += 1
            if y == prediction:
                correct += 1
        return correct/total
    
    
    def build_forest(self, workers_opinions, verbose=False):
        acc = 1
        forest = []
        wo = list(workers_opinions)
        attributes = ['w'+str(i) for i in range(len(wo))]
        
        Y = []
        W = []
        for x in np.array(wo).T:
            y, weight = self.mv(x)
            Y.append(str(int(y)))
            W.append(weight)
        
        while acc >= 0.5:
            c45 = self.construct_tree(wo,attributes, Y, W)
            forest.append(c45)
            acc = self.accuracy(c45)
            if verbose:
                print("Accuracy: ", acc, "    -Workers:", len(wo))
            if len(wo) > 1:
                if c45.tree.label not in attributes: #the root is a leaf
                    root_node_position = attributes.index(attributes[0]) #remove first attribute 
                else:
                    root_node_position = attributes.index(c45.tree.label)
                del wo[root_node_position]
                del attributes[root_node_position]
                if verbose:
                    print("Deleting node:", c45.tree.label)
                    print("Current nodes: ", attributes)
                    print("")
                
            else:
                break  
            
        return forest
    
    def weight_workers_tree(self, c45, num_workers):
        c = c45.tree
        weights = [0]*num_workers
        
        if not c.isLeaf:
            #root node
            node_id = int(c.label[1:])
            weights[node_id] += 1 
    
            #second level nodes
            leftChild = c.children[0]
            rightChild = c.children[1]
    
            if not leftChild.isLeaf:
                lefChild_id = int(leftChild.label[1:])
                weights[lefChild_id] += 1/pow(2,3)
    
                #third level nodes
                llChild = leftChild.children[0]
                lrChild = leftChild.children[1]
                if not llChild.isLeaf:
                    llChild_id = int(llChild.label[1:])
                    weights[llChild_id] += 1/pow(3,3)
    
                if not lrChild.isLeaf:
                    lrChild_id = int(lrChild.label[1:])
                    weights[lrChild_id] += 1/pow(3,3) 
    
            if not rightChild.isLeaf:
                rightChild_id = int(rightChild.label[1:])
                weights[rightChild_id] += 1/pow(2,3) 
           
                rlChild = rightChild.children[0]
                rrChild = rightChild.children[1]
                if not rlChild.isLeaf:
                    rlChild_id = int(rlChild.label[1:])
                    weights[rlChild_id] += 1/pow(3,3)
                if not rrChild.isLeaf:
                    rrChild_id = int(rrChild.label[1:])
                    weights[rrChild_id] += 1/pow(3,3) 
        return weights
    
    def weight_workers(self, forest, n_workers):
        weights = np.array([0]*n_workers, dtype=float)
        max_weight = 0
        
        for i, tree in enumerate(forest):
            weight_tree = 1/(i+1)
            weights_workers = np.array(self.weight_workers_tree(tree, n_workers))
            max_weight += weight_tree * np.sum(weights_workers) 
            weights += np.array(weight_tree * weights_workers)

        if max_weight != 0:
            weights /= max_weight
            
        return weights
    
    def run(self, O):
        n_workers = len(O)
        workers_opinions = copy.deepcopy(O)
        
        forest = self.build_forest(workers_opinions)
        weights = self.weight_workers(forest, n_workers)
    
        labels = []
        weights_labels = []
        for features in np.array(workers_opinions).T:
            digitized_feature = self.digitize_features(features)
            l, w = self.mv(digitized_feature, weights)
            labels.append(l)
            weights_labels.append(w)
            
        return self.reverse_digitize_features(labels)
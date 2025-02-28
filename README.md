# CUKG: Consolidation of Uncertain Knowledge Graphs Algorithm

The **Consolidation of Uncertain Knowledge Graphs (CUKG)** algorithm is a collaborative approach for consolidating the numerical uncertainties associated with facts in an Uncertain Knowledge Graph (UKG). It allows workers to provide their opinions on facts within the graph, along with the confidence in those opinions. The algorithm then determines the probabilities of the facts that maximize collective trust across all workers.

CUKG is especially useful in scenarios such as **uncertain knowledge graph consolidation**, **crowdsourcing**, and **conflict resolution** where multiple conflicting opinions must be reconciled to create a more reliable and confident representation of knowledge.

## Features
- Collaborative approach to consolidate opinions from multiple workers.
- Computes confidence-adjusted probabilities for facts based on worker input.
- Can be applied to crowdsourcing tasks, uncertainty reduction, and conflict resolution problems.

## Installation
```
git clone https://github.com/maurodlt/CUKG.git
cd CUKG
pip install scipy
pip install numpy
```

## Usage
To use the CUKG algorithm, simply add the *CUKG.py* file to your source directory. Then, you can run it as follows:
```
from CUKG import CUKG

workers_opinions = [
    [0.7, 0.5, 0.8],  # Worker 1's opinions
    [0.9, 0.6, 0.75], # Worker 2's opinions
    [0.6, 0.4, 0.85]  # Worker 3's opinions
]

confidence_opinions = [
    [0.9, 0.8, 0.7],  # Worker 1's confidence
    [0.7, 0.9, 0.8],  # Worker 2's confidence
    [0.8, 0.6, 0.9]   # Worker 3's confidence
]

cukg = CUKG()
result = cukg.run(confidence_opinions, workers_opinions)

print(result)

```
- *workers_opinions*: A matrix (list of lists) where each row represents a worker and each column represents the opinion on a specific fact (values between 0 and 1).
- *confidence_opinions*: A matrix (list of lists) where each row represents a worker’s confidence in their opinion about each fact (values between 0 and 1).

The run() method returns the consolidated result, which is a matrix reflecting the collective trust across all workers.

## License
This project is licensed under the MIT License - see the LICENSE file for details.



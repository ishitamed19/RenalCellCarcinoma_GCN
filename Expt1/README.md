An implementation of the paper: [Convolutional Neural Network Approach to Lung Cancer Classification Integrating Protein Interaction Network and Gene Expression Profiles](https://ieeexplore.ieee.org/document/8567475)

### Implementation
Used the HINT 4.0 PPI as mentioned in the paper. Alongside that, also used a 55-gene KEGG Pathway graph for benchmarking. 
### Results
The KEGG-Pathway, due to its small size and more connected components, was able to perform better than the PPI. 
### Model Details
Adam Optimizer; lr = 0.0001
Batch size = 16; Epochs = 25
Additional things implemented (which are not there in the paper): Prelu activation, weighted sampling

Surprisingly, this model fails to capture KICH patients, and hence has 0 accuracy for this category.

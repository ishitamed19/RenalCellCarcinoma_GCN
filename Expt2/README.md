An implementation of the paper: [Cancer subtype classification and modeling by pathway attention and propagation](https://academic.oup.com/bioinformatics/article-abstract/36/12/3818/5811233?redirectedFrom=fulltext)

### Implementation
● INPUT:
  ○ KEGG Pathways
  ○ Gene expression profile with cancer subtype 
● OUTPUT:
  ○ Subtype Classification
  ○ Importance of pathways with interaction information 
 
-[X]GCN Model (ChebNet) for each pathway (110 pathways -> 110 models) 
-[X]Attention-based ensemble model : To find out the contribution of each pathway towards predicting that patient’s class 

### Results
1. Trained an ensemble of 11 attention-based models
2. Used 5-fold cross-validation
3. Overall f1-weighted score:
96.8%




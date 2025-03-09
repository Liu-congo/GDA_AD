# GDA_AD

A repository of Adapting Graph Data Augmentation into Alzheimer Diagnoistics

The overall process include the following stage:
1. Data loading and preprocessing: mainly filtering the brain connectivity matrix to capture discriminant brain substructure fro AD diagnosing. Filtering can be threshold-based, distance-based, dynamically, learnable..., threshold-based has been implemented in src.Datasets module, others waiting for update :(
2. Data Augmentation: Implemented in src.Transforms module, applying some typical GDA method for AD data, together with a simple Mixup on Graph.(To seperate this part from model, manifold-like method will not be added)
3. Model Construction: With the help of PyG, we implement MLP,GCN,GAT with its api, but also extenable in src.Model for customized model whoever developed.
4. Model Training & Evaluating: src.Trainer construct a trainer to record every experiment we did automatically.

Environment Setting:

torch + pyg + scikit-learn + tqdm + numpy + ...(Maybe a environment.txt is needed)

TODO:
1. fixing Mask Strategy
2. change config setting update method 

Example Usage:

python main.py
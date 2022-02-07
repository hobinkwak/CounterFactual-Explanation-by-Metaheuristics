## Counterfactual WOA and GA
- Interpretation of Black-Box Model with Counterfactual Instance generated by Metaheuristics
  - used Whale Optimization Algorithm and Genetic Algorithm

### Concept

- Counterfactual
- Generate the most similar input data (features) with the most different predictions
  - the more changes (of input features), the more important
- Objective Function
  - **min**imize difference between a sample (which we want to interprete) **features** and a generated one
  - **max**imize difference between a sample (which we want to interprete) **targets** and a generated one
  - I used RMSE but you can use Cosine Similarity considering your input data distribution


### Data

- N_features : 10 
  - Not Informative Features : 4, 6, 9 (index)
- N_targets : 3
- This data was generated by using sklearn's make_regression function

### Model

- simple DNN
  - 5 fc layers
 
### Requirements
- geneticalgorithm==1.0.2
- matplotlib==3.5.1
- numpy==1.20.0
- pandas==1.3.4
- scikit-learn==1.0.1
- scipy==1.7.3
- torch==1.10.0

```bash
pip install -r requirements.txt
```

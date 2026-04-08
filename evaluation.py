import numpy as np

def rmse(model, ratings):
  errors = [(r - model.predict(u, i)) ** 2 for u, i, r in ratings]
  return np.sqrt(np.mean(errors))
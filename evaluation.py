import numpy as np

def rmse(model, ratings):
  errors = [(r - model.predict(u, i)) ** 2 for u, i, r in ratings]
  return np.sqrt(np.mean(errors))

# decaying learning rate (equation 2.6)
def decay_learning_rate(k, K1, alpha_0, alpha_K1) -> int:
  alpha_k = (1 - k / K1) * alpha_0 + (k / K1) * alpha_K1

  return alpha_k
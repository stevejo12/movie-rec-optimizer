import numpy as np

# stochastic gradient descent
# @Param
# model: the model -> Matrix Factorization class
# train_ratings -> the dataset for training
# epoch -> number of loop of iterating the data
#   1 epoch => for example 80k data means be 80k iterations in a epoch
# alpha_0 -> intial learning rate
# alpha_K1 ->
def stochastic_gd(model, train_ratings, epochs=20, alpha_0=0.01, alpha_K1=0.001):
  history = []
  train_list = list(train_ratings)
  K1 = epochs * len(train_list)
  k = 0

  print("=== Stochastic Gradient Descent Iteration ===")
  for epoch in range(epochs):
    # make sure that the list order checked randomized every epoch
    np.random.shuffle(train_list)

    for u, i, r in train_list:
      # decaying learning rate (equation 2.6)
      alpha_k = (1 - k / K1) * alpha_0 + (k / K1) * alpha_K1

      # user_id and movie_id starts with 1 (not 0)
      # want to point to index #
      u_idx = u - 1
      i_idx = i - 1

      error = r - model.U[u_idx] @ model.V[i_idx]

      u_old = model.U[u_idx].copy()
      model.U[u_idx] += alpha_k * (error * model.V[i_idx] - model.lam * model.U[u_idx])
      model.V[i_idx] += alpha_k * (error * u_old - model.lam * model.V[i_idx])

      k += 1

    loss = model.loss(train_list)
    history.append(loss)
    print(f"Epoch {epoch+1}/{epochs}: loss = {loss:.4f}")

  return history
import numpy as np
import math
from evaluation import decay_learning_rate
from scipy.optimize import minimize

# stochastic gradient descent
# @Param
# model: the model -> Matrix Factorization class
# train_ratings -> the dataset for training
# epoch -> number of loop of iterating the data
#   1 epoch => for example 80k data means be 80k iterations in a epoch
# alpha_0 -> intial learning rate
# alpha_K1 -> final learning rate
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
      alpha_k = decay_learning_rate(k,K1,alpha_0, alpha_K1)

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

# Gradient Descent + BackTracking Line Search
# @Param
# model: the model -> Matrix Factorization class
# train_ratings -> the dataset for training
# alpha -> initial step size
# beta -> shrink factor
# sigma -> sufficient decrease parameter
def gd_btls(model, train_ratings: list[tuple[int, int, int]], alpha, beta, sigma):
  history = []
  train_list = list(train_ratings)
  max_iter = 100

  print("=== Gradient Descent with Backtracking Line Search Iteration ===")
  alpha_k = alpha
  for iteration in range(max_iter):
    grad_U, grad_V = model.gradients(train_ratings) 
    grad_norm_sq = np.sum(grad_U ** 2) + np.sum(grad_V ** 2)
    old_loss = model.loss(train_list)
    U_old, V_old = model.get_params()

    # backtracking line search
    while True:
      model.U = U_old - alpha_k * grad_U
      model.V = V_old - alpha_k * grad_V
      new_loss = model.loss(train_list)

      if new_loss <= (old_loss - sigma * alpha_k * grad_norm_sq):
        break
      else:
        alpha_k *= beta
        # revert back to the original
        model.set_params(U_old, V_old)

    history.append(new_loss)
    print(f"Epoch {iteration+1}/{max_iter}: loss = {new_loss:.4f}")

  return history

# Mini Batch Gradient Descent
def mini_batch_gd(model, train_ratings, batch_size, epochs=20, alpha_0=0.01, alpha_K1=0.001):
  history = []
  train_list = list(train_ratings)
  k = 0
  b = math.ceil(len(train_list) / batch_size) # total batch
  K1 = epochs * b

  print("=== Mini Batch Gradient Descent Iteration ===")
  for epoch in range(epochs):
    # make sure that the list order checked randomized every epoch
    np.random.shuffle(train_list)

    for n in range(b):
      lower_range = n * batch_size
      upper_range = lower_range + batch_size
      batch = train_list[lower_range:upper_range]

      grad_U, grad_V = model.gradients(batch)
      grad_U /= len(batch)  # normalize by batch size
      grad_V /= len(batch)
      alpha_k = decay_learning_rate(k,K1,alpha_0, alpha_K1)

      model.U -= alpha_k * grad_U
      model.V -= alpha_k * grad_V

      k += 1
    
    loss = model.loss(train_list)
    history.append(loss)
    print(f"Epoch {epoch+1}/{epochs}: loss = {loss:.4f}") 

  return history

# @Parameter
def bfgs(model, train_ratings, max_iter=50):
  history = []
  train_list = list(train_ratings)

  def loss_and_grad(flat_vector):
    # unflatten into U and V
    U = flat_vector[:model.n_users * model.k].reshape(model.n_users, model.k)
    V = flat_vector[model.n_users * model.k:].reshape(model.n_items, model.k)
    
    # set on model
    model.set_params(U, V)
    
    # compute loss and gradients using your existing methods
    loss = model.loss(train_list)
    grad_U, grad_V = model.gradients(train_list)
    
    # flatten gradients back into one vector
    flat_grad = np.concatenate([grad_U.flatten(), grad_V.flatten()])
    
    return loss, flat_grad
  
  def callback(flat_vector):
    loss = model.loss(train_list)
    history.append(loss)
    print(f"Iteration {len(history)}/{max_iter}: loss = {loss:.4f}")

  print("=== BFGS Algorithm Iteration ===")
  flatten_matrix = np.concatenate([model.U.ravel(), model.V.ravel()])

  result = minimize(
    loss_and_grad,
    flatten_matrix,
    method='L-BFGS-B', # (Limited-memory) BFGS (Bounded) -> note: bounded is not used here.
    jac=True,          # loss_and_grad returns both loss and gradient
    callback=callback,
    options={'maxiter': max_iter, 'disp': True}
  )
  
  # unflatten the matrix
  split = model.n_users * model.k
  U = result.x[:split].reshape(model.n_users, model.k)
  V = result.x[split:].reshape(model.n_items, model.k)

  model.set_params(U,V)

  return history
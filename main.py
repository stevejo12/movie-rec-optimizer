import time
from dataloader import DataLoader
from matrix_factorization import MatrixFactorization
from optimizer import stochastic_gd, gd_btls, mini_batch_gd, bfgs
from evaluation import rmse

loader = DataLoader(filepath="u.data", seed=42)
loader.load()

n_users = max(u for u, _, _ in loader.train_ratings + loader.test_ratings)
n_items = max(m for _, m, _ in loader.train_ratings + loader.test_ratings)
mf_model = MatrixFactorization(n_users=n_users, n_items=n_items, k=10, lam=0.01)
U_init, V_init = mf_model.get_params()

mf_model.set_params(U_init, V_init)
start_sgd = time.time()
history = stochastic_gd(mf_model, loader.train_ratings, epochs=20)
elapsed_sgd = time.time() - start_sgd

test_rmse = rmse(mf_model, loader.test_ratings)
print(f"Test RMSE for Stochastic Gradient Descent: {test_rmse:.4f}")
print(f"Time taken to run Stochastic Gradient Descent: {elapsed_sgd:.2f}")

mf_model.set_params(U_init, V_init)
start_gdbtls = time.time()
gd_btls(mf_model, loader.train_ratings, 1, 0.5, 0.1)
elapsed_gdbtls = time.time() - start_gdbtls

test_rmse = rmse(mf_model, loader.test_ratings)
print(f"Test RMSE for Gradient Descent with BackTracking Line Search: {test_rmse:.4f}")
print(f"Time taken to run Gradient Descent with BackTracking Line Search: {elapsed_gdbtls:.2f}")

mf_model.set_params(U_init, V_init)
start_mb = time.time()
mini_batch_gd(mf_model, loader.train_ratings, 256, 100, 1.0, 0.01)
elapsed_mb = time.time() - start_mb

test_rmse = rmse(mf_model, loader.test_ratings)
print(f"Test RMSE for Mini Batch Gradient Descent: {test_rmse:.4f}")
print(f"Time taken to run Mini Batch Gradient Descent: {elapsed_mb:.2f}")

mf_model.set_params(U_init, V_init)
start_bfgs = time.time()
bfgs(mf_model, loader.train_ratings, 50)
elapsed_bfgs = time.time() - start_bfgs

test_rmse = rmse(mf_model, loader.test_ratings)
print(f"Test RMSE for BFGS Algorithm: {test_rmse:.4f}")
print(f"Time taken to run BFGS Algorithm: {elapsed_bfgs:.2f}")
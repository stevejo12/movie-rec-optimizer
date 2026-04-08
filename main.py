from dataloader import DataLoader
from matrix_factorization import MatrixFactorization
from optimizer import stochastic_gd

loader = DataLoader(filepath="u.data", seed=42)
loader.load()

n_users = max(u for u, _, _ in loader.train_ratings + loader.test_ratings)
n_items = max(m for _, m, _ in loader.train_ratings + loader.test_ratings)
mf_model = MatrixFactorization(n_users=n_users, n_items=n_items, k=10, lam=0.01)

print(mf_model.get_params())

history = stochastic_gd(mf_model, loader.train_ratings, epochs=20)
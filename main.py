from dataloader import DataLoader
from matrix_factorization import MatrixFactorization

loader = DataLoader(filepath="u.data", seed=42)
loader.load()

n_users = max(u for u, _, _ in loader.train_ratings + loader.test_ratings)
n_items = max(m for _, m, _ in loader.train_ratings + loader.test_ratings)
mf = MatrixFactorization(n_users=n_users, n_items=n_items, k=10, lam=0.01)

print(mf.get_params())
# print(loader.train_ratings)
# print(loader.test_ratings)
import numpy as np

class MatrixFactorization:
  def __init__(
    self,
    n_users: int,
    n_items: int,
    lam: int, #lambda
    seed: int = 42,
    k: int = 10, # hyperparameter / latent features
  ):
    if k <= 0:
      raise ValueError("k must be a positive integer")
    
    self.n_users = n_users
    self.n_items = n_items
    self.k = k
    self.lam = lam

    # random number generator
    rng = np.random.default_rng(seed)
    # mean:0 -> make it not bias, 
    # std deviation: 0.01 -> keep it small initially
    self.U = rng.normal(0, 0.01, (n_users, k)) 
    self.V = rng.normal(0, 0.01, (n_items, k))

  def get_params(self) -> tuple[np.ndarray, np.ndarray]:
    return self.U.copy(), self.V.copy()
  
  def set_params(self, U: np.ndarray, V: np.ndarray) -> None:
    expected_U = (self.n_users, self.k)
    expected_V = (self.n_items, self.k)
    if U.shape != expected_U:
      raise ValueError(f"U shape mismatch: expected {expected_U}, got {U.shape}")
    if V.shape != expected_V:
      raise ValueError(f"V shape mismatch: expected {expected_V}, got {V.shape}")
    self.U = U.copy()
    self.V = V.copy()

  def predict(self, user_id: int, movie_id: int) -> float:
    if user_id < 1 or user_id > self.n_users:
      raise IndexError(f"user_id {user_id} out of bounds [1, {self.n_users}]")
    if movie_id < 1 or movie_id > self.n_items:
      raise IndexError(f"movie_id {movie_id} out of bounds [1, {self.n_items}]")
    raw = self.U[user_id - 1] @ self.V[movie_id - 1]
    # rating can be only between 1 and 5
    # this make sure the raw values within the range
    return float(np.clip(raw, 1.0, 5.0))

  def loss(self, ratings: list[tuple[int, int, int]]) -> float:
    if not ratings:
      return 0.0
    sq_err = 0.0
    for u, i, r in ratings:
      pred = self.U[u - 1] @ self.V[i - 1]
      sq_err += (r - pred) ** 2
    reg = self.lam * (np.sum(self.U ** 2) + np.sum(self.V ** 2))
    return float(sq_err + reg)

  def gradients(
    self, ratings: list[tuple[int, int, int]]
  ) -> tuple[np.ndarray, np.ndarray]:
    grad_U = np.zeros_like(self.U)
    grad_V = np.zeros_like(self.V)
    if not ratings:
      return grad_U, grad_V
    grad_U += 2 * self.lam * self.U
    grad_V += 2 * self.lam * self.V
    for u, i, r in ratings:
      u_idx = u - 1
      i_idx = i - 1
      err = r - self.U[u_idx] @ self.V[i_idx]
      grad_U[u_idx] += -2 * err * self.V[i_idx]
      grad_V[i_idx] += -2 * err * self.U[u_idx]
    return grad_U, grad_V

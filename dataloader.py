from typing import List, Tuple
import numpy as np
import scipy.sparse

class DataLoader:
  # initialize variables to the environment
  def __init__(self, filepath: str = "u.data", seed: int = 42) -> None:
    self._filepath = filepath
    self._seed = seed
    self._train: List[Tuple[int, int, int]] = []
    self._test: List[Tuple[int, int, int]] = []
    self._sparse: scipy.sparse.csr_matrix | None = None

  def load(self):
    rows: List[Tuple[int, int, int]] = []
    with open("u.data", 'r') as f:
      # iterate the data inside
      for line_no, line in enumerate(f, start=1):
        line = line.rstrip("\n")
        parts = line.split("\t")
        
        # log if the line has exactly "not 4" values
        if len(parts) != 4:
          raise ValueError(
            f"Malformed row at line {line_no}: expected 4 columns, "
            f"got {len(parts)} — '{line}'"
          )
        
        try:
          user_id = int(parts[0])
          movie_id = int(parts[1])
          rating = int(parts[2])
          # timestamp = (parts[3]) # ignore this for now (not needed)
        except ValueError:
          raise ValueError(
            f"Non-integer value at line {line_no}: '{line}'"
          )
        
        # check if values are other than 1,2,3,4,5
        if rating not in {1, 2, 3, 4, 5}:
          raise ValueError(
            f"Rating out of range at line {line_no}: "
            f"got {rating}, expected one of {{1,2,3,4,5}} "
            f"(user_id={user_id}, movie_id={movie_id})"
          )
        
        # output
        rows.append((user_id, movie_id, rating))

      # Shuffle with fixed seed for reproducibility
      rng = np.random.default_rng(self._seed)
      indices = np.arange(len(rows))
      rng.shuffle(indices)
      shuffled = [rows[i] for i in indices]

      # splitting training data 80/20
      split = int(len(shuffled) * 0.8)
      self._train = shuffled[:split]
      self._test = shuffled[split:]

      # u-1 because userid and itemsid start with 1 -> index starts at 0
      users = np.array([u - 1 for u, _, _ in self._train], dtype=np.int32)
      items = np.array([i - 1 for _, i, _ in self._train], dtype=np.int32)
      ratings = np.array([r for _, _, r in self._train], dtype=np.float64)

      # store sparse
      n_users = max(u for u, _, _ in self._train + self._test) 
      n_items = max(m for _, m, _ in self._train + self._test)
      self._sparse = scipy.sparse.csr_matrix(
        (ratings, (users, items)), shape=(n_users, n_items)
      )

  # Properties to call
  @property
  def train_ratings(self) -> List[Tuple[int, int, int]]:
    self._require_loaded()
    return self._train

  @property
  def test_ratings(self) -> List[Tuple[int, int, int]]:
    self._require_loaded()
    return self._test

  @property
  def sparse_matrix(self) -> scipy.sparse.csr_matrix:
    self._require_loaded()
    return self._sparse

  # load to check if value is still valid
  def _require_loaded(self) -> None:
    if not self._train and not self._test:
      raise RuntimeError(
        "Data has not been loaded yet. Call load() first."
      )




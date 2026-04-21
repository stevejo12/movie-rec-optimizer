# Movie Recommendation via Matrix Factorization

A comparison of optimization algorithms for collaborative filtering on the MovieLens 100K dataset. Course project for ECE 503 — Optimization for Machine Learning at the University of Victoria.

## Overview

This project formulates movie recommendation as a matrix factorization problem, decomposing the sparse user-movie rating matrix into two low-rank latent factor matrices. Four optimization algorithms are implemented and compared:

- **Stochastic Gradient Descent (SGD)** — per-sample updates with decaying learning rate
- **Gradient Descent with Backtracking Line Search (GD+BTLS)** — full gradient with Armijo condition
- **Mini-Batch Gradient Descent** — batch gradient with normalized updates
- **L-BFGS** — quasi-Newton method via scipy

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings (1–5) from 943 users on 1,682 movies.

## Project Structure

```
├── data_loader.py            # Load and split MovieLens data
├── matrix_factorization.py   # MF model (predict, loss, gradients)
├── optimizers.py             # SGD, GD+BTLS, Mini-Batch, L-BFGS
├── evaluation.py             # RMSE computation
├── main.py                   # Run experiments
├── generate_plots.py         # Convergence plots
└── u.data                    # MovieLens 100K ratings
```

## Usage

```bash
python main.py
```

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib

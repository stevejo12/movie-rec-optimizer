import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

def graph_plot(sgd_loss, sgd_time, gd_btls_loss, gd_btls_time, mini_batch_loss, mini_batch_time, lbfgs_loss, lbfgs_time):
  # ── Plot 1: Loss vs Epoch/Iteration ──
  fig, ax = plt.subplots(figsize=(8, 5))

  ax.plot(range(1, len(sgd_loss)+1), sgd_loss, 'o-', label='SGD (20 epochs)', markersize=3, linewidth=1.5)
  ax.plot(range(1, len(gd_btls_loss)+1), gd_btls_loss, 's-', label='GD+BTLS (100 iter)', markersize=2, linewidth=1.5)
  ax.plot(range(1, len(mini_batch_loss)+1), mini_batch_loss, '^-', label='Mini-batch (100 epochs)', markersize=2, linewidth=1.5)
  ax.plot(range(1, len(lbfgs_loss)+1), lbfgs_loss, 'D-', label='L-BFGS (50 iter)', markersize=2, linewidth=1.5)

  ax.set_xlabel('Epoch / Iteration')
  ax.set_ylabel('Training Loss')
  ax.set_title('Convergence Comparison: Loss vs. Epoch/Iteration')
  ax.legend()
  ax.set_yscale('log')
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('./convergence_epoch.png', dpi=200)
  plt.close()

  # ── Plot 2: Loss vs Time ──
  fig, ax = plt.subplots(figsize=(8, 5))

  # Create time arrays (linearly spaced from 0 to total time)
  sgd_times = np.linspace(0, sgd_time, len(sgd_loss))
  gd_btls_times = np.linspace(0, gd_btls_time, len(gd_btls_loss))
  mini_batch_times = np.linspace(0, mini_batch_time, len(mini_batch_loss))
  lbfgs_times = np.linspace(0, lbfgs_time, len(lbfgs_loss))

  ax.plot(sgd_times, sgd_loss, 'o-', label='SGD', markersize=3, linewidth=1.5)
  ax.plot(gd_btls_times, gd_btls_loss, 's-', label='GD+BTLS', markersize=2, linewidth=1.5)
  ax.plot(mini_batch_times, mini_batch_loss, '^-', label='Mini-batch', markersize=2, linewidth=1.5)
  ax.plot(lbfgs_times, lbfgs_loss, 'D-', label='L-BFGS', markersize=2, linewidth=1.5)

  ax.set_xlabel('Time (seconds)')
  ax.set_ylabel('Training Loss')
  ax.set_title('Convergence Comparison: Loss vs. Time')
  ax.legend()
  ax.set_yscale('log')
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('./convergence_time.png', dpi=200)
  plt.close()

  print("Plots saved!")
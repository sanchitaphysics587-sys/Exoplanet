import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

# 1. NASA Kepler Archive se data search aur download
target_star = "Kepler-10"
print(f"Searching data for {target_star}...")
search_result = lk.search_lightcurve(target_star, author="Kepler", quarter=5)
lc = search_result.download()

# 2. Pre-processing: Outliers aur noise hatana
lc_clean = lc.remove_nans().remove_outliers(sigma=5)
lc_flat = lc_clean.flatten(window_length=401)  # Detrending

# 3. Periodicity Search: Box-Least Squares (BLS)
period_grid = np.linspace(0.5, 5, 5000)
bls = lc_flat.to_periodogram(method="bls", period=period_grid)
best_period = bls.period_at_max_power
print(f"\n[DETECTED] Most probable orbital period: {best_period:.4f}")

# 4. Graphs Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Detrended Flux Time Series
lc_flat.scatter(ax=ax1, color='black', s=1, label='Detrended Flux')
ax1.set_title(f"{target_star} - Detrended Time Series Data")

# Plot 2: Phase Folded Light Curve
lc_folded = lc_flat.fold(period=best_period)
lc_folded.scatter(ax=ax2, color='blue', s=2, label=f'Folded at P = {best_period:.4f} days')
ax2.set_title("Phase Folded Light Curve (Transit Signal Visualized)")

plt.tight_layout()
plt.show()

"""
Visual explanation of entry filters for the MA50 + Momentum strategy.
Run: python docs/visualize_filters.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Entry Filter Examples - MA50 + Momentum Strategy', fontsize=16, fontweight='bold')

days = np.arange(60)

# =============================================================================
# GOOD EXAMPLE: Stock passes all filters
# =============================================================================
ax1 = axes[0, 0]

# Create uptrending price data
np.random.seed(42)
base_trend = 1000 + days * 3  # Gentle uptrend
noise = np.random.randn(60) * 15
price_good = base_trend + noise

# Calculate MA50
ma50_good = np.convolve(price_good, np.ones(50)/50, mode='valid')
ma50_good = np.concatenate([np.full(49, np.nan), ma50_good])

# Plot
ax1.plot(days, price_good, 'b-', linewidth=2, label='Price')
ax1.plot(days, ma50_good, 'orange', linewidth=2, linestyle='--', label='MA50')

# Highlight current price point
current_price = price_good[-1]
current_ma50 = ma50_good[-1]
dist50_pct = (current_price - current_ma50) / current_ma50 * 100

ax1.scatter([days[-1]], [current_price], color='green', s=150, zorder=5, edgecolors='black', linewidths=2)
ax1.axhline(y=current_ma50, color='orange', linestyle=':', alpha=0.5)

# Annotations
ax1.annotate(f'Today: {current_price:.0f}\n({dist50_pct:.1f}% above MA50)',
             xy=(days[-1], current_price), xytext=(days[-1]-15, current_price+30),
             fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='green'))

ax1.annotate(f'MA50: {current_ma50:.0f}',
             xy=(days[-1], current_ma50), xytext=(days[-1]-10, current_ma50-40),
             fontsize=10, ha='center', color='orange')

ax1.set_title('GOOD: Passes All Filters', fontsize=12, fontweight='bold', color='green')
ax1.set_xlabel('Days')
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Add checkmarks
checks = [
    "Above MA50: Yes",
    f"Distance ({dist50_pct:.1f}%) < 15%: Yes",
    "MA50 Rising: Yes"
]
ax1.text(0.02, 0.98, '\n'.join(checks), transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# =============================================================================
# BAD EXAMPLE 1: Price below MA50
# =============================================================================
ax2 = axes[0, 1]

# Create price that dropped below MA50
np.random.seed(123)
base_trend2 = 1000 + days * 2
noise2 = np.random.randn(60) * 15
price_bad1 = base_trend2 + noise2
price_bad1[-10:] = price_bad1[-10] - np.arange(10) * 5  # Drop at the end

ma50_bad1 = np.convolve(price_bad1, np.ones(50)/50, mode='valid')
ma50_bad1 = np.concatenate([np.full(49, np.nan), ma50_bad1])

ax2.plot(days, price_bad1, 'b-', linewidth=2, label='Price')
ax2.plot(days, ma50_bad1, 'orange', linewidth=2, linestyle='--', label='MA50')

current_price2 = price_bad1[-1]
current_ma50_2 = ma50_bad1[-1]
dist50_pct2 = (current_price2 - current_ma50_2) / current_ma50_2 * 100

ax2.scatter([days[-1]], [current_price2], color='red', s=150, zorder=5, edgecolors='black', linewidths=2)
ax2.axhline(y=current_ma50_2, color='orange', linestyle=':', alpha=0.5)

# Shade the "danger zone" below MA50
ax2.fill_between(days[49:], ma50_bad1[49:], price_bad1.min()-20, alpha=0.2, color='red', label='Below MA50 (Danger)')

ax2.annotate(f'Today: {current_price2:.0f}\n({dist50_pct2:.1f}% BELOW MA50)',
             xy=(days[-1], current_price2), xytext=(days[-1]-15, current_price2-40),
             fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='red'))

ax2.set_title('BAD: Price Below MA50 (Downtrend)', fontsize=12, fontweight='bold', color='red')
ax2.set_xlabel('Days')
ax2.set_ylabel('Price')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

checks2 = [
    "Above MA50: NO",
    "REJECTED - In downtrend"
]
ax2.text(0.02, 0.98, '\n'.join(checks2), transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# =============================================================================
# BAD EXAMPLE 2: Overextended (too far above MA50)
# =============================================================================
ax3 = axes[1, 0]

np.random.seed(456)
base_trend3 = 1000 + days * 2
noise3 = np.random.randn(60) * 10
price_bad2 = base_trend3 + noise3
price_bad2[-5:] = price_bad2[-5] + np.array([20, 50, 80, 100, 120])  # Spike up

ma50_bad2 = np.convolve(price_bad2, np.ones(50)/50, mode='valid')
ma50_bad2 = np.concatenate([np.full(49, np.nan), ma50_bad2])

ax3.plot(days, price_bad2, 'b-', linewidth=2, label='Price')
ax3.plot(days, ma50_bad2, 'orange', linewidth=2, linestyle='--', label='MA50')

current_price3 = price_bad2[-1]
current_ma50_3 = ma50_bad2[-1]
dist50_pct3 = (current_price3 - current_ma50_3) / current_ma50_3 * 100

ax3.scatter([days[-1]], [current_price3], color='red', s=150, zorder=5, edgecolors='black', linewidths=2)
ax3.axhline(y=current_ma50_3, color='orange', linestyle=':', alpha=0.5)
ax3.axhline(y=current_ma50_3 * 1.15, color='red', linestyle=':', alpha=0.5, label='15% limit')

# Shade overextended zone
ax3.fill_between(days[49:], ma50_bad2[49:] * 1.15, current_price3 + 50, alpha=0.2, color='red')

ax3.annotate(f'Today: {current_price3:.0f}\n({dist50_pct3:.1f}% above MA50)',
             xy=(days[-1], current_price3), xytext=(days[-1]-15, current_price3+20),
             fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='red'))

ax3.annotate('15% limit', xy=(days[-1], current_ma50_3 * 1.15), xytext=(days[-5], current_ma50_3 * 1.15 + 10),
             fontsize=9, color='red')

ax3.set_title('BAD: Overextended (>15% above MA50)', fontsize=12, fontweight='bold', color='red')
ax3.set_xlabel('Days')
ax3.set_ylabel('Price')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

checks3 = [
    "Above MA50: Yes",
    f"Distance ({dist50_pct3:.1f}%) < 15%: NO",
    "REJECTED - Too extended, risky"
]
ax3.text(0.02, 0.98, '\n'.join(checks3), transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# =============================================================================
# BAD EXAMPLE 3: MA50 Falling (weak trend)
# =============================================================================
ax4 = axes[1, 1]

np.random.seed(789)
# Create declining trend
base_trend4 = 1200 - days * 2  # Downward slope
noise4 = np.random.randn(60) * 15
price_bad3 = base_trend4 + noise4
# Make current price slightly above MA50 but MA50 is falling
price_bad3[-1] = price_bad3[-1] + 20

ma50_bad3 = np.convolve(price_bad3, np.ones(50)/50, mode='valid')
ma50_bad3 = np.concatenate([np.full(49, np.nan), ma50_bad3])

ax4.plot(days, price_bad3, 'b-', linewidth=2, label='Price')
ax4.plot(days, ma50_bad3, 'orange', linewidth=2, linestyle='--', label='MA50 (FALLING)')

current_price4 = price_bad3[-1]
current_ma50_4 = ma50_bad3[-1]
dist50_pct4 = (current_price4 - current_ma50_4) / current_ma50_4 * 100

# Calculate slope
ma50_5d_ago = ma50_bad3[-6]
slope = (current_ma50_4 - ma50_5d_ago) / ma50_5d_ago * 100

ax4.scatter([days[-1]], [current_price4], color='red', s=150, zorder=5, edgecolors='black', linewidths=2)

# Draw arrow showing MA50 direction
ax4.annotate('', xy=(days[-1], current_ma50_4), xytext=(days[-10], ma50_bad3[-10]),
             arrowprops=dict(arrowstyle='->', color='red', lw=3))
ax4.text(days[-5], ma50_bad3[-5] + 20, 'MA50 falling!', fontsize=10, color='red', fontweight='bold')

ax4.annotate(f'Today: {current_price4:.0f}\n({dist50_pct4:.1f}% above MA50)',
             xy=(days[-1], current_price4), xytext=(days[-1]-15, current_price4+30),
             fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='orange'))

ax4.set_title('BAD: MA50 Slope Falling (Weak Trend)', fontsize=12, fontweight='bold', color='red')
ax4.set_xlabel('Days')
ax4.set_ylabel('Price')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

checks4 = [
    "Above MA50: Yes",
    f"Distance ({dist50_pct4:.1f}%) < 15%: Yes",
    f"MA50 Rising: NO (slope={slope:.2f}%)",
    "REJECTED - Trend weakening"
]
ax4.text(0.02, 0.98, '\n'.join(checks4), transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# =============================================================================
# Final adjustments
# =============================================================================
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Add summary at bottom
summary_text = """
SUMMARY: A stock must pass ALL 3 filters to be considered:
1. Price ABOVE MA50 (uptrend)  |  2. Distance < 15% (not overextended)  |  3. MA50 not falling (healthy trend)
"""
fig.text(0.5, 0.01, summary_text, ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Save and show
output_path = 'docs/entry_filters_explained.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Chart saved to: {output_path}")

plt.show()

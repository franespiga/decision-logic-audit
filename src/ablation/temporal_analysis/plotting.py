import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional




# PLOTTING FUNCTIONS
# Occlusion curves by regime  -----------------------------------------------------------------
def plot_occlusion_curves_by_regime(
    occlusion_results: Dict,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot temporal occlusion curves by regime with 95% confidence intervals.
    """
    if 'regime_results' not in occlusion_results:
        print("No regime results available for plotting")
        return

    fig, axes = plt.subplots(figsize=figsize)


    lag_positions = occlusion_results['lag_positions'].numpy()
    regime_results = occlusion_results['regime_results']

    colors = kwargs.get('colors',
                        dict(zip(regime_results.keys(),
                                 plt.cm.Set1(np.linspace(0, 1, len(regime_results))))))


    for i, (regime_name, regime_data) in enumerate(regime_results.items()):
        # Average across action dimensions for visualization
        mean_curve = regime_data['mean_curve'].mean(dim=0).detach().numpy()
        std_curve = regime_data['std_curve'].mean(dim=0).detach().numpy()

        # 95% confidence interval
        n_samples = regime_data['n_samples']
        se_curve = std_curve / np.sqrt(n_samples)
        ci_lower = mean_curve - 1.96 * se_curve
        ci_upper = mean_curve + 1.96 * se_curve

        # Plot mean curve
        axes.plot(lag_positions, mean_curve, label=f'{regime_name} (n={n_samples})',
                  color=colors[regime_name], linewidth=2)

        # Plot confidence interval
        axes.fill_between(lag_positions, ci_lower, ci_upper, alpha=kwargs.get('alpha', 0.1), color=colors[regime_name])

    axes.set_xlabel('Lag (timesteps)')
    axes.set_ylabel('Occlusion Effect (Attribution)')
    axes.set_title('Temporal Occlusion Curves by Regime (95% CI)')
    axes.legend()
    axes.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/occlusion_curves_by_regime.png", dpi=800, bbox_inches='tight')

    plt.show()


# Half-life ----------------------------------------------------------------------------------
def plot_half_life_comparison(
    half_life_results: Dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot bar chart comparing temporal half-life across regimes.
    """
    if 'regime_stats' not in half_life_results:
        print("No regime half-life results available for plotting")
        return

    regime_data = half_life_results['regime_stats']
    regime_names = list(regime_data.keys())
    mean_half_lives = [regime_data[name]['mean_half_life'] for name in regime_names]
    std_half_lives = [regime_data[name]['std_half_life'] for name in regime_names]

    fig, ax = plt.subplots(figsize=figsize)

    colors = kwargs.get('colors',
                        dict(zip(regime_names,
                                 plt.cm.Set1(np.linspace(0, 1, len(regime_names))))))

    x_pos = np.arange(len(regime_names))
    bars = ax.bar(x_pos, mean_half_lives, yerr=std_half_lives, capsize=5,
                  alpha=0.3, color=[colors[name] for name in regime_names])

    ax.set_xlabel('Market Regime')
    ax.set_ylabel('Temporal Half-life L1/2 (lags)')
    ax.set_title('Temporal Half-life Comparison Across Regimes')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regime_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, mean_half_lives, std_half_lives)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.1,
                f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/half_life_comparison.png", dpi=800, bbox_inches='tight')

    plt.show()


# Volatility-elasticity maps -----------------------------------------------------------------
# ----------------------- Figure 1: Volatility scatterplot ----------------------
def plot_volatility_elasticity_scatter(
    elasticity_results: Dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot scatter of realized volatility vs. short-term attribution share.
    """
    if 'aggregate_regression' not in elasticity_results:
        print("No aggregate regression results available for plotting")
        return

    agg_data = elasticity_results['aggregate_regression']
    x_data = agg_data['x_data']
    y_data = agg_data['y_data']
    y_pred = agg_data['y_pred']

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(x_data, y_data, alpha=0.6, s=30, color='steelblue', label='Data points')

    # Regression line
    ax.plot(x_data, y_pred, color='red', linewidth=2, label='Regression line')

    # Add regression equation and R²
    coef = agg_data['coefficient']
    intercept = agg_data['intercept']
    r2 = agg_data['r_squared']

    equation_text = f'y = {coef:.4f}x + {intercept:.4f}\nR² = {r2:.3f}'
    ax.text(0.05, 0.95, equation_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Realized Volatility')
    ax.set_ylabel('Share of Attribution in Short Horizons')
    ax.set_title(f'Volatility-Elasticity of Temporal Focus\n(Short horizons: lags {elasticity_results["short_horizon_lags"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/volatility_elasticity_scatter.png", dpi=800, bbox_inches='tight')

    plt.show()

# ----------------------- Figure 2: Coef + CI per action ----------------------
def plot_action_elasticity_boxplot(df_elasticity,
                                   save_path: Optional[str] = None):

    plt.figure(figsize=(9, 4.8))
    x = np.arange(len(df_elasticity))
    y = df_elasticity["beta"].values
    yerr = [
        y - df_elasticity["ci_low"].values,
        df_elasticity["ci_high"].values - y
    ]
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", linewidth=1, capsize=3)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(x, df_elasticity["action"].tolist(), rotation=0)
    plt.ylabel("β (elasticity of short-lag share to vol)")
    plt.title("Volatility–elasticity per action (coefficient with 95% CI)")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/vol_elasticity_coef_ci.png", dpi=800, bbox_inches='tight')
    plt.show()


# -------------------- Figure 3: Scatter of β vs. R² per action ---------------
def plot_action_level_elasticity(df_elasticity,     save_path: Optional[str] = None):
    plt.figure(figsize=(7.5, 5))
    plt.scatter(df_elasticity["beta"].values, df_elasticity["r2"].values, s=40)
    for i, row in df_elasticity.iterrows():
        plt.text(row["beta"], row["r2"], row["action"], fontsize=8, ha="left", va="bottom")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("β (elasticity)")
    plt.ylabel("R²")
    plt.title("Action-level elasticity: coefficient vs. explanatory power")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/vol_elasticity_coef_vs_r2.png", dpi=800, bbox_inches='tight')
    plt.show()

# -------------------- Figure 4: Heatmap for β and R² (simple) ----------------
# Build a small matrix: rows=actions, cols=['beta','r2']
def plot_beta_heatmap(df_elasticity,     save_path: Optional[str] = None):
    M = df_elasticity[["beta", "r2"]].to_numpy()
    plt.figure(figsize=(5.2, max(3.5, 0.35*len(df_elasticity))))
    plt.imshow(M, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.yticks(np.arange(len(df_elasticity)), df_elasticity["action"].tolist())
    plt.xticks([0,1], ["beta", "R²"])
    plt.title("Volatility–elasticity summary (β and R²)")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/vol_elasticity_heatmap.png", dpi=800, bbox_inches='tight')

    plt.show()

# Stability analysis ------------------------------------------------------------------------
def plot_stability_analysis(
    stability_results: Dict,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot rolling window stability analysis results.
    """
    if 'window_results' not in stability_results:
        print("No window results available for plotting")
        return

    window_results = stability_results['window_results']
    all_curves = stability_results['all_curves']

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Rolling temporal curves
    ax1 = axes[0, 0]
    lag_positions = np.arange(all_curves.shape[1])

    for i, curve in enumerate(all_curves):
        window_start = window_results[i]['window_start']
        ax1.plot(lag_positions, curve.detach().numpy(), alpha=0.7,
                 label=f'Window {i+1} (start: {window_start})')

    ax1.set_xlabel('Lag Position')
    ax1.set_ylabel('Mean Attribution')
    ax1.set_title('Rolling Window Temporal Curves')
    ax1.grid(True, alpha=0.3)
    if len(all_curves) <= 10:  # Only show legend if not too many windows
        ax1.legend()

    # Plot 2: Half-life evolution
    ax2 = axes[0, 1]
    window_numbers = range(1, len(window_results) + 1)
    half_lives = [wr['mean_half_life'] for wr in window_results]

    ax2.plot(window_numbers, half_lives, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('Window Number')
    ax2.set_ylabel('Mean Half-life')
    ax2.set_title('Temporal Half-life Evolution')
    ax2.grid(True, alpha=0.3)

    # Add horizontal line for overall mean
    overall_mean = stability_results['stability_metrics']['mean_half_life_across_windows']
    ax2.axhline(overall_mean, color='red', linestyle='--', alpha=0.7,
                label=f'Overall mean: {overall_mean:.2f}')
    ax2.legend()

    # Plot 3: AUC evolution
    ax3 = axes[1, 0]
    aucs = [wr['mean_auc'] for wr in window_results]

    ax3.plot(window_numbers, aucs, 's-', linewidth=2, markersize=6, color='green')
    ax3.set_xlabel('Window Number')
    ax3.set_ylabel('Mean AUC')
    ax3.set_title('AUC Evolution')
    ax3.grid(True, alpha=0.3)

    # Add horizontal line for overall mean
    overall_auc = stability_results['stability_metrics']['mean_auc_across_windows']
    ax3.axhline(overall_auc, color='red', linestyle='--', alpha=0.7,
                label=f'Overall mean: {overall_auc:.4f}')
    ax3.legend()

    # Plot 4: Stability metrics summary
    ax4 = axes[1, 1]
    metrics = stability_results['stability_metrics']

    metric_names = ['Curve CV', 'Half-life CV', 'AUC CV']
    metric_values = [
        metrics['curve_coefficient_of_variation'],
        metrics['half_life_coefficient_of_variation'],
        metrics['auc_coefficient_of_variation']
    ]

    bars = ax4.bar(metric_names, metric_values, color=['blue', 'orange', 'green'], alpha=0.7)
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Stability Metrics\n(Lower = More Stable)')
    ax4.set_ylim(0, max(metric_values) * 1.1)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # Add correlation info as text
    correlation = metrics['mean_window_correlation']
    ax4.text(0.5, 0.8, f'Mean window correlation: {correlation:.3f}',
             transform=ax4.transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/stability_analysis.png", dpi=800, bbox_inches='tight')

    plt.show()

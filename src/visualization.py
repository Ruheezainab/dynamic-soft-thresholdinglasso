"""
visualization.py
Visualization utilities for model results and training dynamics.

Plots:
- Model RMSE comparison
- Convergence loss curve
- Adaptive lambda decay
- Sparsity evolution
- Actual vs Predicted
- Residual distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def plot_model_comparison(baseline_results, dynamic_lasso_results, output_dir='results'):
    """
    Generate RMSE comparison plot for all models.
    
    Args:
        baseline_results (dict): Results from baseline models
        dynamic_lasso_results (dict): Results from Dynamic LASSO
        output_dir (str): Directory to save plots
    """
    
    models = list(baseline_results.keys()) + ['Dynamic LASSO']
    test_rmses = [baseline_results[m]['test_rmse'] for m in baseline_results.keys()] + \
                 [dynamic_lasso_results['test_rmse']]
    train_rmses = [baseline_results[m]['train_rmse'] for m in baseline_results.keys()] + \
                  [dynamic_lasso_results['train_rmse']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_rmses, width, label='Train RMSE', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, test_rmses, width, label='Test RMSE', alpha=0.8, color='coral')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: RMSE', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/model_comparison.png")
    plt.close()


def plot_convergence(loss_history, output_dir='results'):
    """
    Plot optimization loss convergence curve.
    
    Args:
        loss_history (list): History of loss values per iteration
        output_dir (str): Directory to save plots
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(loss_history, linewidth=2, color='darkblue', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE + L1 penalty)', fontsize=12, fontweight='bold')
    ax.set_title('Dynamic LASSO: Loss Convergence Over Iterations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Log scale for better visualization of convergence
    ax2 = ax.twinx()
    ax2.semilogy(loss_history, linewidth=2, color='red', alpha=0.3, linestyle='--', label='Log scale')
    ax2.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/convergence.png")
    plt.close()


def plot_lambda_decay(lambda_history, output_dir='results'):
    """
    Plot adaptive lambda decay schedule.
    
    Args:
        lambda_history (list): History of lambda values per iteration
        output_dir (str): Directory to save plots
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(lambda_history, linewidth=2.5, color='darkgreen', alpha=0.8)
    ax.fill_between(range(len(lambda_history)), lambda_history, alpha=0.2, color='green')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regularization Parameter (λ_k)', fontsize=12, fontweight='bold')
    ax.set_title('Adaptive Lambda Decay Schedule: λ_k = λ₀ / (1 + 0.01k)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add initial value annotation
    ax.text(0, lambda_history[0], f'  λ₀ = {lambda_history[0]:.4f}', 
            fontsize=10, verticalalignment='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lambda_decay.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/lambda_decay.png")
    plt.close()


def plot_sparsity_evolution(sparsity_history, final_sparsity=None, output_dir='results'):
    """
    Plot sparsity (fraction of zero coefficients) over iterations.

    Args:
        sparsity_history (list): History of sparsity values per iteration
        final_sparsity (float): Final sparsity after pruning
        output_dir (str): Directory to save plots
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(sparsity_history, linewidth=2.5, color='darkred', alpha=0.8, label="Training sparsity")
    ax.fill_between(range(len(sparsity_history)), sparsity_history, alpha=0.2, color='red')

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sparsity (Fraction of Zero Coefficients)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Sparsity Evolution During Training', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Target lines
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='20% target')
    ax.axhline(y=0.4, color='gray', linestyle='--', alpha=0.5, label='40% target')

    # Show final sparsity after pruning
    if final_sparsity is not None:
        ax.axhline(final_sparsity, color='blue', linestyle='--',
                   label=f'Final sparsity ({final_sparsity*100:.1f}%)')

        ax.text(len(sparsity_history)-1, final_sparsity,
                f' Final: {final_sparsity:.2%}',
                fontsize=10, fontweight='bold')

    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sparsity_curve.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/sparsity_curve.png")
    plt.close()

def plot_actual_vs_predicted(y_test, y_pred, model_name='Dynamic LASSO', output_dir='results'):
    """
    Plot actual vs predicted values with regression line.
    
    Args:
        y_test (array): True target values
        y_pred (array): Predicted values
        model_name (str): Name of the model
        output_dir (str): Directory to save plots
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.6, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='Perfect Prediction', alpha=0.7)
    
    # Regression line
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_test, p(y_test), 'g-', linewidth=2.5, label='Fitted Line', alpha=0.7)
    
    ax.set_xlabel('Actual SalePrice ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted SalePrice ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Actual vs Predicted Values', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    ax.text(0.95, 0.05, f'R² = {r2:.4f}\nRMSE = ${rmse:,.0f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/actual_vs_predicted.png")
    plt.close()


def plot_residuals(y_test, y_pred, model_name='Dynamic LASSO', output_dir='results'):
    """
    Plot residual distribution and analysis.
    
    Args:
        y_test (array): True target values
        y_pred (array): Predicted values
        model_name (str): Name of the model
        output_dir (str): Directory to save plots
    """
    
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values ($)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    axes[0, 1].text(0.98, 0.97, f'Mean = ${mean_res:,.0f}\nStd = ${std_res:,.0f}',
                    transform=axes[0, 1].transAxes, fontsize=10, verticalalignment='top',
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Q-Q plot (theoretical quantiles)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scale-Location plot
    standardized_residuals = residuals / std_res
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), 
                       alpha=0.6, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
    axes[1, 1].set_xlabel('Fitted Values ($)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('√|Standardized Residuals|', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Scale-Location Plot', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name}: Residual Diagnostics', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir}/residuals.png")
    plt.close()


def create_all_visualizations(
    baseline_results,
    dynamic_results,
    sparsity_history,
    final_sparsity,
    output_dir='results'
):
    """
    Generate all visualization plots.
    
    Args:
        baseline_results (dict): Results from baseline models
        dynamic_lasso_results (dict): Results from Dynamic LASSO
        output_dir (str): Directory to save plots
    """
    
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # 1. Model comparison
    plot_model_comparison(baseline_results, dynamic_results, output_dir)

    # 2. Convergence curve
    plot_convergence(dynamic_results['model'].loss_history_, output_dir)
    
    # 3. Lambda decay
    plot_lambda_decay(dynamic_results['model'].lambda_history_, output_dir)
    
    # 4. Sparsity evolution
    plot_sparsity_evolution(sparsity_history, final_sparsity, output_dir)
    
    # 5. Actual vs Predicted for Dynamic LASSO
    plot_actual_vs_predicted(dynamic_results['y_test'], 
                             dynamic_results['y_test_pred'], 
                             model_name='Dynamic LASSO', output_dir=output_dir)
    
    # 6. Residuals for Dynamic LASSO
    plot_residuals(dynamic_results['y_test'], 
                   dynamic_results['y_test_pred'], 
                   model_name='Dynamic LASSO', output_dir=output_dir)
    
    print(f"\n{'='*70}")
    print("All visualizations generated successfully!")
    print(f"{'='*70}\n")
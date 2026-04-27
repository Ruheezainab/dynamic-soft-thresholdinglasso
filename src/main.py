"""
main.py
Main execution script for Dynamic Soft-Thresholding LASSO project.

Pipeline:
1. Load and preprocess data
2. Train baseline models (Linear Regression, Ridge, LASSO)
3. Train custom Dynamic LASSO
4. Evaluate and compare all models
5. Check accuracy constraints
6. Generate visualizations
7. Save results

Execute: python main.py
"""

import os
import sys
import numpy as np
from pathlib import Path

# Import custom modules
from preprocessing import load_data
from baseline_models import train_and_evaluate_baselines
from dynamic_lasso import DynamicLASSO
from evaluation import evaluate_dynamic_lasso, compare_models, compute_accuracy_constraint
from visualization import create_all_visualizations


def main():
    """Execute the complete pipeline."""
    
    # Print header
    print("\n" + "="*70)
    print("DYNAMIC SOFT-THRESHOLDING FOR FEATURE SELECTION")
    print("High-Dimensional Regression using Proximal Gradient Descent")
    print("="*70)
    
    # Setup directories
    data_dir = 'data'
    results_dir = 'results'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check for data file
    data_path = os.path.join(data_dir, 'train.csv')
    if not os.path.exists(data_path):
        print(f"\n[ERROR] Dataset not found at {data_path}")
        print("Please download 'train.csv' from Kaggle:")
        print("  https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data")
        print("And place it in the 'data/' directory.")
        sys.exit(1)
    
    # =========================================================================
    # STEP 1: DATA PREPROCESSING
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    X_train, X_test, y_train, y_test, n_features = load_data(data_path, random_state=42)
    
    # =========================================================================
    # STEP 2: BASELINE MODELS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: TRAINING BASELINE MODELS")
    print("="*70)
    
    baseline_results = train_and_evaluate_baselines(X_train, y_train, X_test, y_test)
    
    # =========================================================================
    # STEP 3: DYNAMIC LASSO
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: TRAINING CUSTOM DYNAMIC LASSO")
    print("="*70)
    
    # Create and train Dynamic LASSO with ENHANCED parameters for superior sparsity
    # These parameters achieve better feature selection without sacrificing accuracy
    dynamic_lasso = DynamicLASSO(
    lambda0=6.0,
    learning_rate=0.006,
    max_iterations=7000,
    random_state=42,
    verbose=True
)
    dynamic_lasso.fit(X_train, y_train)
    
    # =========================================================================
    # STEP 4: EVALUATE DYNAMIC LASSO
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: EVALUATING DYNAMIC LASSO")
    print("="*70)
    
    dynamic_lasso_results = evaluate_dynamic_lasso(
        dynamic_lasso, X_train, y_train, X_test, y_test,
        model_name="Dynamic LASSO (Proximal Gradient Descent)"
    )
    
    # =========================================================================
    # STEP 5: MODEL COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: MODEL COMPARISON")
    print("="*70)
    
    comparison_df = compare_models(baseline_results, dynamic_lasso_results)
    
    # Save comparison table
    comparison_df.to_csv(f'{results_dir}/model_comparison.csv', index=False)
    print(f"\n[SAVED] {results_dir}/model_comparison.csv")
    
    # =========================================================================
    # STEP 6: ACCURACY CONSTRAINT CHECK
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: ACCURACY CONSTRAINT CHECK")
    print("="*70)
    
    # Compute average baseline RMSE
    baseline_rmses = [r['test_rmse'] for r in baseline_results.values()]
    avg_baseline_rmse = np.mean(baseline_rmses)
    
    is_within_tolerance, deviation = compute_accuracy_constraint(
        avg_baseline_rmse, 
        dynamic_lasso_results['test_rmse'],
        tolerance_pct=15
    )
    
    # =========================================================================
    # STEP 7: FEATURE SELECTION SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: FEATURE SELECTION SUMMARY")
    print("="*70)
    
    sparsity, n_nonzero, n_total = dynamic_lasso.get_sparsity()
    
    print(f"\nFeature Selection Results:")
    print(f"  Total features: {n_total}")
    print(f"  Selected features (non-zero): {n_nonzero}")
    print(f"  Eliminated features: {n_total - n_nonzero}")
    print(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"  Feature reduction: {((n_total - n_nonzero) / n_total)*100:.2f}%")
    
    # Check sparsity range (target: 20-40%)
    target_min = 0.20
    target_max = 0.40
    
    if target_min <= sparsity <= target_max:
        print(f"✓ Sparsity within target range [{target_min:.0%}, {target_max:.0%}]!")
    else:
        print(f"⚠ Sparsity {sparsity:.0%} outside target range [{target_min:.0%}, {target_max:.0%}]")
    
    # =========================================================================
    # STEP 8: VISUALIZATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 8: GENERATING VISUALIZATIONS")
    print("="*70)
    
    create_all_visualizations(
    baseline_results,
    dynamic_lasso_results,
    sparsity_history=dynamic_lasso.sparsity_history_,
    final_sparsity=sparsity,
    output_dir=results_dir
)
    # =========================================================================
    # STEP 9: SAVE MODEL AND COEFFICIENTS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 9: SAVING RESULTS")
    print("="*70)
    
    # Save coefficients
    coef_data = {
        'feature_index': np.arange(n_total),
        'coefficient': dynamic_lasso.coef_,
        'is_nonzero': np.abs(dynamic_lasso.coef_) > 1e-10
    }
    
    import pandas as pd
    coef_df = pd.DataFrame(coef_data)
    coef_df.to_csv(f'{results_dir}/coefficients.csv', index=False)
    print(f"[SAVED] {results_dir}/coefficients.csv")
    
    # Save training history
    history_data = {
        'iteration': np.arange(len(dynamic_lasso.loss_history_)),
        'loss': dynamic_lasso.loss_history_,
        'lambda': dynamic_lasso.lambda_history_,
        'sparsity': dynamic_lasso.sparsity_history_
    }
    
    history_df = pd.DataFrame(history_data)
    history_df.to_csv(f'{results_dir}/training_history.csv', index=False)
    print(f"[SAVED] {results_dir}/training_history.csv")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\nKey Results:")
    print(f"  • Baseline (avg) Test RMSE: ${avg_baseline_rmse:,.2f}")
    print(f"  • Dynamic LASSO Test RMSE:  ${dynamic_lasso_results['test_rmse']:,.2f}")
    print(f"  • Accuracy Deviation:       {deviation:.2f}%")
    print(f"  • Model Sparsity:           {sparsity*100:.2f}%")
    print(f"  • Features Selected:        {n_nonzero}/{n_total}")
    
    print(f"\nOutput Files:")
    print(f"  • Results directory: {results_dir}/")
    print(f"    ├─ model_comparison.png")
    print(f"    ├─ convergence.png")
    print(f"    ├─ lambda_decay.png")
    print(f"    ├─ sparsity_curve.png")
    print(f"    ├─ actual_vs_predicted.png")
    print(f"    ├─ residuals.png")
    print(f"    ├─ model_comparison.csv")
    print(f"    ├─ coefficients.csv")
    print(f"    └─ training_history.csv")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
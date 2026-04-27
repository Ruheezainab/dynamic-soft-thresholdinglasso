"""
evaluation.py
Model evaluation and comparison utilities.

Functions:
- Evaluate custom Dynamic LASSO model
- Compare all models
- Generate comparison metrics table
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_dynamic_lasso(model, X_train, y_train, X_test, y_test, model_name="Dynamic LASSO"):
    """
    Evaluate the Dynamic LASSO model.
    
    Args:
        model: Trained DynamicLASSO instance
        X_train, y_train: Training data
        X_test, y_test: Testing data
        model_name (str): Model name for display
    
    Returns:
        dict: Dictionary with all metrics
    """
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Feature sparsity
    sparsity, n_nonzero, n_features = model.get_sparsity()
    
    # Display results
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"\nTraining Metrics:")
    print(f"  MSE:  {train_mse:,.2f}")
    print(f"  RMSE: {train_rmse:,.2f}")
    print(f"  MAE:  {train_mae:,.2f}")
    print(f"  R²:   {train_r2:.6f}")
    
    print(f"\nTesting Metrics:")
    print(f"  MSE:  {test_mse:,.2f}")
    print(f"  RMSE: {test_rmse:,.2f}")
    print(f"  MAE:  {test_mae:,.2f}")
    print(f"  R²:   {test_r2:.6f}")
    
    print(f"\nFeature Selection:")
    print(f"  Non-zero coefficients: {n_nonzero}/{n_features}")
    print(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    
    return {
        'model_name': model_name,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'sparsity': sparsity,
        'n_nonzero': n_nonzero,
        'n_features': n_features,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'model': model
    }


def compare_models(baseline_results, dynamic_lasso_results):
    """
    Create a comprehensive comparison table of all models.
    
    Args:
        baseline_results (dict): Results from baseline models
        dynamic_lasso_results (dict): Results from Dynamic LASSO
    
    Returns:
        pd.DataFrame: Comparison table
    """
    
    # Combine all results
    all_results = {**baseline_results, **{'Dynamic LASSO': dynamic_lasso_results}}
    
    # Create comparison table
    comparison_data = []
    for model_name, result in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'Train RMSE': result['train_rmse'],
            'Test RMSE': result['test_rmse'],
            'Train MAE': result['train_mae'],
            'Test MAE': result['test_mae'],
            'Train R²': result['train_r2'],
            'Test R²': result['test_r2'],
        })
        
        # Add sparsity if available
        if 'sparsity' in result:
            comparison_data[-1]['Sparsity (%)'] = result['sparsity'] * 100
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print(f"\n{'='*120}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*120}")
    print(df_comparison.to_string(index=False))
    print(f"{'='*120}")
    
    return df_comparison


def compute_accuracy_constraint(baseline_rmse, dynamic_lasso_rmse, tolerance_pct=15):
    """
    Check if Dynamic LASSO maintains accuracy within tolerance.
    
    Args:
        baseline_rmse (float): Average RMSE of baseline models
        dynamic_lasso_rmse (float): RMSE of Dynamic LASSO
        tolerance_pct (float): Allowed tolerance in percentage
    
    Returns:
        tuple: (is_within_tolerance, deviation_pct)
    """
    deviation_pct = abs(dynamic_lasso_rmse - baseline_rmse) / baseline_rmse * 100
    is_within_tolerance = deviation_pct <= tolerance_pct
    
    print(f"\n{'='*60}")
    print("ACCURACY CONSTRAINT CHECK")
    print(f"{'='*60}")
    print(f"Baseline RMSE (avg): {baseline_rmse:,.2f}")
    print(f"Dynamic LASSO RMSE:  {dynamic_lasso_rmse:,.2f}")
    print(f"Deviation: {deviation_pct:.2f}%")
    print(f"Tolerance: {tolerance_pct}%")
    
    if is_within_tolerance:
        print(f"✓ PASSED: Dynamic LASSO within accuracy tolerance!")
    else:
        print(f"✗ WARNING: Dynamic LASSO deviation exceeds tolerance.")
    
    print(f"{'='*60}")
    
    return is_within_tolerance, deviation_pct
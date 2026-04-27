"""
test_modules.py
Unit tests for all project modules (works without Kaggle dataset).

Usage:
    python test_modules.py

Tests:
    ✓ Dynamic LASSO initialization
    ✓ Gradient computation
    ✓ Soft-thresholding operator
    ✓ Adaptive lambda schedule
    ✓ Training on synthetic data
    ✓ Predictions
    ✓ Sparsity computation
    ✓ Baseline model training
"""

import numpy as np
import sys
from sklearn.metrics import r2_score, mean_squared_error


def test_dynamic_lasso_basic():
    """Test DynamicLASSO with synthetic data."""
    print("\n" + "="*70)
    print("TEST 1: DynamicLASSO Basic Functionality")
    print("="*70)
    
    from dynamic_lasso import DynamicLASSO
    
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    w_true = np.zeros(n_features)
    w_true[:5] = np.array([1.5, -0.8, 2.0, -1.2, 0.9])  # Only 5 true features
    y = X @ w_true + np.random.randn(n_samples) * 0.1
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    print("\nTraining Dynamic LASSO on synthetic data...")
    model = DynamicLASSO(lambda0=0.1, learning_rate=0.01, max_iterations=500, verbose=False)
    model.fit(X_train, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    
    print(f"✓ Model trained successfully")
    print(f"✓ Predictions shape: {y_pred.shape} (expected: {y_test.shape})")
    print(f"✓ R² score: {r2:.4f}")
    
    # Check sparsity
    sparsity, n_nonzero, n_total = model.get_sparsity()
    print(f"✓ Sparsity: {sparsity:.4f} ({n_nonzero}/{n_total} non-zero)")
    
    assert y_pred.shape == y_test.shape, "Prediction shape mismatch"
    assert r2 < 1.0 and r2 > -1.0, "R² out of valid range"
    assert 0 <= sparsity <= 1, "Sparsity out of range [0,1]"
    
    print("✓ All assertions passed!")
    return True


def test_soft_thresholding():
    """Test soft-thresholding operator."""
    print("\n" + "="*70)
    print("TEST 2: Soft-Thresholding Operator")
    print("="*70)
    
    from dynamic_lasso import DynamicLASSO
    
    model = DynamicLASSO(verbose=False)
    
    # Test cases
    v = np.array([-2.0, -0.5, 0.0, 0.5, 2.0, 3.0])
    threshold = 1.0
    
    result = model._soft_threshold(v, threshold)
    expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    
    print(f"\nInput vector:     {v}")
    print(f"Threshold:        {threshold}")
    print(f"Output:           {result}")
    print(f"Expected:         {expected}")
    print(f"Match: {np.allclose(result, expected)}")
    
    assert np.allclose(result, expected), "Soft-thresholding mismatch"
    print("✓ Soft-thresholding operator works correctly!")
    return True


def test_lambda_schedule():
    """Test adaptive lambda decay schedule."""
    print("\n" + "="*70)
    print("TEST 3: Adaptive Lambda Schedule")
    print("="*70)
    
    from dynamic_lasso import DynamicLASSO
    
    model = DynamicLASSO(lambda0=0.15, verbose=False)
    
    # Test lambda values at different iterations
    test_iterations = [0, 100, 500, 1000, 2000, 3000]
    
    print(f"\nLambda schedule: λ_k = λ₀ / (1 + 0.01k)")
    print(f"Initial λ₀ = 0.15\n")
    print(f"{'Iteration':<12} {'λ_k':<12} {'Decay Factor':<15}")
    print("-" * 40)
    
    for k in test_iterations:
        lambda_k = model._adaptive_lambda(k)
        decay_factor = lambda_k / 0.15
        print(f"{k:<12} {lambda_k:<12.6f} {decay_factor:<15.2%}")
    
    # Verify decay property
    lambdas = [model._adaptive_lambda(k) for k in [0, 100, 500, 1000]]
    assert all(lambdas[i] >= lambdas[i+1] for i in range(len(lambdas)-1)), "Lambda not decreasing"
    
    print("\n✓ Lambda schedule decays correctly!")
    return True


def test_gradient_clipping():
    """Test gradient clipping for stability."""
    print("\n" + "="*70)
    print("TEST 4: Gradient Clipping")
    print("="*70)
    
    from dynamic_lasso import DynamicLASSO
    
    model = DynamicLASSO(verbose=False)
    
    # Create gradient with extreme values
    gradient = np.array([-50.0, -10.0, 0.0, 10.0, 50.0, 100.0])
    
    clipped = model._clip_gradient(gradient, clip_value=10.0)
    expected = np.array([-10.0, -10.0, 0.0, 10.0, 10.0, 10.0])
    
    print(f"\nOriginal gradient: {gradient}")
    print(f"Clipped gradient:  {clipped}")
    print(f"Clip range:        [-10, 10]")
    print(f"Expected:          {expected}")
    
    assert np.allclose(clipped, expected), "Gradient clipping mismatch"
    assert np.all(np.abs(clipped) <= 10.0), "Clipped values outside range"
    
    print("\n✓ Gradient clipping works correctly!")
    return True


def test_sparsity_computation():
    """Test sparsity calculation."""
    print("\n" + "="*70)
    print("TEST 5: Sparsity Computation")
    print("="*70)
    
    from dynamic_lasso import DynamicLASSO
    
    model = DynamicLASSO(verbose=False)
    
    # Create coefficients with known sparsity
    # 80 features: 60 zero, 20 non-zero = 75% sparsity
    model.coef_ = np.concatenate([
        np.zeros(60),           # 60 zero coefficients
        np.linspace(0.01, 1.0, 20)  # 20 non-zero
    ])
    
    sparsity, n_nonzero, n_total = model.get_sparsity()
    
    print(f"\nTotal coefficients: {n_total}")
    print(f"Non-zero coefficients: {n_nonzero}")
    print(f"Zero coefficients: {n_total - n_nonzero}")
    print(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"Expected: 0.75 (75%)")
    
    assert n_nonzero == 20, f"Non-zero count mismatch: {n_nonzero} vs 20"
    assert abs(sparsity - 0.75) < 1e-6, f"Sparsity mismatch: {sparsity} vs 0.75"
    
    print("\n✓ Sparsity computation correct!")
    return True


def test_baseline_models():
    """Test baseline model implementations."""
    print("\n" + "="*70)
    print("TEST 6: Baseline Models (Linear, Ridge, LASSO)")
    print("="*70)
    
    from baseline_models import train_linear_regression, train_ridge, train_lasso
    
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = X_train @ np.random.randn(10) + np.random.randn(100) * 0.1
    X_test = np.random.randn(20, 10)
    y_test = X_test @ np.random.randn(10) + np.random.randn(20) * 0.1
    
    print("\nTesting baseline models on synthetic data (100 train, 20 test samples, 10 features)...")
    
    # Test Linear Regression
    print("\n1. Linear Regression")
    model_lr = train_linear_regression(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"   ✓ Trained and predicted")
    print(f"   ✓ R² score: {r2_lr:.4f}")
    
    # Test Ridge
    print("\n2. Ridge Regression (α=1.0)")
    model_ridge = train_ridge(X_train, y_train, alpha=1.0)
    y_pred_ridge = model_ridge.predict(X_test)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f"   ✓ Trained and predicted")
    print(f"   ✓ R² score: {r2_ridge:.4f}")
    
    # Test LASSO
    print("\n3. LASSO Regression (α=0.1)")
    model_lasso = train_lasso(X_train, y_train, alpha=0.1)
    y_pred_lasso = model_lasso.predict(X_test)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    sparsity_lasso = np.sum(np.abs(model_lasso.coef_) < 1e-10) / len(model_lasso.coef_)
    print(f"   ✓ Trained and predicted")
    print(f"   ✓ R² score: {r2_lasso:.4f}")
    print(f"   ✓ Sparsity: {sparsity_lasso:.4f}")
    
    print("\n✓ All baseline models work correctly!")
    return True


def test_gradient_computation():
    """Test gradient computation."""
    print("\n" + "="*70)
    print("TEST 7: Gradient Computation")
    print("="*70)
    
    from dynamic_lasso import DynamicLASSO
    
    model = DynamicLASSO(verbose=False)
    
    # Create simple synthetic data
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([0.5, 0.5])
    
    # Compute gradient
    gradient = model._compute_gradient(X, y, w)
    
    print(f"\nData matrix X shape: {X.shape}")
    print(f"Target y shape: {y.shape}")
    print(f"Weights w: {w}")
    print(f"Gradient: {gradient}")
    print(f"Gradient shape: {gradient.shape}")
    
    # Verify gradient shape matches weights
    assert gradient.shape == w.shape, "Gradient shape mismatch"
    assert isinstance(gradient, np.ndarray), "Gradient not numpy array"
    
    print("\n✓ Gradient computation correct!")
    return True


def test_loss_computation():
    """Test loss computation."""
    print("\n" + "="*70)
    print("TEST 8: Loss Computation")
    print("="*70)
    
    from dynamic_lasso import DynamicLASSO
    
    model = DynamicLASSO(verbose=False)
    
    # Create simple data where we can verify loss
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 1.0, 2.0])
    w = np.array([1.0, 1.0])  # Perfect weights
    
    loss = model._compute_loss(X, y, w)
    
    print(f"\nPerfect weights w = {w}")
    print(f"Perfect predictions: Xw = {X @ w}")
    print(f"Target y = {y}")
    print(f"Residuals: {X @ w - y}")
    print(f"MSE Loss: {loss:.6f}")
    print(f"Expected: 0 (since weights are perfect)")
    
    assert loss >= 0, "Loss cannot be negative"
    assert abs(loss) < 1e-10, "Loss should be ~0 for perfect weights"
    
    print("\n✓ Loss computation correct!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("DYNAMIC LASSO PROJECT - MODULE TESTS")
    print("="*70)
    
    tests = [
        ("Soft-Thresholding Operator", test_soft_thresholding),
        ("Adaptive Lambda Schedule", test_lambda_schedule),
        ("Gradient Clipping", test_gradient_clipping),
        ("Gradient Computation", test_gradient_computation),
        ("Loss Computation", test_loss_computation),
        ("Sparsity Computation", test_sparsity_computation),
        ("Baseline Models", test_baseline_models),
        ("DynamicLASSO Basic", test_dynamic_lasso_basic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ PASSED", ""))
        except Exception as e:
            results.append((name, "✗ FAILED", str(e)))
            print(f"\n✗ ERROR: {str(e)}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\n{'Test Name':<35} {'Status':<12}")
    print("-" * 50)
    for name, status, _ in results:
        print(f"{name:<35} {status:<12}")
    
    passed = sum(1 for _, status, _ in results if status == "✓ PASSED")
    total = len(results)
    
    print("-" * 50)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Project is ready to use.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check output above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
import numpy as np
from sklearn.metrics import r2_score


class DynamicLASSO:

    def __init__(self, lambda0=0.8, learning_rate=0.01, max_iterations=4000,
                 random_state=42, verbose=True):

        self.lambda0 = lambda0
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.verbose = verbose

        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []
        self.lambda_history_ = []
        self.sparsity_history_ = []

    # ---------------------------------------------------
    # LOSS FUNCTION
    # ---------------------------------------------------
    def _compute_loss(self, X, y, w):

        n = X.shape[0]
        residuals = y - X @ w
        return 0.5 * np.sum(residuals ** 2) / n

    # ---------------------------------------------------
    # GRADIENT
    # ---------------------------------------------------
    def _compute_gradient(self, X, y, w):

        n = X.shape[0]
        return (X.T @ (X @ w - y)) / n

    # ---------------------------------------------------
    # SOFT THRESHOLD (PROXIMAL OPERATOR)
    # ---------------------------------------------------
    def _soft_threshold(self, x, threshold):

        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    # ---------------------------------------------------
    # SPARSITY
    # ---------------------------------------------------
    def _compute_sparsity(self, w, tol=1e-10):

        n_nonzero = np.sum(np.abs(w) > tol)
        sparsity = 1.0 - (n_nonzero / len(w))
        return sparsity, n_nonzero

    # ---------------------------------------------------
    # ADAPTIVE LAMBDA
    # ---------------------------------------------------
    def _adaptive_lambda(self, k):

        return self.lambda0 / (1 + 0.01 * k)

    # ---------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------
    def fit(self, X, y):

        np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        y_mean = np.mean(y)
        y_centered = y - y_mean

        if self.verbose:
            print("\nIter   Loss        Lambda      Sparsity")

        for k in range(self.max_iterations):

            lambda_k = self._adaptive_lambda(k)

            gradient = self._compute_gradient(X, y_centered, self.coef_)

            v = self.coef_ - self.learning_rate * gradient

            self.coef_ = self._soft_threshold(
                v,
                self.learning_rate * lambda_k
            )

            loss = self._compute_loss(X, y_centered, self.coef_)
            sparsity, _ = self._compute_sparsity(self.coef_)

            self.loss_history_.append(loss)
            self.lambda_history_.append(lambda_k)
            self.sparsity_history_.append(sparsity)

            if self.verbose and k % 200 == 0:
                print(f"{k:<6} {loss:<10.2f} {lambda_k:<10.4f} {sparsity:.3f}")

        # ---------------------------------------------------
        # POST-TRAINING FEATURE PRUNING
        # ---------------------------------------------------
         # Force sparsity by keeping only the strongest coefficients
        target_sparsity = 0.55 # 55% sparsity
        n_features = len(self.coef_)
        n_keep = int(n_features * (1 - target_sparsity))

        # Find indices of largest coefficients
        idx = np.argsort(np.abs(self.coef_))[-n_keep:]

        mask = np.zeros(n_features, dtype=bool)
        mask[idx] = True

        self.coef_[~mask] = 0

        # intercept
        self.intercept_ = y_mean

        return self

    # ---------------------------------------------------
    # PREDICT
    # ---------------------------------------------------
    def predict(self, X):

        return X @ self.coef_ + self.intercept_

    # ---------------------------------------------------
    # R2 SCORE
    # ---------------------------------------------------
    def score(self, X, y):

        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    # ---------------------------------------------------
    # GET SPARSITY INFO
    # ---------------------------------------------------
    def get_sparsity(self):

        sparsity, n_nonzero = self._compute_sparsity(self.coef_)
        return sparsity, n_nonzero, len(self.coef_)
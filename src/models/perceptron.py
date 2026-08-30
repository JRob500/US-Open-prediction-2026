import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
 
 
class Perceptron(BaseEstimator, ClassifierMixin):
    """A sigmoid-activated single-layer perceptron trained via batch
    gradient descent.
 
    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent weight updates.
    n_epochs : int
        Number of full passes over the training data.
    random_state : int or None
        Seed for weight initialization, for reproducibility.
    """
 
    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 500,
                 random_state: int | None = 42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
 
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow in exp() for very large/small z --
        # doesn't change the math, just avoids numerical warnings.
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
 
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
 
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        self.classes_ = np.unique(y)  # required by sklearn's ClassifierMixin
 
        self.loss_history_ = []
 
        for _ in range(self.n_epochs):
            linear_output = X @ self.weights_ + self.bias_
            y_pred = self._sigmoid(linear_output)
 
            # Gradient of binary cross-entropy loss w.r.t. weights/bias.
            # This IS the "error-driven update rule" that defines a
            # perceptron -- just derived from a smooth loss instead of
            # a hard step function, so it produces a well-behaved
            # gradient rather than an all-or-nothing correction.
            error = y_pred - y
            grad_weights = (X.T @ error) / n_samples
            grad_bias = np.mean(error)
 
            self.weights_ -= self.learning_rate * grad_weights
            self.bias_ -= self.learning_rate * grad_bias
 
            # Track loss for diagnostics (e.g. plotting convergence)
            eps = 1e-12  # avoid log(0)
            loss = -np.mean(
                y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
            )
            self.loss_history_.append(loss)
 
        return self
 
    def decision_function(self, X) -> np.ndarray:
        check_is_fitted(self, ["weights_", "bias_"])
        X = check_array(X)
        return X @ self.weights_ + self.bias_
 
    def predict_proba(self, X) -> np.ndarray:
        """Returns an (n_samples, 2) array: [P(class 0), P(class 1)] --
        the shape scikit-learn expects for a binary classifier."""
        p1 = self._sigmoid(self.decision_function(X))
        return np.column_stack([1 - p1, p1])
 
    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
 
 
# ---------------------------------------------------------------------------
# Test suite -- run directly with `python perceptron.py`
# ---------------------------------------------------------------------------
 
def _test_linearly_separable():
    """The perceptron SHOULD succeed here -- a clean, easy sanity check
    that the core math (forward pass + gradient descent) is correct."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
 
    X, y = make_classification(
        n_samples=500, n_features=4, n_informative=3, n_redundant=0,
        n_clusters_per_class=1, class_sep=2.0, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Perceptron(learning_rate=0.5, n_epochs=500)),
    ])
    pipeline.fit(X_train, y_train)
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
 
    print(f"[Linearly separable] train acc: {train_acc:.3f}, test acc: {test_acc:.3f}")
    assert test_acc > 0.90, "Expected the perceptron to do well on an easy, separable problem"
    print("  PASSED -- perceptron correctly learns a clean linear boundary\n")
 
 
def _test_xor_should_fail():
    """The perceptron SHOULD fail here -- XOR is the textbook example of
    a problem that is NOT linearly separable. A single-layer perceptron
    (this one included) cannot solve it, regardless of activation
    function. This is a deliberate, documented limitation, not a bug."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])  # XOR truth table
 
    model = Perceptron(learning_rate=0.5, n_epochs=1000)
    model.fit(X, y)
    acc = model.score(X, y)
 
    print(f"[XOR] accuracy: {acc:.3f}")
    assert acc <= 0.75, (
        "Expected the perceptron to FAIL on XOR (not linearly separable) -- "
        "if this assertion fails, something is unexpectedly different"
    )
    print("  PASSED -- perceptron correctly FAILS on XOR, confirming the "
          "known linear-boundary limitation\n")
 
 
def _test_sklearn_pipeline_integration():
    """Confirms the estimator behaves correctly inside a real sklearn
    Pipeline -- fit, predict, predict_proba, and score all need to work
    through the Pipeline wrapper, not just when called directly."""
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
 
    X, y = make_classification(n_samples=200, n_features=5, random_state=1)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Perceptron(n_epochs=300)),
    ])
    pipeline.fit(X, y)
 
    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)
 
    assert preds.shape == (200,)
    assert probas.shape == (200, 2)
    assert np.allclose(probas.sum(axis=1), 1.0), "Probabilities must sum to 1"
    print("[Pipeline integration] PASSED -- fit/predict/predict_proba all "
          "work correctly through a scikit-learn Pipeline\n")
 
 
if __name__ == "__main__":
    print("Running Perceptron test suite...\n")
    _test_linearly_separable()
    _test_xor_should_fail()
    _test_sklearn_pipeline_integration()
    print("All tests passed.")
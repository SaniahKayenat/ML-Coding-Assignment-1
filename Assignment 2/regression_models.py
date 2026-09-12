"""regression_models.py -- from-scratch linear regression for A2: Predicting Car Price.

Based on the LinearRegression class from "03 - Regularization.ipynb", extended in Task 1
(r2, xavier/zeros init, momentum, feature-importance plot) and Task 2 (lines marked "A2").

Contents
    LinearRegression      gradient-descent linear regression with K-fold CV (batch / mini / sto)
    NoRegularization      zero penalty          -> "normal" regression
    Lasso, Ridge          L1 / L2 penalties
    NormalRegression, RidgeRegression, LassoRegression, PolynomialRegression
                          convenience subclasses used in the experiment

Usage (in the notebook or the web app):
    from regression_models import RidgeRegression
    model = RidgeRegression(method='mini', lr=0.01, l=0.001, weight_init='xavier', momentum=0.9)
    model.fit(X_train, y_train)          # X must already include the intercept column of ones
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class LinearRegression(object):
    """Linear regression trained with gradient descent (batch / mini-batch / stochastic) and K-fold CV."""

    kfold = KFold(n_splits=5)

    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=500, bs=50,
                 cv=kfold, weight_init='zeros', momentum=0.0, verbose=True):
        self.lr             = lr
        self.num_epochs     = num_epochs
        self.bs             = bs
        self.method         = method
        self.cv             = cv
        self.regularization = regularization
        self.weight_init    = weight_init   # 'zeros' (default) or 'xavier'
        self.verbose        = verbose
        # A2: fail early on a typo instead of silently falling back to batch
        if method not in ('batch', 'mini', 'sto'):
            raise ValueError("method must be 'batch', 'mini' or 'sto'")
        if not (0 <= momentum < 1):
            raise ValueError('Momentum must be in [0,1)')
        self.momentum = momentum

    # ------------------------------------------------------------------ init
    def _init_weights(self, n_features):
        if self.weight_init == 'zeros':
            self.theta = np.zeros(n_features)
        elif self.weight_init == 'xavier':
            m = n_features                                   # number of inputs (incl. intercept column)
            lower, upper = -(1.0 / np.sqrt(m)), (1.0 / np.sqrt(m))
            numbers = np.random.rand(n_features)             # U[0,1), one number per weight
            self.theta = lower + numbers * (upper - lower)   # rescale to U[lower, upper)
        else:
            raise ValueError('weight_init must be zeros or xavier!')

    # --------------------------------------------------------------- metrics
    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def r2(self, ytrue, ypred):
        # 1 = perfect fit, 0 = no better than predicting the mean, <0 = worse than the mean
        ss_res = ((ypred - ytrue) ** 2).sum()                # our squared error
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()         # squared error of "always predict the mean"
        return 1 - (ss_res / ss_tot)

    # ------------------------------------------------------------------- fit
    def fit(self, X_train, y_train):
        """K-fold cross-validation. Stores per-fold validation MSE and r2."""
        self.kfold_scores = list()   # validation MSE per fold
        self.kfold_r2     = list()   # A2: validation r2 per fold (the task compares both)
        self.epochs_run   = list()   # A2: how many epochs each fold needed before early stopping
        self.val_history  = list()   # A2: validation-loss curve of each fold (for plots / MLflow)
        self.diverged     = False    # A2: True if the loss blew up to inf/nan

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]

            # fresh weights and zero "velocity" for every fold, so no fold leaks into the next
            self._init_weights(X_cross_train.shape[1])
            self.prev_step = np.zeros_like(self.theta)

            self.val_loss_old = np.inf
            history = []
            for epoch in range(self.num_epochs):
                # shuffle so the order of rows does not bias the updates
                perm = np.random.permutation(X_cross_train.shape[0])
                X_cross_train = X_cross_train[perm]
                y_cross_train = y_cross_train[perm]

                # errstate: a diverging run overflows; we detect that below instead of
                # flooding the notebook with RuntimeWarnings
                with np.errstate(over='ignore', invalid='ignore'):
                    self._run_epoch(X_cross_train, y_cross_train)
                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                history.append(val_loss_new)

                # A2: stop if the loss exploded (e.g. too large lr * momentum)
                if not np.isfinite(val_loss_new):
                    self.diverged = True
                    break
                # early stopping: validation loss no longer changing
                if np.allclose(val_loss_new, self.val_loss_old):
                    break
                self.val_loss_old = val_loss_new

            self.val_history.append(history)
            self.epochs_run.append(epoch + 1)
            if self.diverged:
                # the remaining folds use the same settings and would blow up too, so we
                # stop here and report the configuration as diverged
                self.kfold_scores.append(np.nan)
                self.kfold_r2.append(np.nan)
                if self.verbose:
                    print(f"Fold {fold}: diverged at epoch {epoch + 1}")
                break
            with np.errstate(over='ignore', invalid='ignore'):
                self.kfold_scores.append(val_loss_new)
                self.kfold_r2.append(self.r2(y_cross_val, yhat_val))
            if self.verbose:
                print(f"Fold {fold}: MSE={val_loss_new:.4f}  r2={self.kfold_r2[-1]:.4f}  epochs={epoch + 1}")
        return self

    def fit_final(self, X, y):
        """A2: after CV has picked the settings, train ONE model on the whole training set.
        (After fit(), theta only holds the weights of the last fold, trained on 80% of the data.)
        There is no validation fold here, so early stopping watches the training loss."""
        self._init_weights(X.shape[1])
        self.prev_step = np.zeros_like(self.theta)
        self.train_history = []
        loss_old = np.inf
        for epoch in range(self.num_epochs):
            perm = np.random.permutation(X.shape[0])
            X, y = X[perm], y[perm]
            self._run_epoch(X, y)
            loss = self.mse(y, self.predict(X))
            self.train_history.append(loss)
            if np.allclose(loss, loss_old):
                break
            loss_old = loss
        return self

    def _run_epoch(self, X, y):
        """One pass over the data with the chosen gradient-descent flavour."""
        if self.method == 'sto':
            # A2: stochastic GD -> one update per sample.
            # X[i:i+1] keeps the (1, n) matrix shape and y[i:i+1] keeps shape (1,),
            # so predict() and mse() work unchanged.
            for i in range(X.shape[0]):
                self._train(X[i:i + 1], y[i:i + 1])
        elif self.method == 'mini':
            # mini-batch GD -> one update per batch of self.bs samples
            for batch_idx in range(0, X.shape[0], self.bs):
                self._train(X[batch_idx:batch_idx + self.bs], y[batch_idx:batch_idx + self.bs])
        else:
            # batch GD -> one update per epoch, using every sample
            self._train(X, y)

    # ----------------------------------------------------------------- train
    def _train(self, X, y):
        # 1. predict
        yhat = self.predict(X)

        # 2. gradient of the MSE part
        m = X.shape[0]
        grad = (1 / m) * X.T @ (yhat - y)                    # (n, m) @ (m,) -> (n,)

        # A2: gradient of the penalty. We do NOT penalise the intercept theta[0]:
        # our target is log(price) ~ 12.9, so shrinking the bias towards 0 would shift every
        # prediction down. Regularisation is meant to shrink feature weights only.
        reg_grad = np.array(self.regularization.derivation(self.theta), dtype=float) * np.ones_like(self.theta)
        reg_grad[0] = 0.0
        grad = grad + reg_grad

        # 3. update
        if self.momentum > 0:
            # velocity = current gradient step + a fraction of the previous step:
            # directions where gradients keep agreeing build up speed, directions where they
            # flip sign partly cancel (less oscillation).
            # Note: this is the standard "heavy-ball" form theta <- theta - (lr*g + beta*v).
            # The handout writes theta - step + momentum*prev_step with prev_step = lr*g, which
            # would move AGAINST the previous direction, so we keep the velocity form.
            step = self.lr * grad + self.momentum * self.prev_step
            self.theta = self.theta - step
            self.prev_step = step
        else:
            self.theta = self.theta - self.lr * grad        # plain gradient descent

        return self.mse(y, yhat)

    # --------------------------------------------------------------- predict
    def predict(self, X):
        return X @ self.theta                                # (m, n) @ (n,) = (m,)

    def _coef(self):
        return self.theta[1:]                                # feature weights (intercept excluded)

    def _bias(self):
        return self.theta[0]

    # ------------------------------------------------------ feature importance
    def plot_feature_importance(self, feature_names=None, top_k=None):
        """Horizontal bar chart of the coefficients, sorted by |coefficient|.
        Only meaningful because the numeric inputs were standardised before fitting."""
        coefs = self._coef()
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(coefs))]
        if len(feature_names) != len(coefs):
            raise ValueError(f"got {len(feature_names)} feature names but {len(coefs)} coefficients")
        feature_names = np.array(feature_names)

        order = np.argsort(np.abs(coefs))                    # ascending -> largest ends up on top
        coefs, feature_names = coefs[order], feature_names[order]
        if top_k is not None:                                # A2: long lists (e.g. polynomial) -> keep the top k
            coefs, feature_names = coefs[-top_k:], feature_names[-top_k:]

        colors = ['tab:blue' if c > 0 else 'tab:red' for c in coefs]   # blue raises price, red lowers it
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(coefs))))
        ax.barh(feature_names, coefs, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("Coefficient (effect on log price per unit of the scaled feature)")
        ax.set_title("Feature importance (sorted by |coefficient|)")
        plt.tight_layout()
        return fig


# ------------------------------------------------------------ regularisers
class NoRegularization:
    """A2: 'normal' linear regression -> no penalty at all."""
    def __call__(self, theta):
        return 0.0
    def derivation(self, theta):
        return np.zeros_like(theta)


class Lasso:
    """L1 penalty: l * sum(|theta|). Pushes small weights towards exactly 0."""
    def __init__(self, l):
        self.l = l
    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))
    def derivation(self, theta):
        return self.l * np.sign(theta)


class Ridge:
    """L2 penalty: l * sum(theta^2). Shrinks all weights smoothly."""
    def __init__(self, l):
        self.l = l
    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))
    def derivation(self, theta):
        return self.l * 2 * theta


class NormalRegression(LinearRegression):
    """Plain linear regression (no penalty)."""
    def __init__(self, method, lr, **kwargs):
        super().__init__(NoRegularization(), lr=lr, method=method, **kwargs)


class LassoRegression(LinearRegression):
    """Linear regression with an L1 (Lasso) penalty."""
    def __init__(self, method, lr, l, **kwargs):
        super().__init__(Lasso(l), lr=lr, method=method, **kwargs)


class RidgeRegression(LinearRegression):
    """Linear regression with an L2 (Ridge) penalty."""
    def __init__(self, method, lr, l, **kwargs):
        super().__init__(Ridge(l), lr=lr, method=method, **kwargs)


class PolynomialRegression(NormalRegression):
    """A2: 'polynomial' = normal (unpenalised) regression trained on polynomial features.
    The class itself is identical; the difference is the input matrix, which comes from
    the polynomial preprocessor (degree-2 terms of the numeric columns)."""
    pass

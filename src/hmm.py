"""Gaussian HMM in NumPy: Baum-Welch training and Viterbi decoding."""

from __future__ import annotations

import numpy as np


class GaussianHMM:
    """Continuous-observation HMM with diagonal-covariance Gaussian emissions."""

    def __init__(
        self,
        n_states: int,
        n_features: int,
        state_names: list[str] | None = None,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.state_names = state_names or [f"S{i}" for i in range(n_states)]
        self.rng = np.random.default_rng(random_state)

        self.pi: np.ndarray | None = None  # (N,)
        self.A: np.ndarray | None = None  # (N, N)
        self.means: np.ndarray | None = None  # (N, D)
        self.vars: np.ndarray | None = None  # (N, D) diagonal variances
        self.log_likelihoods_: list[float] = []

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def initialize(self, X: np.ndarray, labels: np.ndarray | None = None) -> None:
        """
        Initialize parameters.

        If labels are provided (supervised warm-start), estimate emissions
        per class and a simple transition matrix from consecutive labels.
        Otherwise use random / k-means-like seeding.
        """
        N, D = self.n_states, self.n_features
        assert X.shape[1] == D

        if labels is not None:
            self.pi = np.ones(N) / N
            self.A = np.full((N, N), 1e-2)
            self.means = np.zeros((N, D))
            self.vars = np.ones((N, D))

            for i in range(N):
                Xi = X[labels == i]
                if len(Xi) == 0:
                    self.means[i] = X.mean(axis=0) + self.rng.normal(0, 0.1, D)
                    self.vars[i] = np.maximum(X.var(axis=0), 1e-3)
                else:
                    self.means[i] = Xi.mean(axis=0)
                    self.vars[i] = np.maximum(Xi.var(axis=0), 1e-3)

            for t in range(len(labels) - 1):
                self.A[labels[t], labels[t + 1]] += 1.0
            self.A = self.A / self.A.sum(axis=1, keepdims=True)

            # empirical start distribution
            counts = np.bincount(labels, minlength=N).astype(float)
            self.pi = counts / counts.sum()
        else:
            self.pi = np.ones(N) / N
            self.A = np.full((N, N), 1.0 / N)
            # slight preference to stay in same state (realistic for activities)
            self.A = 0.7 * np.eye(N) + 0.3 * self.A
            self.A = self.A / self.A.sum(axis=1, keepdims=True)

            idx = self.rng.choice(len(X), size=N, replace=False)
            self.means = X[idx].copy()
            global_var = np.maximum(X.var(axis=0), 1e-3)
            self.vars = np.tile(global_var, (N, 1))

    # ------------------------------------------------------------------ #
    # Emissions
    # ------------------------------------------------------------------ #
    def _log_gaussian(self, x: np.ndarray) -> np.ndarray:
        """Log emission density for one observation under each state. Shape (N,)."""
        # log N(x | mu, diag(var))
        log_prob = -0.5 * (
            np.sum(np.log(2 * np.pi * self.vars), axis=1)
            + np.sum((x - self.means) ** 2 / self.vars, axis=1)
        )
        return log_prob

    def _log_emissions(self, X: np.ndarray) -> np.ndarray:
        """Log emission matrix B_log[t, i] = log p(x_t | z_t = i). Shape (T, N)."""
        T = len(X)
        B = np.zeros((T, self.n_states))
        for t in range(T):
            B[t] = self._log_gaussian(X[t])
        return B

    # ------------------------------------------------------------------ #
    # Forward / Backward (log-space)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _logsumexp(a: np.ndarray, axis: int | None = None):
        a = np.asarray(a, dtype=float)
        if axis is None:
            a_max = np.max(a)
            return float(a_max + np.log(np.sum(np.exp(a - a_max))))
        a_max = np.max(a, axis=axis, keepdims=True)
        out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
        return np.squeeze(out, axis=axis)

    def _forward(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        T, N = log_B.shape
        log_alpha = np.zeros((T, N))
        log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]
        log_A = np.log(self.A + 1e-300)

        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = self._logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_B[t, j]

        log_likelihood = float(self._logsumexp(log_alpha[-1]))
        return log_alpha, log_likelihood

    def _backward(self, log_B: np.ndarray) -> np.ndarray:
        T, N = log_B.shape
        log_beta = np.zeros((T, N))
        log_A = np.log(self.A + 1e-300)

        for t in range(T - 2, -1, -1):
            for i in range(N):
                log_beta[t, i] = self._logsumexp(
                    log_A[i, :] + log_B[t + 1, :] + log_beta[t + 1, :]
                )
        return log_beta

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        log_B = self._log_emissions(X)
        log_alpha, ll = self._forward(log_B)
        log_beta = self._backward(log_B)
        T, N = log_B.shape
        log_A = np.log(self.A + 1e-300)

        # gamma[t, i] = P(z_t = i | X)
        log_gamma = log_alpha + log_beta
        log_gamma -= self._logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        # xi[t, i, j] = P(z_t=i, z_{t+1}=j | X)
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            log_xi = (
                log_alpha[t][:, None]
                + log_A
                + log_B[t + 1][None, :]
                + log_beta[t + 1][None, :]
            )
            log_xi -= self._logsumexp(log_xi.ravel())
            xi[t] = np.exp(log_xi)

        return gamma, xi, ll

    # ------------------------------------------------------------------ #
    # Baum–Welch
    # ------------------------------------------------------------------ #
    def fit(
        self,
        sequences: list[np.ndarray],
        labels_list: list[np.ndarray] | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = True,
    ) -> "GaussianHMM":
        """
        Train with Baum–Welch across one or more observation sequences.

        Stopping criterion: |ll_new - ll_old| < tol (log-likelihood convergence).
        """
        all_X = np.vstack(sequences)
        all_labels = None
        if labels_list is not None:
            all_labels = np.concatenate(labels_list)

        if self.pi is None:
            self.initialize(all_X, all_labels)

        self.log_likelihoods_ = []
        prev_ll = -np.inf

        for it in range(max_iter):
            # accumulate sufficient statistics across sequences
            gamma_sum = np.zeros(self.n_states)
            pi_acc = np.zeros(self.n_states)
            A_num = np.zeros((self.n_states, self.n_states))
            mean_num = np.zeros((self.n_states, self.n_features))
            var_num = np.zeros((self.n_states, self.n_features))
            total_ll = 0.0
            n_obs = 0

            for X in sequences:
                gamma, xi, ll = self._e_step(X)
                total_ll += ll
                n_obs += len(X)

                pi_acc += gamma[0]
                A_num += xi.sum(axis=0)
                gamma_sum += gamma.sum(axis=0)
                mean_num += gamma.T @ X
                # for variances, use current means (standard EM for Gaussian HMM)
                for i in range(self.n_states):
                    diff = X - self.means[i]
                    var_num[i] += (gamma[:, i][:, None] * (diff**2)).sum(axis=0)

            # M-step
            self.pi = pi_acc / pi_acc.sum()
            self.A = A_num / np.maximum(A_num.sum(axis=1, keepdims=True), 1e-12)
            self.means = mean_num / np.maximum(gamma_sum[:, None], 1e-12)
            self.vars = var_num / np.maximum(gamma_sum[:, None], 1e-12)
            self.vars = np.maximum(self.vars, 1e-3)  # floor for stability

            avg_ll = total_ll / max(len(sequences), 1)
            self.log_likelihoods_.append(avg_ll)

            if verbose:
                print(f"Iter {it + 1:03d} | avg log-likelihood = {avg_ll:.6f}")

            if abs(avg_ll - prev_ll) < tol:
                if verbose:
                    print(f"Converged at iteration {it + 1} (|dll| < {tol})")
                break
            prev_ll = avg_ll

        return self

    # ------------------------------------------------------------------ #
    # Viterbi
    # ------------------------------------------------------------------ #
    def decode(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Viterbi decoding: most likely state sequence.

        Returns
        -------
        path : (T,) int state indices
        log_prob : log probability of the best path
        """
        T = len(X)
        N = self.n_states
        log_B = self._log_emissions(X)
        log_A = np.log(self.A + 1e-300)

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        delta[0] = np.log(self.pi + 1e-300) + log_B[0]

        for t in range(1, T):
            for j in range(N):
                scores = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = scores[psi[t, j]] + log_B[t, j]

        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path, float(delta[-1, path[-1]])

    def predict(self, X: np.ndarray) -> np.ndarray:
        path, _ = self.decode(X)
        return path

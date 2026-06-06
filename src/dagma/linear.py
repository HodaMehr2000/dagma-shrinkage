import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from scipy.special import expit as sigmoid
from sklearn.covariance import LedoitWolf
from tqdm.auto import tqdm
import typing


__all__ = ["DagmaLinear"]

class DagmaLinear:
    """
    A Python object that contains the implementation of DAGMA for linear models using numpy and scipy.
    """
    
    def __init__(self, loss_type: str, verbose: bool = False, dtype: type = np.float64) -> None:
        r"""
        Parameters
        ----------
        loss_type : str
            One of ["l2", "logistic"]. ``l2`` refers to the least squares loss, while ``logistic``
            refers to the logistic loss. For continuous data: use ``l2``. For discrete 0/1 data: use ``logistic``.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.linear.DagmaLinear.fit`. Defaults to ``False``.
        dtype : type, optional
           Defines the float precision, for large number of nodes it is recommened to use ``np.float64``. 
           Defaults to ``np.float64``.
        """
        super().__init__()
        losses = ['l2', 'logistic']
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None
            
    def _score(self, W: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value and gradient of the score function.
    
        In the l2 case, self.cov is either:
            - empirical covariance, for standard DAGMA
            - shrinkage covariance, for SCGL/DAGMA-shrinkage
        """
        if self.loss_type == 'l2':
            dif = self.Id - W
            rhs = self.cov @ dif
            loss = 0.5 * np.trace(dif.T @ rhs)
            G_loss = -rhs
    
        elif self.loss_type == 'logistic':
            R = self.X @ W
            loss = 1.0 / self.n * (np.logaddexp(0, R) - self.X * R).sum()
            G_loss = (1.0 / self.n * self.X.T) @ sigmoid(R) - self.cov
    
        return loss, G_loss

    def _condition_number(self, A: np.ndarray, eps: float = 1e-12) -> float:
        """
        Compute spectral condition number for a symmetric covariance matrix.
        """
        eigvals = np.linalg.eigvalsh(A)
        lam_max = float(np.max(eigvals))
        lam_min = float(np.min(eigvals))
        lam_min = max(lam_min, eps)
        return lam_max / lam_min

    def _compute_ledoit_wolf_shrinkage_cov(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Ledoit-Wolf shrinkage covariance.
    
        X must already be centered.
    
        This function stores:
            self.cov_sample
            self.cov_shrink
            self.shrinkage_rho
            self.shrinkage_mu
            self.cond_sample
            self.cond_shrink
    
        and returns:
            shrinkage covariance matrix
        """
        X = np.asarray(X, dtype=self.dtype)
        n, d = X.shape
    
        sample_cov = (X.T @ X) / float(n)
    
        lw = LedoitWolf().fit(X)
        shrink_cov = lw.covariance_.astype(self.dtype)
    
        self.cov_sample = sample_cov.astype(self.dtype)
        self.cov_shrink = shrink_cov
        self.shrinkage_rho = float(lw.shrinkage_)
        self.shrinkage_mu = float(np.trace(sample_cov) / d)
    
        self.cond_sample = self._condition_number(self.cov_sample)
        self.cond_shrink = self._condition_number(self.cov_shrink)
    
        return self.cov_shrink
    
    
    def _h(self, W: np.ndarray, s: float = 1.0) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value and gradient of the logdet acyclicity constraint.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix
        s : float, optional
            Controls the domain of M-matrices. Defaults to 1.0.

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            h value, and gradient of h
        """
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h

    def _func(self, W: np.ndarray, mu: float, s: float = 1.0) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value of the penalized objective function.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix
        mu : float
            Weight of the score function.
        s : float, optional
            Controls the domain of M-matrices. Defaults to 1.0.

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            Objective value, and gradient of the objective
        """
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def _adam_update(self, grad: np.ndarray, iter: int, beta_1: float, beta_2: float) -> np.ndarray:
        r"""
        Performs one update of Adam.

        Parameters
        ----------
        grad : np.ndarray
            Current gradient of the objective.
        iter : int
            Current iteration number.
        beta_1 : float
            Adam hyperparameter.
        beta_2 : float
            Adam hyperparameter.

        Returns
        -------
        np.ndarray
            Updates the gradient by the Adam method.
        """
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad
    
    def minimize(
            self,
            W: np.ndarray,
            mu: float,
            max_iter: int,
            s: float,
            lr: float,
            tol: float = 1e-6,
            beta_1: float = 0.99,
            beta_2: float = 0.999,
            pbar: typing.Optional[tqdm] = None,
        ) -> typing.Tuple[np.ndarray, bool]:
        r"""
        Solves one inner optimization problem in the DAGMA central path:
    
            min_W  mu * (Q(W; X) + lambda1 * ||W||_1) + h_logdet(W)
    
        In the SCGL version, Q(W; X) automatically uses the shrinkage covariance
        if self.cov has already been set to self.cov_shrink inside fit().
        """
    
        obj_prev = np.inf
        self.opt_m, self.opt_v = 0, 0
    
        self.vprint(
            f"\n\nMinimize with -- mu:{mu} -- lr:{lr} -- s:{s} "
            f"-- l1:{self.lambda1} for {max_iter} max iterations"
        )
    
        # ------------------------------------------------------------------
        # Mask for forced included edges
        # ------------------------------------------------------------------
        mask_inc = np.zeros((self.d, self.d), dtype=self.dtype)
        if self.inc_c is not None:
            mask_inc[self.inc_r, self.inc_c] = -2.0 * mu * self.lambda1
    
        # ------------------------------------------------------------------
        # Mask for excluded edges + zero diagonal
        # ------------------------------------------------------------------
        mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
    
        if self.exc_c is not None:
            mask_exc[self.exc_r, self.exc_c] = 0.0
    
        # No self-loops
        np.fill_diagonal(mask_exc, 0.0)
    
        # Make sure the starting point obeys the mask
        W = W.astype(self.dtype, copy=True)
        W *= mask_exc
    
        # ------------------------------------------------------------------
        # Helper: check DAGMA M-matrix domain
        # ------------------------------------------------------------------
        def _inverse_if_valid(W_current: np.ndarray):
            """
            DAGMA requires:
                sI - W o W
            to stay inside the M-matrix domain.
    
            We use the inverse non-negativity condition as in the original code.
            """
            try:
                M_inv = sla.inv(s * self.Id - W_current * W_current)
            except Exception:
                return None
    
            if not np.all(np.isfinite(M_inv)):
                return None
    
            # Tiny negative values can happen from numerical noise.
            if np.any(M_inv < -1e-12):
                return None
    
            return M_inv + 1e-16
    
        # ------------------------------------------------------------------
        # Main optimization loop
        # ------------------------------------------------------------------
        for iter_idx in range(1, max_iter + 1):
    
            # --------------------------------------------------------------
            # 1. Check current point is valid and compute inverse
            # --------------------------------------------------------------
            M_inv = _inverse_if_valid(W)
    
            if M_inv is None:
                self.vprint(f"W is outside the M-matrix domain for s={s} at iteration {iter_idx}")
                return W, False
    
            # --------------------------------------------------------------
            # 2. Score gradient
            #    This is the important part:
            #    _score uses self.cov.
            #    If self.cov = self.cov_shrink, then this is SCGL.
            # --------------------------------------------------------------
            _, G_loss = self._score(W)
            G_score = mu * G_loss
    
            # --------------------------------------------------------------
            # 3. Full objective gradient
            # --------------------------------------------------------------
            G_h = 2.0 * W * M_inv.T
    
            Gobj = (
                G_score
                + mu * self.lambda1 * np.sign(W)
                + G_h
                + mask_inc * np.sign(W)
            )
    
            if not np.all(np.isfinite(Gobj)):
                self.vprint(f"Non-finite gradient at iteration {iter_idx}")
                return W, False
    
            # --------------------------------------------------------------
            # 4. Adam direction
            # --------------------------------------------------------------
            grad = self._adam_update(Gobj, iter_idx, beta_1, beta_2)
    
            if not np.all(np.isfinite(grad)):
                self.vprint(f"Non-finite Adam direction at iteration {iter_idx}")
                return W, False
    
            # --------------------------------------------------------------
            # 5. Backtracking line search to stay inside DAGMA domain
            # --------------------------------------------------------------
            step_lr = lr
            accepted = False
    
            while step_lr > 1e-16:
                W_trial = W - step_lr * grad
                W_trial *= mask_exc
    
                # Numerical safety: no self-loops
                np.fill_diagonal(W_trial, 0.0)
    
                M_trial_inv = _inverse_if_valid(W_trial)
    
                if M_trial_inv is not None:
                    W = W_trial
                    accepted = True
                    break
    
                step_lr *= 0.5
    
            if not accepted:
                self.vprint(
                    f"Could not find a valid step at iteration {iter_idx}; "
                    f"last step_lr={step_lr:.2e}"
                )
                return W, True
    
            # --------------------------------------------------------------
            # 6. Check convergence
            # --------------------------------------------------------------
            if iter_idx % self.checkpoint == 0 or iter_idx == max_iter:
                obj_new, score, h = self._func(W, mu, s)
    
                self.vprint(f"\nInner iteration {iter_idx}")
                self.vprint(f"\th(W_est): {h:.4e}")
                self.vprint(f"\tscore(W_est): {score:.4e}")
                self.vprint(f"\tobj(W_est): {obj_new:.4e}")
                self.vprint(f"\tstep_lr: {step_lr:.4e}")
    
                if not np.isfinite(obj_new):
                    self.vprint(f"Non-finite objective at iteration {iter_idx}")
                    return W, False
    
                if np.isfinite(obj_prev):
                    rel_change = abs(obj_prev - obj_new) / max(abs(obj_prev), 1.0)
    
                    if rel_change <= tol:
                        if pbar is not None:
                            pbar.update(max_iter - iter_idx + 1)
                        break
    
                obj_prev = obj_new
    
            if pbar is not None:
                pbar.update(1)
    
        W *= mask_exc
        np.fill_diagonal(W, 0.0)
    
        return W, True
    
    def fit(
            self,
            X: np.ndarray,
            lambda1: float = 0.03,
            w_threshold: float = 0.3,
            T: int = 5,
            mu_init: float = 1.0,
            mu_factor: float = 0.1,
            s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6],
            warm_iter: int = 3e4,
            max_iter: int = 6e4,
            lr: float = 0.0003,
            checkpoint: int = 1000,
            beta_1: float = 0.99,
            beta_2: float = 0.999,
            exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
            include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
            use_shrinkage: bool = True,
        ) -> np.ndarray:
        """
        Runs DAGMA and returns a weighted adjacency matrix.
    
        If use_shrinkage=True and loss_type='l2', the empirical covariance
        in the least-squares score is replaced by Ledoit-Wolf shrinkage covariance.
        """
    
        # ------------------------------------------------------------
        # Initialize variables safely
        # ------------------------------------------------------------
        self.X = np.asarray(X, dtype=self.dtype).copy()
        self.lambda1 = lambda1
        self.checkpoint = checkpoint
    
        self.n, self.d = self.X.shape
        self.Id = np.eye(self.d, dtype=self.dtype)
    
        # ------------------------------------------------------------
        # Center data for l2 loss, matching DAGMA linear behavior
        # ------------------------------------------------------------
        if self.loss_type == 'l2':
            self.X -= self.X.mean(axis=0, keepdims=True)
    
        # ------------------------------------------------------------
        # Edge constraints
        # ------------------------------------------------------------
        self.exc_r, self.exc_c = None, None
        self.inc_r, self.inc_c = None, None
    
        if exclude_edges is not None:
            if (
                isinstance(exclude_edges, tuple)
                and len(exclude_edges) > 0
                and isinstance(exclude_edges[0], tuple)
                and np.all(np.array([len(e) for e in exclude_edges]) == 2)
            ):
                self.exc_r, self.exc_c = zip(*exclude_edges)
            else:
                raise ValueError(
                    "exclude_edges should be a tuple of edges, e.g., ((1,2), (2,3))"
                )
    
        if include_edges is not None:
            if (
                isinstance(include_edges, tuple)
                and len(include_edges) > 0
                and isinstance(include_edges[0], tuple)
                and np.all(np.array([len(e) for e in include_edges]) == 2)
            ):
                self.inc_r, self.inc_c = zip(*include_edges)
            else:
                raise ValueError(
                    "include_edges should be a tuple of edges, e.g., ((1,2), (2,3))"
                )
    
        # ------------------------------------------------------------
        # Covariance choice: standard DAGMA vs SCGL-DAGMA
        # ------------------------------------------------------------
        if self.loss_type == 'l2':
            self.cov_sample = (self.X.T @ self.X) / float(self.n)
            self.cond_sample = self._condition_number(self.cov_sample)
    
            if use_shrinkage:
                self.cov = self._compute_ledoit_wolf_shrinkage_cov(self.X)
                self.use_shrinkage = True
    
                self.vprint(
                    f"Using Ledoit-Wolf shrinkage covariance | "
                    f"rho={self.shrinkage_rho:.6f}, "
                    f"cond(sample)={self.cond_sample:.4e}, "
                    f"cond(shrink)={self.cond_shrink:.4e}, "
                    f"improvement={self.cond_sample / self.cond_shrink:.2f}x"
                )
            else:
                self.cov = self.cov_sample
                self.cov_shrink = None
                self.shrinkage_rho = 0.0
                self.shrinkage_mu = float(np.trace(self.cov_sample) / self.d)
                self.cond_shrink = None
                self.use_shrinkage = False
    
                self.vprint(
                    f"Using empirical covariance | "
                    f"cond(sample)={self.cond_sample:.4e}"
                )
    
        elif self.loss_type == 'logistic':
            self.cov = (self.X.T @ self.X) / float(self.n)
            self.use_shrinkage = False
    
            if use_shrinkage:
                self.vprint(
                    "Warning: use_shrinkage=True was ignored because "
                    "shrinkage covariance is only used for loss_type='l2'."
                )
    
        # ------------------------------------------------------------
        # Initialize W at zero matrix
        # ------------------------------------------------------------
        self.W_est = np.zeros((self.d, self.d), dtype=self.dtype)
        mu = mu_init
    
        # ------------------------------------------------------------
        # Prepare s schedule
        # ------------------------------------------------------------
        if isinstance(s, list):
            if len(s) < T:
                self.vprint(
                    f"Length of s is {len(s)}, using last value in s "
                    f"for iteration t >= {len(s)}"
                )
                s = s + (T - len(s)) * [s[-1]]
    
        elif isinstance(s, (int, float)):
            s = T * [s]
    
        else:
            raise ValueError("s should be a list, int, or float.")
    
        # ------------------------------------------------------------
        # Start DAGMA central path
        # ------------------------------------------------------------
        with tqdm(total=(T - 1) * int(warm_iter) + int(max_iter)) as pbar:
            for i in range(int(T)):
                self.vprint(f"\nIteration -- {i + 1}:")
    
                lr_adam = lr
                success = False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
    
                while success is False:
                    W_temp, success = self.minimize(
                        self.W_est.copy(),
                        mu,
                        inner_iters,
                        s[i],
                        lr=lr_adam,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        pbar=pbar,
                    )
    
                    if success is False:
                        self.vprint("Retrying with larger s and smaller learning rate")
                        lr_adam *= 0.5
                        s[i] += 0.1
    
                self.W_est = W_temp
                mu *= mu_factor
    
        # ------------------------------------------------------------
        # Store pre-threshold values
        # ------------------------------------------------------------
        self.h_before_threshold, _ = self._h(self.W_est)
        self.score_before_threshold, _ = self._score(self.W_est)
    
        # ------------------------------------------------------------
        # Threshold weak edges
        # ------------------------------------------------------------
        self.W_est[np.abs(self.W_est) < w_threshold] = 0.0
        np.fill_diagonal(self.W_est, 0.0)
    
        # ------------------------------------------------------------
        # Store final values after threshold
        # ------------------------------------------------------------
        self.h_final, _ = self._h(self.W_est)
        self.score_final, _ = self._score(self.W_est)
    
        return self.W_est

def test():
    from . import utils
    from timeit import default_timer as timer
    utils.set_random_seed(1)
    
    n, d, s0 = 500, 20, 20 # the ground truth is a DAG of 20 nodes and 20 edges in expectation
    graph_type, sem_type = 'ER', 'gauss'
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    
    model = DagmaLinear(loss_type='l2')
    start = timer()
    W_est = model.fit(X, lambda1=0.02)
    end = timer()
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)
    print(f'time: {end-start:.4f}s')
    
if __name__ == '__main__':
    test()

    

    

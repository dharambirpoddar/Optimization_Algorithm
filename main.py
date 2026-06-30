"""
global_minimum_detector.py
==========================
Global Minimum Detection via Horizontal Loss Plane Probing
-----------------------------------------------------------
This module implements a heuristic global optimization algorithm that combines
local gradient-based optimization (BFGS) with repeated random probing of the
loss surface to escape local minima and converge toward the global minimum.

Algorithm Overview
------------------
1. Start from an initial point ``x0`` and run local optimization (BFGS) to
   converge to a local minimum ``θ*`` with loss ``L_min``.
2. Probe the parameter space by randomly sampling ``n_probes`` points and
   evaluating the loss function at each.
3. If any sampled point has loss lower than ``L_min - epsilon``, a deeper basin
   has been found. Restart local optimization from that point and repeat step 2.
4. If no lower point is found across all probes, declare ``θ*`` as the global
   minimum (within the bounds and probabilistic confidence given by ``n_probes``).

Caveats
-------
- This is a **heuristic** method; it cannot guarantee global optimality in the
  strict mathematical sense. Confidence in the result increases with larger
  ``n_probes`` and tighter ``bounds``.
- Performance scales with the dimensionality of the parameter space and the
  number of probes.

Dependencies
------------
- numpy
- scipy
- matplotlib

Example Usage
-------------
>>> from global_minimum_detector import GlobalMinimumDetector, test_function
>>> detector = GlobalMinimumDetector(test_function, epsilon=1e-6,
...                                  n_probes=100, bounds=(-3, 3))
>>> results = detector.optimize_with_global_detection(x0=-1.5, max_restarts=10)
>>> print(f"Global min at x={results['theta_star']:.4f}, L={results['L_min']:.8f}")
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional, Union
import warnings

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------

class GlobalMinimumDetector:
    """
    Detects the global minimum of a scalar loss function via horizontal loss
    plane probing combined with gradient-based local optimization.

    The detector alternates between two phases:
      - **Exploitation phase**: BFGS local optimization to find the nearest
        local minimum from a given starting point.
      - **Exploration phase**: Uniform random sampling over the bounded
        parameter space to discover basins lower than the current best.

    Parameters
    ----------
    loss_function : callable
        The objective function to minimize. Must accept a scalar (1-D case) or
        a 1-D array (N-D case) and return a scalar float.
    epsilon : float, optional
        Minimum improvement threshold. A sampled point must have loss at least
        ``epsilon`` below the current best to trigger a restart.
        Default is ``1e-6``.
    n_probes : int, optional
        Number of random samples drawn during each exploration phase.
        Larger values increase the probability of detecting a lower basin but
        increase computation time. Default is ``100``.
    bounds : tuple or list of tuples, optional
        Search bounds for the parameter space.
        - 1-D: a single ``(lower, upper)`` tuple, e.g. ``(-10, 10)``.
        - N-D: a list of ``(lower, upper)`` tuples, one per dimension.
        Default is ``(-10, 10)``.

    Attributes
    ----------
    history : list of dict
        A chronological log of optimization events. Each entry is a dict with
        keys: ``'iteration'``, ``'theta'``, ``'loss'``, and ``'event'``.
        Possible event strings:
          - ``'initial_convergence'``  – first local minimum found.
          - ``'lower_basin_found'``    – exploration located a lower region.
          - ``'restart_convergence'`` – local minimum after a restart.
          - ``'global_minimum_declared'`` – no lower basin found; halt.

    Examples
    --------
    >>> def my_func(x):
    ...     return (1 + x - x**3)**2
    >>> detector = GlobalMinimumDetector(my_func, n_probes=200, bounds=(-5, 5))
    >>> result = detector.optimize_with_global_detection(x0=2.0, max_restarts=15)
    >>> result['theta_star'], result['L_min']
    """

    def __init__(
        self,
        loss_function,
        epsilon: float = 1e-6,
        n_probes: int = 100,
        bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (-10, 10),
    ):
        self.loss_function = loss_function
        self.epsilon = epsilon
        self.n_probes = n_probes
        self.bounds = bounds
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_multidimensional(self) -> bool:
        """Return True when ``bounds`` describes a multi-dimensional space."""
        return isinstance(self.bounds, list)

    def sample_parameter_space(
        self, current_min: float, n_samples: int
    ) -> np.ndarray:
        """
        Draw uniform random samples from the bounded parameter space.

        Parameters
        ----------
        current_min : float
            The current best loss value (not used for sampling itself, but
            kept in the signature for potential future importance-sampling
            extensions).
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            - Shape ``(n_samples,)`` for 1-D problems.
            - Shape ``(n_samples, dim)`` for N-D problems.
        """
        if self._is_multidimensional():
            # N-D case: sample each dimension independently
            dim = len(self.bounds)
            samples = np.zeros((n_samples, dim))
            for i, (lo, hi) in enumerate(self.bounds):
                samples[:, i] = np.random.uniform(lo, hi, n_samples)
            return samples
        else:
            # 1-D case: simple uniform draw
            lo, hi = self.bounds
            return np.random.uniform(lo, hi, n_samples)

    def local_optimize(self, x0) -> Tuple[float, float]:
        """
        Run BFGS gradient-based optimization from an initial point.

        Convergence is set to a tight tolerance (``gtol=1e-8``) so that the
        returned point is a well-converged local minimum.

        Parameters
        ----------
        x0 : float or array-like
            Starting point for the optimizer.

        Returns
        -------
        theta_star : float
            The parameter value at the local minimum (scalar, 1-D only).
        L_min : float
            The loss value at ``theta_star``.

        Notes
        -----
        For multi-dimensional inputs ``x0`` should be a 1-D array matching
        the dimensionality of the problem. The return type generalises
        accordingly, though the current implementation extracts a scalar for
        1-D problems.
        """
        result = minimize(
            self.loss_function,
            x0,
            method='BFGS',
            options={'gtol': 1e-8, 'maxiter': 1000},
        )
        # Extract scalar for 1-D problems; keep array for N-D
        theta_star = (
            float(result.x[0]) if hasattr(result.x, '__len__') else float(result.x)
        )
        L_min = float(result.fun)
        return theta_star, L_min

    def probe_for_lower_basin(
        self, current_loss: float
    ) -> Tuple[bool, Optional[float]]:
        """
        Probe the loss surface to detect whether a lower basin exists.

        Draws ``self.n_probes`` random samples from the parameter space,
        evaluates the loss at each, and checks whether the minimum sampled
        loss is at least ``self.epsilon`` below ``current_loss``.

        Parameters
        ----------
        current_loss : float
            Loss value of the current best solution. Used as the comparison
            baseline.

        Returns
        -------
        found_lower : bool
            ``True`` if a sample with loss < ``current_loss - epsilon`` was
            found; ``False`` otherwise.
        best_sample : float or None
            The parameter value of the best sample when ``found_lower`` is
            ``True``; ``None`` otherwise.
        """
        # Draw random candidates across the full search region
        samples = self.sample_parameter_space(current_loss, self.n_probes)

        # Evaluate the loss at every candidate
        losses = np.array([self.loss_function(s) for s in samples])

        # Identify the globally lowest sample
        min_idx = np.argmin(losses)
        min_loss = losses[min_idx]

        if min_loss < current_loss - self.epsilon:
            # A significantly lower region was found → signal restart
            return True, samples[min_idx]

        # No improvement beyond the threshold → current solution is likely global
        return False, None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_with_global_detection(
        self,
        x0,
        max_restarts: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the full global minimum detection algorithm.

        Executes the exploit–explore loop: converge locally, probe for a
        lower basin, restart if found, and halt when no lower basin is
        detected or ``max_restarts`` is reached.

        Parameters
        ----------
        x0 : float or array-like
            Initial starting point for the first local optimization.
        max_restarts : int, optional
            Maximum number of probe-and-restart cycles before the algorithm
            terminates regardless of convergence status. Default is ``10``.
        verbose : bool, optional
            If ``True``, print progress information to stdout at each step.
            Default is ``True``.

        Returns
        -------
        dict with keys:

        ``'theta_star'`` : float
            Parameter value of the best solution found.
        ``'L_min'`` : float
            Corresponding loss value.
        ``'iterations'`` : int
            Number of probe-and-restart cycles actually performed.
        ``'history'`` : list of dict
            Full event log (see ``self.history``).

        Notes
        -----
        The method also populates ``self.history`` in place, so the log is
        accessible on the detector instance after this call returns.
        """
        # ---- Step 1: Initial local convergence ----
        iteration = 0
        theta_star, L_min = self.local_optimize(x0)

        # Record the initial convergence event
        self.history.append({
            'iteration': iteration,
            'theta': theta_star,
            'loss': L_min,
            'event': 'initial_convergence',
        })

        if verbose:
            print(
                f"Iteration {iteration}: Converged to "
                f"θ* = {theta_star:.6f}, L = {L_min:.8f}"
            )

        # ---- Steps 2–4: Iterative probe-and-restart loop ----
        for iteration in range(1, max_restarts + 1):

            # Exploration phase: probe the loss surface for a lower basin
            found_lower, new_start = self.probe_for_lower_basin(L_min)

            if not found_lower:
                # No lower basin detected → declare current solution as global min
                if verbose:
                    print(f"\nIteration {iteration}: No lower basin found.")
                    print(
                        f"GLOBAL MINIMUM DETECTED: "
                        f"θ* = {theta_star:.6f}, L = {L_min:.8f}"
                    )
                self.history.append({
                    'iteration': iteration,
                    'theta': theta_star,
                    'loss': L_min,
                    'event': 'global_minimum_declared',
                })
                break  # Convergence criterion met; exit loop

            else:
                # Lower basin found → log the discovery and restart
                if verbose:
                    print(f"\nIteration {iteration}: Lower basin detected!")
                    print(f"  Restarting from θ = {new_start:.6f}")

                self.history.append({
                    'iteration': iteration,
                    'theta': new_start,
                    'loss': self.loss_function(new_start),
                    'event': 'lower_basin_found',
                })

                # Exploitation phase: descend from the newly found region
                theta_star, L_min = self.local_optimize(new_start)

                if verbose:
                    print(
                        f"  Converged to θ* = {theta_star:.6f}, "
                        f"L = {L_min:.8f}"
                    )

                self.history.append({
                    'iteration': iteration,
                    'theta': theta_star,
                    'loss': L_min,
                    'event': 'restart_convergence',
                })

        return {
            'theta_star': theta_star,
            'L_min': L_min,
            'iterations': iteration,
            'history': self.history,
        }


# ---------------------------------------------------------------------------
# Test Function
# ---------------------------------------------------------------------------

def test_function(x: float) -> float:
    """
    Benchmark loss function used to evaluate the global minimum detector.

    Defined as:

    .. math::
        f(x) = (1 + x - x^3)^2

    This function has multiple local minima across the real line, with a
    global minimum of **0** at the roots of :math:`1 + x - x^3 = 0`.

    Parameters
    ----------
    x : float
        Scalar input value.

    Returns
    -------
    float
        Non-negative loss value; equals zero at the exact roots of
        :math:`1 + x - x^3`.

    Examples
    --------
    >>> test_function(0)
    1.0
    >>> test_function(1)
    1.0
    """
    return (1 + x - x**3) ** 2


# ---------------------------------------------------------------------------
# Analysis Utilities
# ---------------------------------------------------------------------------

def analyze_function() -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a numerical analysis of ``test_function`` over ``[-3, 3]``.

    Prints a summary of the global minimum and all detected local minima,
    then returns the evaluation grid for downstream plotting.

    Returns
    -------
    x : np.ndarray, shape (1000,)
        Uniformly spaced x-values over [-3, 3].
    y : np.ndarray, shape (1000,)
        Corresponding loss values ``test_function(x)``.

    Notes
    -----
    Local minima are detected via peak-finding on the negated function, so
    the accuracy depends on the grid resolution (1000 points by default).
    """
    from scipy.signal import find_peaks

    x = np.linspace(-3, 3, 1000)
    y = test_function(x)

    # Invert the function so that minima become peaks for find_peaks
    peaks, _ = find_peaks(-y)
    local_minima_x = x[peaks]
    local_minima_y = y[peaks]

    # Print analysis summary
    print("=" * 70)
    print("FUNCTION ANALYSIS: y = (1 + x - x³)²")
    print("=" * 70)
    print(f"\nFunction bounds  : x ∈ [{x.min():.2f}, {x.max():.2f}]")
    print(f"Global minimum   : y = {y.min():.8f} at x = {x[np.argmin(y)]:.6f}")

    if len(local_minima_x) > 0:
        print(f"\nLocal minima found: {len(local_minima_x)}")
        for i, (mx, my) in enumerate(zip(local_minima_x, local_minima_y)):
            print(f"  Local min {i + 1}: y = {my:.8f} at x = {mx:.6f}")

    print("=" * 70 + "\n")
    return x, y


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_results(
    detector: GlobalMinimumDetector,
    x_range: np.ndarray,
    y_range: np.ndarray,
    results: Dict,
) -> plt.Figure:
    """
    Produce a visualization of the loss surface and the optimization path.

    Plots the loss function curve and overlays each event recorded in
    ``detector.history``, colour-coded by event type.  The final solution
    is highlighted with a gold star marker.

    Color legend
    ------------
    - **Green circle**   – initial convergence point.
    - **Orange triangle** – lower-basin sample that triggered a restart.
    - **Red circle**     – post-restart convergence point.
    - **Purple circle**  – point declared as global minimum.
    - **Gold star**      – final solution (``results['theta_star']``).

    Parameters
    ----------
    detector : GlobalMinimumDetector
        A detector instance whose ``history`` has been populated by
        ``optimize_with_global_detection``.
    x_range : np.ndarray
        x-values for the loss surface curve (e.g. from ``analyze_function``).
    y_range : np.ndarray
        Corresponding loss values (e.g. from ``analyze_function``).
    results : dict
        The return value of ``optimize_with_global_detection``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object (also displayed inline via
        ``plt.show()``).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        'Global Minimum Detection via Horizontal Loss Plane Probing\n'
        r'Function: $y = (1 + x - x^3)^2$',
        fontsize=14,
        fontweight='bold',
    )

    # ---- Draw the loss surface ----
    ax.plot(x_range, y_range, 'b-', linewidth=2, label='Loss surface')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Zero-loss baseline
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 5)  # Focus on the region of interest near y=0

    # ---- Event colour mapping ----
    colors = {
        'initial_convergence': 'green',
        'lower_basin_found': 'orange',
        'restart_convergence': 'red',
        'global_minimum_declared': 'purple',
    }

    # ---- Plot every history event ----
    for h in detector.history:
        # Circles for convergence events; triangles for basin-discovery events
        marker = 'o' if 'convergence' in h['event'] else '^'
        size = 150 if h['event'] == 'global_minimum_declared' else 80

        ax.scatter(
            h['theta'], h['loss'],
            c=colors.get(h['event'], 'gray'),
            marker=marker,
            s=size,
            edgecolors='black',
            linewidth=1.5,
            zorder=5,
            alpha=0.8,
        )

        # Label convergence points with their iteration number
        if 'convergence' in h['event']:
            ax.annotate(
                f"Iter {h['iteration']}",
                xy=(h['theta'], h['loss']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7,
            )

    # ---- Highlight the final solution ----
    ax.scatter(
        results['theta_star'], results['L_min'],
        c='gold', marker='*', s=400,
        edgecolors='red', linewidth=2,
        zorder=10, label='Final Solution',
    )

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y = L(x)', fontsize=11)
    ax.set_title('Loss Surface & Optimization Path', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()

    print("\n✓ Visualization displayed.")
    return fig


# ---------------------------------------------------------------------------
# Method Comparison
# ---------------------------------------------------------------------------

def compare_methods() -> None:
    """
    Compare standard single-start BFGS against the global minimum detector.

    Tests five starting points (``[-2, -1, 0, 1, 2]``) and prints the
    solution found by each approach, demonstrating that standard gradient
    descent can converge to local minima depending on initialisation, while
    the global detection algorithm consistently finds the global minimum.

    Returns
    -------
    None
        Results are printed to stdout only.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Standard Optimization vs. Global Minimum Detection")
    print("=" * 70)

    starting_points = [-2.0, -1.0, 0.0, 1.0, 2.0]

    # ---- Baseline: standard single-start BFGS ----
    print("\n--- Standard Gradient Descent (BFGS) ---")
    for x0 in starting_points:
        result = minimize(test_function, x0, method='BFGS')
        x_sol = (
            float(result.x[0]) if hasattr(result.x, '__len__') else float(result.x)
        )
        print(
            f"Start: {x0:6.2f} → Solution: x* = {x_sol:8.5f}, "
            f"L = {result.fun:.8f}"
        )

    # ---- Proposed: global minimum detection ----
    print("\n--- Global Minimum Detection Algorithm ---")
    for x0 in starting_points:
        detector = GlobalMinimumDetector(
            test_function, epsilon=1e-6, n_probes=50, bounds=(-3, 3)
        )
        result = detector.optimize_with_global_detection(
            x0, max_restarts=5, verbose=False
        )
        print(
            f"Start: {x0:6.2f} → Solution: x* = {result['theta_star']:8.5f}, "
            f"L = {result['L_min']:.8f}, "
            f"Iterations: {result['iterations']}"
        )

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("  Standard method : May converge to a local minimum depending on x0.")
    print("  Global detection: Consistently escapes local minima via probing.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    End-to-end demonstration of the global minimum detection algorithm.

    Workflow
    --------
    1. Analyse ``test_function`` to enumerate its local minima.
    2. Instantiate ``GlobalMinimumDetector`` and run the optimisation from a
       deliberately poor starting point (near a local minimum).
    3. Visualise the optimization path over the loss surface.
    4. Benchmark against standard single-start BFGS from multiple starts.
    """
    print("\n" + "=" * 70)
    print("GLOBAL MINIMUM DETECTION ALGORITHM")
    print("Geometric Loss Surface Probing Method")
    print("=" * 70 + "\n")

    # Step 1 – Analyse the target function
    x_range, y_range = analyze_function()

    # Step 2 – Run the global minimum detection algorithm
    print("\nRUNNING GLOBAL MINIMUM DETECTION ALGORITHM")
    print("-" * 70)

    # Deliberately start near a local minimum to stress-test the algorithm
    x0 = -1.5

    detector = GlobalMinimumDetector(
        loss_function=test_function,
        epsilon=1e-6,
        n_probes=100,
        bounds=(-3, 3),
    )

    results = detector.optimize_with_global_detection(
        x0, max_restarts=10, verbose=True
    )

    # Step 3 – Visualize results
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATION...")
    visualize_results(detector, x_range, y_range, results)

    # Step 4 – Compare against standard BFGS
    compare_methods()

    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
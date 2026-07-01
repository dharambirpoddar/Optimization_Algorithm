"""
Tunneling-Based Global Minimum Finder
======================================

PSEUDOCODE
----------
FUNCTION tunneling_global_minimum(f, x0):
    f_prime = derivative(f)
    visited = []                      # critical points already tested
    current_start = x0

    LOOP:
        # Step 2-3: local minimum via gradient descent
        x1 = gradient_descent(f_prime, start = current_start)
        y1 = f(x1)

        IF x1 already in visited (basin escape failed):
            current_start = current_start + push_further(direction)
            CONTINUE

        visited.append(x1)

        # Step 4-6: horizontal tangent test
        roots = solve( f(x) = y1 )                 # all real roots
        other_roots = roots EXCLUDING x1

        # Step 7: interpret
        IF other_roots is EMPTY:
            RETURN x1, y1                # Case A: global minimum found
        ELSE:
            # Step 8: restart from NEAREST crossing, not farthest
            x_next = root in other_roots CLOSEST to x1
            current_start = x_next + tiny_step_beyond(x_next)
            CONTINUE   # go back to Step 2

ALGORITHM STEPS (see docstring in each function below for detail)
-------------------------------------------------------------------
1. Define the curve f(x) (n-th order polynomial).
2. Gradient descent on f'(x)=0 -> local critical point x1.
3. y1 = f(x1).
4. Horizontal tangent line: y = y1.
5. Set up f(x) = y1.
6. Solve for ALL real roots of f(x) - y1 = 0.
7. Case A: no other real root -> x1 is the GLOBAL MINIMUM, stop.
   Case B: other real root(s) exist -> x1 is only a LOCAL minimum.
8. Restart gradient descent from the root NEAREST to x1 (not the farthest)
   and repeat from Step 2.
"""

import sympy as sp

x = sp.symbols('x', real=True)


# ---------------------------------------------------------------------
# Step 2: Gradient descent to solve f'(x) = 0
# ---------------------------------------------------------------------
def gradient_descent(fprime_func, x0, lr=0.01, tol=1e-10, max_iter=200_000,
                      max_step=0.05):
    """
    Gradient descent that walks 'downhill' on f until f'(x) ~ 0.

    For steep polynomials f'(x) can be huge far from the root, so a plain
    x_new = x - lr*grad update can overshoot straight past the intended
    basin into a completely different one. The per-step move is clipped to
    `max_step` so the walk always respects local basin boundaries, while
    still taking tiny, precise steps once it's near a critical point.
    """
    xi = x0
    for _ in range(max_iter):
        grad = fprime_func(xi)
        step = lr * grad
        step = max(-max_step, min(max_step, step))  # clip to avoid overshoot
        xi_new = xi - step
        if abs(xi_new - xi) < tol:
            return xi_new
        xi = xi_new
    return xi


# ---------------------------------------------------------------------
# Steps 5 & 6: Solve f(x) = y1 for real roots
# ---------------------------------------------------------------------
def real_roots_of(f_expr, y_val, x_sym):
    """Return sorted real roots of f(x) - y_val = 0 (f_expr must be polynomial)."""
    diff_expr = sp.expand(f_expr - y_val)
    poly = sp.Poly(diff_expr, x_sym)
    roots = poly.nroots(n=15)
    real_roots = sorted(
        {round(float(sp.re(r)), 6) for r in roots if abs(sp.im(r)) < 1e-6}
    )
    return real_roots


# ---------------------------------------------------------------------
# Full algorithm (Steps 1-8), restart uses NEAREST crossing (corrected)
# ---------------------------------------------------------------------
def tunneling_global_minimum(f_expr, x_sym=x, x0=-5.0, lr=0.01,
                              push_past=0.005, max_rounds=50, verbose=True):
    """
    push_past : how far beyond the chosen crossing point to place the next
                gradient-descent start, so it lands in the neighboring
                basin rather than exactly on the root itself.
    """
    f_prime_expr = sp.diff(f_expr, x_sym)
    f_lamb = sp.lambdify(x_sym, f_expr, 'numpy')
    fprime_lamb = sp.lambdify(x_sym, f_prime_expr, 'numpy')

    current_start = x0
    history = []
    visited_x1 = []

    for round_no in range(1, max_rounds + 1):
        # --- Step 2 & 3 ---
        x1 = gradient_descent(fprime_lamb, current_start, lr=lr)
        y1 = float(f_lamb(x1))

        # Guard: landed back on an already-visited critical point ->
        # push the start further out and retry.
        if any(abs(x1 - v) < 1e-3 for v in visited_x1):
            if verbose:
                print(f"\n--- Round {round_no}: re-landed on visited point "
                      f"x = {x1:.6f}; pushing start further out ---")
            direction = 1 if current_start >= x1 else -1
            current_start += direction * push_past * 10
            continue

        visited_x1.append(x1)
        history.append((x1, y1))

        if verbose:
            print(f"\n--- Round {round_no} ---")
            print(f"Start point : x0 = {current_start:.4f}")
            print(f"Critical pt : x1 = {x1:.6f}   y1 = f(x1) = {y1:.6f}")

        # --- Step 4-6: horizontal tangent, solve f(x) = y1 ---
        roots = real_roots_of(f_expr, y1, x_sym)
        other_roots = [r for r in roots if abs(r - x1) > 1e-3]

        if verbose:
            print(f"Real roots of f(x)=y1 : {roots}")
            print(f"Other roots (!= x1)   : {other_roots}")

        # --- Step 7, Case A: global minimum confirmed ---
        if not other_roots:
            if verbose:
                print(f"\n>>> No further real intersection found.")
                print(f">>> GLOBAL MINIMUM: x = {x1:.6f}, f(x) = {y1:.6f}")
            return x1, y1, history

        # --- Step 7, Case B + Step 8: restart from NEAREST root ---
        nearest = min(other_roots, key=lambda r: abs(r - x1))
        direction = 1 if nearest > x1 else -1
        current_start = nearest + direction * push_past

        if verbose:
            print(f">>> x1 is NOT global minimum "
                  f"(curve returns to y1 at x = {nearest:.6f})")
            print(f">>> Restarting gradient descent from x0 = {current_start:.4f}")

    print("Max rounds reached without full confirmation.")
    return history[-1][0], history[-1][1], history


# ---------------------------------------------------------------------
# User-facing entry point: takes f(x) as a string
# ---------------------------------------------------------------------
def find_global_minimum(f_str, x0=None, verbose=True):
    """
    f_str : the curve as a string, e.g. "(1 + x - x**3)**2"
    x0    : optional starting point for gradient descent; if None, a
            reasonable default is chosen automatically.
    """
    f_expr = sp.sympify(f_str, locals={'x': x})

    if not f_expr.is_polynomial(x):
        raise ValueError("This implementation requires f(x) to be a polynomial.")

    if x0 is None:
        x0 = -10.0  # generic default; widen search if needed

    if verbose:
        print("=" * 60)
        print("Tunneling Global Minimum Algorithm")
        print(f"f(x) = {sp.expand(f_expr)}")
        print("=" * 60)

    x_star, y_star, history = tunneling_global_minimum(
        f_expr, x, x0=x0, verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 60)
        print(f"FINAL ANSWER: global minimum at x = {x_star:.6f}, "
              f"f(x) = {y_star:.6f}")
        print("=" * 60)

    return x_star, y_star, history


# ---------------------------------------------------------------------
# Run: takes f(x) as user input
# ---------------------------------------------------------------------
if __name__ == "__main__":
    default_f = "(1 + x - x**3)**2"
    user_f = input(f"Enter f(x) in terms of x [default: {default_f}]: ").strip()
    if not user_f:
        user_f = default_f

    find_global_minimum(user_f, x0=3.0)
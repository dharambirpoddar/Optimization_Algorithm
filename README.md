# Optimization_Algorithm
Finding the global minimum of non-convex functions is a fundamental challenge in math-
ematical optimization, often hindered by the presence of multiple local minima. We present
a deterministic tunneling algorithm specifically designed for single-variable polynomial func-
tions. The method iteratively alternates between gradient descent to locate local minima
and exact polynomial root-finding to deterministically ”tunnel” through energy barriers.
By solving for the real roots of the horizontal tangent line intersecting the current local
minimum, the algorithm identifies adjacent basins of attraction. Crucially, by selecting the
nearest real root as the starting point for the subsequent descent, the algorithm system-
atically maps out the global landscape without getting trapped in bounded local basins or
overshooting the domain. We provide the theoretical framework, algorithmic pseudocode,
and discuss its implementation limits

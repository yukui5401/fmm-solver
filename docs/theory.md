# Fast Multipole Method: theoretical background

## 1. The problem

Given $N$ point sources in 3D with charges (or masses) $q_j$, compute the potential at every source due to all others:

$$
\phi(x_i) = \sum_{j \neq i} \frac{q_j}{|x_i - x_j|}
$$

Direct evaluation is $O(N^2)$. FMM computes the same quantity (to within a controllable error tolerance) in $O(N)$ by exploiting the fact that a cluster of distant sources can be approximated by a compact multipole expansion, rather than summed one-by-one.

## 2. Direct summation (ground truth)

`src/direct_sum.cpp` implements the $O(N^2)$ sum exactly as written above. This is not an approximation (it's the definition) so it's the baseline every later stage of FMM is validated against. Every optimization from here on trades exactness for speed within a chosen error tolerance, and direct summation is how that tolerance gets measured.

## 3. Adaptive octree

`src/octree.cpp` recursively partitions the bounding cube of all source points into 8 octants, refining any octant that holds more than
`max_particles_per_leaf` points, until every leaf is under that cap (or a maximum depth safety valve is hit - relevant mainly for degenerate inputs like duplicate points). Unlike a uniform grid, this means dense clusters get finer subdivision than sparse regions, which is what makes FMM effective on realistic (non-uniform) particle distributions.

Each node owns a cubic bounding box; a point's octant relative to a node's center is determined by 3 independent comparisons (one per axis), giving values 0-7.

## 4. Spherical harmonics

FMM's multipole and local expansions are built from spherical harmonics $Y_n^m(\theta, \phi)$, fully normalized so that they're orthonormal on the unit sphere:

$$
Y_n^m(\theta, \phi) = \sqrt{\frac{2n+1}{4\pi} \cdot \frac{(n-m)!}{(n+m)!}} \; P_n^m(\cos\theta) \; e^{im\phi}
$$

for $-n \le m \le n$, where $P_n^m$ is the associated Legendre function.

**Associated Legendre polynomials** (`associated_legendre` in `spherical_harmonics.cpp`) are computed via the standard stable upward recurrence rather than a closed-form formula (which would be numerically unstable for larger $n$):

1. Seed the diagonal: $P_m^m = (-1)^m (2m-1)!! \, (1-x^2)^{m/2}$
2. Step to the next degree: $P_{m+1}^m = x(2m+1) P_m^m$
3. Climb further via the three-term recurrence:
   $(n-m) P_n^m = x(2n-1) P_{n-1}^m - (n+m-1) P_{n-2}^m$

**Negative orders** are recovered via the identity $Y_n^{-m} = (-1)^m \overline{Y_n^m}$ rather than computed from scratch, since $P_n^m$ is only computed for $m \ge 0$.

**Validation approach**: beyond checking known closed-form values (e.g. $Y_0^0 = 1/\sqrt{4\pi}$ everywhere, $Y_1^0 \propto \cos\theta$), the test suite verifies orthonormality numerically via quadrature over the sphere - $\int Y_{n_1}^{m_1} \overline{Y_{n_2}^{m_2}} \, d\Omega$ should be 1 when indices match and 0 otherwise. This is a stronger check than closed-form spot values alone, since a sign or normalization error in the recurrence would break orthogonality even if a few individual values happened to look plausible.

## 5. Multipole expansion (P2M) and evaluation

`src/multipole.cpp` implements the multipole expansion in the convention of Greengard & Rokhlin, _"A New Version of the Fast Multipole Method for the Laplace Equation in Three Dimensions"_ (Yale RR-1115, 1996), whose Theorem 3.2 gives:

$$
\Phi(P) = \sum_{n=0}^{p} \sum_{m=-n}^{n} \frac{M_n^m}{r^{n+1}} Y_n^m(\theta,\phi), \qquad
M_n^m = \sum_i q_i \, \rho_i^n \, Y_n^{-m}(\alpha_i, \beta_i)
$$

where $(r,\theta,\phi)$ are the target's coordinates relative to the expansion center and $(\rho_i,\alpha_i,\beta_i)$ each source's. Truncation error is bounded by their eq. (39) and shrinks with increasing order $p$ and increasing separation.

**Convention warning (this matters):** their $Y_n^m$ (eq. 34) is

$$
Y_n^m(\theta,\phi) = \sqrt{\frac{(n-|m|)!}{(n+|m|)!}} \, P_n^{|m|}(\cos\theta) \, e^{im\phi}
$$

— _without_ the $\sqrt{(2n+1)/4\pi}$ orthonormalization factor, and with the phase $e^{im\phi}$ taken with signed $m$ (so negative orders also differ from the orthonormal convention by $(-1)^m$). The translation theorems below are stated in this basis and are wrong by exactly those factors if evaluated with orthonormal harmonics. The codebase therefore has both: `spherical_harmonic` (orthonormal, used for standalone validation via quadrature-verified orthonormality) and `spherical_harmonic_gr` (GR convention, used exclusively by the multipole/translation pipeline).

## 6. M2M: exact translation of a multipole expansion

The upward pass builds each parent's expansion by _translating_ its children's coefficients rather than reprocessing particles — GR Theorem
5.1 (eq. 45). With $(\rho, \alpha, \beta)$ the old center's spherical coordinates relative to the new center, and $A_n^m = (-1)^n / \sqrt{(n-m)!\,(n+m)!}$ (their eq. 33):

$$
M_j^k = \sum_{n=0}^{j} \sum_{\substack{m=-n \\ |k-m| \le j-n}}^{n}
\frac{O_{j-n}^{k-m} \; i^{|k|-|m|-|k-m|} \; A_n^m \, A_{j-n}^{k-m} \; \rho^n \, Y_n^{-m}(\alpha,\beta)}{A_j^k}
$$

Two properties worth knowing, both verified in `tests/test_multipole.cpp`:

- **The shift is exact for truncated expansions**: unlike M2L, no additional truncation error is introduced by moving the center. The test suite checks this at the coefficient level: translated coefficients match a direct P2M about the new center to ~1e-10, not merely at the level of evaluated potentials.
- **Exactness survives composition**: chaining shifts (child -> intermediate -> final) still reproduces the direct expansion, which is what the multi-level tree merge relies on.

## 7. Local expansions: M2L, L2L, L2P

`src/local.cpp` implements the remaining translation operators, all in the same GR convention:

- **M2L** (GR Theorem 5.2, eq. 48): converts a multipole expansion into a local expansion about a well-separated center. Validity requires their separation condition (sources within radius $a$ of the multipole center, centers separated by $\rho > (c+1)a$, $c > 1$). Unlike M2M and L2L, this operator introduces genuine truncation error (their eq. 49, worst case $\sim (1/c)^{p+1}$); the tests verify the error shrinks with increasing order (from ~6e-7 at $p{=}4$ to ~1e-12 at $p{=}10$ for the test geometry).
- **L2L** (GR Theorem 5.3, eq. 52): shifts a local expansion's center. A finite Maclaurin shift, exact for truncated expansions - verified by checking that shifted and unshifted expansions evaluate identically at the same physical points to machine precision.
- **L2P** (GR eq. 47): evaluates the accumulated local expansion at a point, $\Phi = \sum_{j,k} L_j^k Y_j^k(\theta,\phi) r^j$.

An integration test exercises the complete operator chain the way the full traversal will - two sub-clusters P2M'd at leaf centers, M2M-merged to a parent, M2L'd across a well-separated gap, L2L'd down to a child center, and L2P-evaluated - agreeing with direct summation to relative 1e-7 at order 12.

## 8. Complexity

- Direct summation: $O(N^2)$.
- Octree construction: $O(N \log N)$ typically.
- Upward pass (P2M + M2M): $O(N p^2)$ for leaf P2M plus $O(p^4)$ per M2M translation over $O(N/s)$ boxes - matching GR's Step 1-2 operation counts.
- Full FMM (once M2L/L2L/L2P/P2P are complete): $O(N)$ for fixed accuracy, the entire point of the method.

"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# A = [[3, 4], [2, 1]],  B = [[1, 2], [0, -1]].
#
# AB:   row1: [3*1 + 4*0, 3*2 + 4*(-1)] = [3, 2]
#       row2: [2*1 + 1*0, 2*2 + 1*(-1)] = [2, 3]          => AB = [[3, 2], [2, 3]]
# BA:   row1: [1*3 + 2*2, 1*4 + 2*1]    = [7, 6]
#       row2: [0*3 + (-1)*2, 0*4 + (-1)*1] = [-2, -1]     => BA = [[7, 6], [-2, -1]]
#       (AB != BA: matrix multiplication is not commutative.)
# (A+B)^T: A+B = [[4, 6], [2, 0]]                          => (A+B)^T = [[4, 2], [6, 0]]
# AB^T: B^T = [[1, 0], [2, -1]]
#       row1: [3*1 + 4*2, 3*0 + 4*(-1)] = [11, -4]
#       row2: [2*1 + 1*2, 2*0 + 1*(-1)] = [4, -1]         => AB^T = [[11, -4], [4, -1]]
# trace(A)  = 3 + 1 = 4
# trace(AB) = 3 + 3 = 6   (and trace(BA) = 7 + (-1) = 6, consistent with P1 Q3)
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.array([[3., 4.], [2., 1.]] if A is None else A, dtype=float)
    B = np.array([[1., 2.], [0., -1.]] if B is None else B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, float(np.trace(A)), float(np.trace(AB))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# M = [[2, -1], [-1, 2]].
# Characteristic polynomial: det(M - lambda*I) = (2 - lambda)^2 - (-1)(-1)
#                                              = lambda^2 - 4*lambda + 3
#                                              = (lambda - 1)(lambda - 3) = 0
#   => lambda_1 = 1, lambda_2 = 3.
# lambda = 1: (M - I)v = [[1, -1], [-1, 1]]v = 0  =>  v1 = v2  =>  v = [1, 1]^T
#   Check: Mv = [2*1 - 1*1, -1*1 + 2*1] = [1, 1] = 1 * [1, 1] (verified)
# lambda = 3: (M - 3I)v = [[-1, -1], [-1, -1]]v = 0  =>  v1 = -v2  =>  v = [1, -1]^T
#   Check: Mv = [2*1 - 1*(-1), -1*1 + 2*(-1)] = [3, -3] = 3 * [1, -1] (verified)
# Normalized: v = [1, 1]/sqrt(2) and [1, -1]/sqrt(2). They are orthogonal, as
# expected for a symmetric matrix (spectral theorem).
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.array([[2., -1.], [-1., 2.]] if M is None else M, dtype=float)
    # eigh is the symmetric solver: real eigenvalues (ascending) and an
    # orthonormal set of eigenvector columns, so repeated roots get a full basis.
    values, vectors = np.linalg.eigh(M)
    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# Let A be m x n and B be n x m, so AB is m x m and BA is n x n.
# By the definition of matrix product, (AB)_ii = sum_{k=1..n} A_ik B_ki.
# trace(AB) = sum_{i=1..m} (AB)_ii = sum_{i=1..m} sum_{k=1..n} A_ik B_ki.
# Both sums are finite, so they can be swapped, and scalar multiplication commutes:
#           = sum_{k=1..n} sum_{i=1..m} B_ki A_ik
#           = sum_{k=1..n} (BA)_kk          since (BA)_kk = sum_{i=1..m} B_ki A_ik
#           = trace(BA).                                                        QED
# (Example from Q1: trace(AB) = 6 = trace(BA), even though AB != BA.)

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# C = [[4, 7], [2, 6]].
# Step 1 (existence): det(C) = 4*6 - 7*2 = 24 - 14 = 10 != 0, so C is invertible.
# Step 2 (2x2 formula): for [[a, b], [c, d]], inverse = (1/det) * [[d, -b], [-c, a]].
#   C^{-1} = (1/10) * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]].
# Step 3 (check): C C^{-1} = (1/10)[[4*6 + 7*(-2), 4*(-7) + 7*4],
#                                   [2*6 + 6*(-2), 2*(-7) + 6*4]]
#                          = (1/10)[[10, 0], [0, 10]] = I (verified)
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.array([[4., 7.], [2., 6.]] if C is None else C, dtype=float)
    # Solve C X = I rather than forming the adjugate; works for any size n.
    return np.linalg.solve(C, np.eye(C.shape[0]))


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# (a) Probabilities must sum to 1:  1 = c * sum_{k>=0} 3^k/k!  = c * e^3
#     (Taylor series e^x = sum_k x^k/k!).   =>  c = e^{-3} ~ 0.049787.
#     So X ~ Poisson(3).
# (b) E[X] = sum_{k>=0} k e^{-3} 3^k/k! = e^{-3} sum_{k>=1} 3^k/(k-1)!
#          = 3 e^{-3} sum_{j>=0} 3^j/j!  (j = k-1)  = 3 e^{-3} e^3 = 3.
#     E[X(X-1)] = e^{-3} sum_{k>=2} 3^k/(k-2)! = 9 e^{-3} sum_{j>=0} 3^j/j! = 9.
#     E[X^2] = E[X(X-1)] + E[X] = 9 + 3 = 12.
#     Var(X) = E[X^2] - E[X]^2 = 12 - 9 = 3.
#     In general for rate r: c = e^{-r}, E[X] = r, Var(X) = r.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    rate = float(rate)
    return float(np.exp(-rate)), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# (a) Definition of conditional probability: P(A|B) = P(A and B)/P(B)
#     => P(A and B) = P(A|B) P(B) = 0.6 * 0.3 = 0.18.
# (b) P(B|A) = P(A and B)/P(A) = 0.18 / 0.5 = 0.36.   (Bayes' rule)
# (c) Inclusion-exclusion: P(A or B) = P(A) + P(B) - P(A and B) = 0.5 + 0.3 - 0.18 = 0.62.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    p_ab = p_a_given_b * p_b
    return p_ab, p_ab / p_a, p_a + p_b - p_ab


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# Data: h = 3 heads, t = 2 tails, n = 5. Flips are i.i.d. Bernoulli(theta).
# (a) Likelihood: L(theta) = prod_i P(x_i|theta) = theta^3 (1-theta)^2.
#     Log-likelihood: l(theta) = 3 ln(theta) + 2 ln(1-theta).
#     dl/dtheta = 3/theta - 2/(1-theta) = 0  =>  3(1-theta) = 2 theta  =>  theta = 3/5.
#     d2l/dtheta2 = -3/theta^2 - 2/(1-theta)^2 < 0, so it is a maximum.
#     theta_MLE = 3/5 = 0.6.   (General: h/n; h=0 -> 0, h=n -> 1 at the boundary.)
# (b) Posterior propto likelihood * prior = theta^3 (1-theta)^2 * 2 theta propto theta^4 (1-theta)^2
#     on [0,1] (prior is zero elsewhere).
#     Log-posterior: 4 ln(theta) + 2 ln(1-theta) + const.
#     Derivative: 4/theta - 2/(1-theta) = 0  =>  4(1-theta) = 2 theta  =>  theta = 4/6.
#     Second derivative -4/theta^2 - 2/(1-theta)^2 < 0, so it is a maximum.
#     theta_MAP = 2/3 ~ 0.6667.
#     The prior 2*theta acts like one extra "pseudo-head" (it is Beta(2,1)), pulling
#     the estimate up from 0.6. General: posterior propto theta^{h+1}(1-theta)^{n-h}, mode
#     (h+1)/(n+1). When h = n the posterior is theta^{n+1}, increasing, so the
#     maximum is on the boundary theta = 1 = (n+1)/(n+1); the formula still holds.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# For i.i.d. x_1..x_N ~ N(mu, sigma^2):
#   l(mu, sigma^2) = -N/2 ln(2 pi) - N/2 ln(sigma^2) - (1/(2 sigma^2)) sum_i (x_i - mu)^2.
# dl/dmu = (1/sigma^2) sum_i (x_i - mu) = 0  =>  mu_MLE = (1/N) sum_i x_i.
# dl/d(sigma^2) = -N/(2 sigma^2) + (1/(2 sigma^4)) sum_i (x_i - mu)^2 = 0
#   =>  sigma^2_MLE = (1/N) sum_i (x_i - mu_MLE)^2   (divisor N, not N-1).
# With X = {1, 3, 5, 7}, N = 4:
#   mu_MLE = (1 + 3 + 5 + 7)/4 = 16/4 = 4.
#   deviations: -3, -1, 1, 3; squares: 9, 1, 1, 9; sum = 20.
#   sigma^2_MLE = 20/4 = 5.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    x = np.array([1., 3., 5., 7.] if data is None else data, dtype=float)
    mu = x.mean()
    return float(mu), float(np.mean((x - mu) ** 2))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# Likelihood x | mu ~ N(mu, sigma^2 = 4); prior mu ~ N(mu0 = 0, tau^2 = 1).
# log p(mu | x) = -(x - mu)^2/(2 sigma^2) - (mu - mu0)^2/(2 tau^2) + const.
# d/dmu: (x - mu)/sigma^2 - (mu - mu0)/tau^2 = 0
#   =>  mu (1/sigma^2 + 1/tau^2) = x/sigma^2 + mu0/tau^2
#   =>  mu_MAP = (tau^2 x + sigma^2 mu0) / (sigma^2 + tau^2).
# Second derivative -(1/sigma^2 + 1/tau^2) < 0, so it is a maximum.
# Plug in: mu_MAP = (1*5 + 4*0)/(4 + 1) = 5/5 = 1.
# The prior (precision 1) is 4x more confident than the observation (precision 1/4),
# so the estimate is pulled strongly from x = 5 toward the prior mean 0.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    return ((prior_variance * x + observation_variance * prior_mean)
            / (observation_variance + prior_variance))


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# Let mu = E[X] (a constant). By definition Var(X) = E[(X - mu)^2].
# Expand the square: (X - mu)^2 = X^2 - 2 mu X + mu^2.
# Linearity of expectation (constants pull out, E[c] = c):
#   Var(X) = E[X^2] - 2 mu E[X] + mu^2 = E[X^2] - 2 mu^2 + mu^2 = E[X^2] - mu^2
#          = E[X^2] - (E[X])^2.                                                QED

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# Let X be a random d-vector with mean mu = E[X]. Its covariance matrix is
#   Sigma = E[(X - mu)(X - mu)^T],  i.e. Sigma_jk = Cov(X_j, X_k)
# (symmetric and positive semidefinite; diagonal entries are the variances).
# Let u be a fixed unit vector (u^T u = 1) and Y = u^T X, a scalar.
# 1. Mean: E[Y] = E[u^T X] = u^T E[X] = u^T mu   (linearity).
# 2. Centered projection: Y - E[Y] = u^T X - u^T mu = u^T (X - mu).
# 3. Variance: Var(Y) = E[(Y - E[Y])^2] = E[(u^T (X - mu))^2].
# 4. A scalar equals its transpose, so (u^T (X-mu))^2 = u^T (X-mu) (X-mu)^T u.
# 5. u is constant, so it pulls out of the expectation:
#    Var(Y) = u^T E[(X - mu)(X - mu)^T] u = u^T Sigma u.                        QED
# For data, replace E with the sample average: Sigma_hat = Xc^T Xc / (n-1), where
# Xc is the centered data matrix. PCA chooses u to maximize u^T Sigma u subject
# to u^T u = 1. The Lagrangian u^T Sigma u - lambda(u^T u - 1) gives Sigma u = lambda u,
# so the best u is the top eigenvector, and the variance along it is lambda.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# generate_data draws n samples from a 2D multivariate normal with mean [2, -1] and
# covariance [[3, 1.2], [1.2, 1]] using numpy.random.default_rng(seed), so results
# are reproducible. The true correlation is 1.2/sqrt(3*1) ~ 0.693, a strong positive
# correlation. With n=500, seed=0 the sample correlation is ~ 0.70. In the scatter
# plot the points form a tilted ellipse: x1 has the larger spread (var 3 vs 1), and
# larger x1 values come with larger x2 values. The major axis is tilted ~25 degrees up
# from the x1-axis rather than lying along either coordinate axis. That means the
# features carry redundant information, which PCA exploits.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.array([2., -1.] if mean is None else mean, dtype=float)
    covariance = np.array([[3., 1.2], [1.2, 1.]] if covariance is None else covariance,
                          dtype=float)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# Step 1 (center): Xc = X - mean. Q1 is stated in terms of X - mu, so the variance of
#   a projection is measured around the mean.
# Step 2 (covariance): Sigma_hat = Xc^T Xc / (n-1), the sample version of
#   E[(X-mu)(X-mu)^T] from Q1 (unbiased n-1 divisor).
# Step 3 (eigendecomposition): from Q1, maximizing u^T Sigma u subject to |u|=1 gives
#   Sigma u = lambda u. np.linalg.eigh returns real eigenvalues and orthonormal
#   eigenvectors for the symmetric Sigma_hat.
# Step 4 (select): sort eigenvalues in descending order. Column k of `components` is
#   the k-th principal direction, and the variance of the data along it is lambda_k
#   (u_k^T Sigma u_k = lambda_k). The scores Xc @ components are the coordinates of
#   each point in the PC basis.
# For the default data (n=500, seed=0) the eigenvalues are ~ 3.46 and 0.41. PC1 ~
#   +/-[0.91, 0.42] (about 25 degrees above the x1-axis) and PC2 is perpendicular to it.
#   In the plot, each PC is drawn as an axis through the mean, extending
#   2*sqrt(lambda) (two standard deviations) on each side. PC1 runs along the long axis of the point cloud and PC2
#   along the short axis.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.array(X, dtype=float)  # copy, so the caller's array is never touched
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (X.shape[0] - 1)
    covariance = (covariance + covariance.T) / 2  # enforce exact symmetry
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    # Covariance is PSD; clip tiny negative round-off in rank-deficient cases.
    eigenvalues = np.clip(eigenvalues[order], 0., None)
    components = eigenvectors[:, order]
    return {
        'mean': mean,
        'centered': centered,
        'covariance': covariance,
        'eigenvalues': eigenvalues,
        'components': components,
        'scores': centered @ components,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# Explained-variance fraction of PC k = lambda_k / sum_j lambda_j. The sum of the
# eigenvalues equals trace(Sigma), the total variance, so these fractions add to 1.
# Default data: PC1 ~ 0.89 (89%), PC2 ~ 0.11 (11%).
# Interpretation: because the two features are strongly correlated, one direction
# captures nearly 9/10 of the total variability. Projecting onto PC1 alone
# (2D -> 1D) keeps ~89% of the variance and discards ~11%, which is the squared
# reconstruction error, mostly the narrow spread across the ellipse.
# Trade-off: keeping fewer components gives a more compact representation that is
# cheaper to compute with, less noisy, easier to visualize, and less affected by the
# curse of dimensionality, but it loses information (higher reconstruction error).
# In practice one keeps the smallest k whose cumulative explained variance passes a
# threshold (e.g. 90-95%), or cuts at an "elbow" in the scree plot. Here, if 89% is
# enough, k=1 halves the dimension. If the features were uncorrelated with equal
# variance (tied eigenvalues), each PC would explain 50% and reduction would cost
# as much as it saves.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# Panel 1 (Q2): raw scatter showing the tilted, positively correlated ellipse.
# Panel 2 (Q3): the same data with PC1/PC2 drawn as axes through the mean, each
#   spanning mean +/- 2*sqrt(eigenvalue)*u. Equal axis scaling keeps them visibly
#   perpendicular.
#   PC1 runs along the major axis of the cloud and PC2 along the minor axis.
# Panel 3 (Q4): bar chart of per-component explained variance with the cumulative
#   curve; PC1 alone reaches ~89%, and both PCs together reach 100%.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    from matplotlib.figure import Figure

    X = np.asarray(X, dtype=float)
    mean = result['mean']
    eigenvalues = result['eigenvalues']
    components = result['components']
    fractions = explained_variance(eigenvalues)
    corr = np.corrcoef(X[:, 0], X[:, 1])[0, 1]

    fig = Figure(figsize=(16, 5), layout='constrained')
    ax_data, ax_pca, ax_var = fig.subplots(1, 3)

    ax_data.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
    ax_data.set_title(f'Synthetic correlated Gaussian data (r = {corr:.2f})')
    ax_data.set_xlabel('Feature $x_1$')
    ax_data.set_ylabel('Feature $x_2$')
    ax_data.set_aspect('equal', adjustable='datalim')
    ax_data.grid(alpha=0.3)

    ax_pca.scatter(X[:, 0], X[:, 1], s=10, alpha=0.3, color='gray', label='samples')
    for k, color in zip(range(components.shape[1]), ['tab:red', 'tab:blue']):
        direction = components[:, k] * 2 * np.sqrt(eigenvalues[k])
        for sign in (1, -1):  # eigenvector sign is arbitrary: draw both halves
            ax_pca.annotate('', xy=mean + sign * direction, xytext=mean,
                            arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax_pca.plot([], [], color=color, lw=2.5,
                    label=f'PC{k + 1}: $\\lambda$ = {eigenvalues[k]:.2f}, '
                          f'u = [{components[0, k]:.2f}, {components[1, k]:.2f}]')
    ax_pca.scatter(*mean, color='black', marker='x', s=80, zorder=3, label='mean')
    ax_pca.set_title(r'Principal axes through the mean (mean $\pm\ 2\sqrt{\lambda}\,u$)')
    ax_pca.set_xlabel('Feature $x_1$')
    ax_pca.set_ylabel('Feature $x_2$')
    ax_pca.set_aspect('equal', adjustable='datalim')
    ax_pca.legend(loc='lower right', fontsize=8)
    ax_pca.grid(alpha=0.3)

    labels = [f'PC{k + 1}' for k in range(len(fractions))]
    bars = ax_var.bar(labels, fractions, color=['tab:red', 'tab:blue'][:len(fractions)],
                      label='individual')
    ax_var.bar_label(bars, labels=[f'{f:.1%}' for f in fractions], label_type='center',
                     color='white', fontweight='bold')
    ax_var.plot(labels, np.cumsum(fractions), 'ko-', label='cumulative')
    ax_var.set_ylim(0, 1.1)
    ax_var.set_title('Explained variance ratio')
    ax_var.set_xlabel('Principal component')
    ax_var.set_ylabel('Fraction of total variance')
    ax_var.legend(loc='center right')
    ax_var.grid(axis='y', alpha=0.3)
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# I completed all three parts individually. Part 1: matrix products, eigenpairs,
# trace proof, inverse. Part 2: Poisson normalization and moments, event
# probabilities, Bernoulli MLE/MAP, Gaussian MLE and MAP, variance identity.
# Part 3: projection-variance derivation, data generation, PCA from scratch,
# explained variance, plots. I checked the closed-form answers against exact
# symbolic computation (SymPy) and the PCA against an SVD of the centered data.
# Resources: lecture slides (Lecture1_intro, Lecture2_review), NumPy and Matplotlib
# documentation. I used an AI assistant (Claude) as permitted by the course AI
# policy; the conversation log is submitted on Canvas.

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print('Part 1')
    for name, value in zip(['AB', 'BA', '(A+B)^T', 'AB^T', 'trace(A)', 'trace(AB)'],
                           matrix_operations()):
        print(f'  {name} =\n{value}')
    values, vectors = eigensystem()
    print(f'  eigenvalues = {values}\n  eigenvectors (columns) =\n{vectors}')
    print(f'  C^-1 =\n{matrix_inverse()}')

    print('Part 2')
    print(f'  Q1 (c, E[X], Var[X]) = {distribution_moments()}')
    print(f'  Q2 (P(A and B), P(B|A), P(A or B)) = {event_probabilities()}')
    print(f'  Q3 (MLE, MAP) = {coin_estimates()}')
    print(f'  Q4 (mu, sigma^2) = {gaussian_mle()}')
    print(f'  Q5 mu_MAP = {gaussian_map()}')

    print('Part 3')
    X = generate_data()
    result = pca(X)
    print(f'  sample correlation = {np.corrcoef(X.T)[0, 1]:.4f}')
    print(f'  covariance =\n{result["covariance"]}')
    print(f'  eigenvalues = {result["eigenvalues"]}')
    print(f'  components (columns) =\n{result["components"]}')
    print(f'  explained variance = {explained_variance(result["eigenvalues"])}')
    make_plots(X, result).savefig('Assignment1_pca.png', dpi=150)
    print('  saved Assignment1_pca.png')

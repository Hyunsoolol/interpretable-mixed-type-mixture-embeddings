# Centered natural-parameter regularization for posterior-score coordinate selection in von Mises-Fisher mixtures

## Abstract

Coordinate sparsity in a von Mises-Fisher mixture can refer to different
statistical targets. A nonzero coordinate in a component mean direction
describes prototype presence, whereas a coordinate that distinguishes
posterior component scores is determined by contrasts of the natural
parameters $\boldsymbol{\eta}_k=\kappa_k\boldsymbol{\mu}_k$. We define the
posterior-score contrast support as the union of the coordinate supports in
all pairwise posterior log-odds linear terms. To estimate this support, we
introduce the centered-$\eta$ coordinate group lasso (E-CGL), which penalizes
the component-centered natural-parameter vector for each observed coordinate
while leaving its common baseline unpenalized. Estimation uses a guarded
proximal generalized-EM path, followed by an unpenalized refit under selected
contrast constraints and BIC selection based on a nominal parameter count.
We establish the posterior-score cancellation identity, label invariance, the relation
between directional and natural-parameter contrasts, and the closed-form
centered proximal update. Numerical studies calibrated by oracle Bayes error
show accurate recovery of sparse posterior-score support across most examined
conditions. Recovery deteriorates in the small-sample setting that combines
high overlap with heterogeneous concentrations, and exact recovery becomes
unreliable when the support is dense or the ambient dimension is increased.

**Keywords:** directional data; finite mixture; von Mises-Fisher distribution;
group regularization; natural parameter; posterior-score contrast support

## 1. Introduction

Directional observations arise when magnitude is secondary to orientation or
when nonnegative feature vectors are normalized to the unit sphere. Examples
include normalized document representations, compositional profiles, and
directional measurements. Finite mixtures of von Mises-Fisher (vMF)
distributions provide a probability model for such data and support both
clustering and posterior uncertainty quantification (Banerjee et al., 2005;
Gopal and Yang, 2014). In high ambient dimension, however, a fitted mixture can
depend on many coordinates even when only a small subset distinguishes its
components.

Coordinate reduction in clustering serves two related but distinct purposes.
It can stabilize estimation by reducing the number of fitted parameters, and
it can identify the coordinates that explain how the fitted components differ.
The second purpose requires a precise definition of relevance. A coordinate
may be nonzero in every component prototype yet contribute no
component-specific linear contrast. Conversely, a coordinate can enter a
posterior comparison through concentration differences even when the mean
directions coincide.

The sparse-prototype model of Rossi and Barbaro (2022) applies an entrywise
penalty to the component mean directions. This target
is appropriate when the scientific question concerns the coordinates present
in the prototypes themselves. It does not, in general, coincide with the
coordinates that enter pairwise posterior component comparisons. The
difference can be consequential when the component concentrations differ.

For component $k$, let $\boldsymbol{\mu}_k$ be its unit mean direction and
$\kappa_k$ its concentration. The linear coefficient in a vMF posterior score
is the natural parameter
$\boldsymbol{\eta}_k=\kappa_k\boldsymbol{\mu}_k$. Hence the coefficient of an
observed vector in a pairwise posterior log-odds is
$\boldsymbol{\eta}_k-\boldsymbol{\eta}_{\ell}$, not
$\boldsymbol{\mu}_k-\boldsymbol{\mu}_{\ell}$. This observation shifts the
selection problem from prototype presence to heterogeneity of posterior-score
coefficients.

The distinction is related to equality-based variable selection in Gaussian
model-based clustering. Penalizing a mean toward zero addresses whether it is
present, whereas grouped or pairwise penalties can address whether component
means differ (Pan and Shen, 2007; Xie et al., 2008; Guo et al., 2010). The vMF
case adds a scale-direction coupling: a directional coefficient is multiplied
by a component-specific concentration before entering the posterior linear
score. A support definition based on the unscaled direction therefore omits a
part of the posterior comparison geometry.

We study coordinate heterogeneity in the natural-parameter space. For each
coordinate, the component-specific natural parameters are decomposed into a
common baseline and a zero-sum contrast. The contrast vector is treated as one
group, so that a coordinate is either retained or removed from all pairwise
linear score contrasts. The resulting E-CGL estimator leaves the common
baseline outside the penalty. This is important: removing a posterior-score
contrast does not require setting the corresponding density coordinate to
zero.

The paper makes three contributions. First, it formalizes posterior-score
contrast support and relates it to prototype and directional-heterogeneity
supports. Second, it develops a centered natural-parameter group penalty,
an adaptive extension, a guarded proximal generalized-EM path, and a
support-constrained refit that preserves common natural-parameter baselines.
Third, numerical experiments distinguish prototype, directional, and
posterior-score targets and identify conditions in which sparse support
recovery becomes unreliable. Each sparse method is assessed first against its
own support target.

Section 2 defines the model, support target, estimators, and fitting procedure.
Section 3 records algebraic and conditional optimization properties. Section 4
presents the numerical studies. Section 5 presents the full-data Classic3
application, with repeated-holdout stability and contrast applications reported
as supplementary analyses. Section 6 discusses the scope and limitations. The
number of mixture components is treated as fixed in the support-recovery
analysis; selecting the component resolution is a separate model-selection
problem.

## 2. Model and methodology

### 2.1. von Mises-Fisher mixtures in natural parameters

Let $\boldsymbol{x}_i\in\mathbb{S}^{d-1}$, $i=1,\ldots,n$, be unit-norm
observations. A $K$-component vMF mixture has density

$$
f(\boldsymbol{x}_i;\boldsymbol{\Theta})=\sum_{k=1}^{K}\pi_k C_d(\kappa_k)\exp\bigl(\kappa_k\boldsymbol{\mu}_k^{\mathsf T}\boldsymbol{x}_i\bigr),
$$

where $\pi_k>0$, $\sum_k\pi_k=1$,
$\lVert\boldsymbol{\mu}_k\rVert_2=1$, and $\kappa_k\geq0$. The normalizing
constant is

$$
C_d(\kappa)=\frac{\kappa^{d/2-1}}{(2\pi)^{d/2}I_{d/2-1}(\kappa)},
$$

with the value at $\kappa=0$ defined by continuous extension. Writing

$$
\boldsymbol{\eta}_k=\kappa_k\boldsymbol{\mu}_k
$$

gives the equivalent component density

$$
f_k(\boldsymbol{x}_i)=C_d\bigl(\lVert\boldsymbol{\eta}_k\rVert_2\bigr)\exp\bigl(\boldsymbol{\eta}_k^{\mathsf T}\boldsymbol{x}_i\bigr).
$$

For $\boldsymbol{\eta}_k\neq\boldsymbol{0}$, the original parameters are
$\kappa_k=\lVert\boldsymbol{\eta}_k\rVert_2$ and
$\boldsymbol{\mu}_k=\boldsymbol{\eta}_k/\lVert\boldsymbol{\eta}_k\rVert_2$.
At $\boldsymbol{\eta}_k=\boldsymbol{0}$ the component is uniform and its
direction is not identified. The observed log-likelihood is

$$
\ell(\boldsymbol{\Theta})=\sum_{i=1}^{n}\log\left[\sum_{k=1}^{K}\pi_k C_d\bigl(\lVert\boldsymbol{\eta}_k\rVert_2\bigr)\exp\bigl(\boldsymbol{\eta}_k^{\mathsf T}\boldsymbol{x}_i\bigr)\right].
$$

### 2.2. Posterior-score contrast support

The score associated with component $k$ is

$$
s_k(\boldsymbol{x})=\log\pi_k+\log C_d\bigl(\lVert\boldsymbol{\eta}_k\rVert_2\bigr)+\boldsymbol{\eta}_k^{\mathsf T}\boldsymbol{x}.
$$

Consequently, for $k\neq\ell$,

$$
\log\frac{\Pr(Z=k\mid\boldsymbol{x})}{\Pr(Z=\ell\mid\boldsymbol{x})}=a_{k\ell}+(\boldsymbol{\eta}_k-\boldsymbol{\eta}_{\ell})^{\mathsf T}\boldsymbol{x},
$$

where

$$
a_{k\ell}=\log\frac{\pi_k}{\pi_{\ell}}+\log\frac{C_d\bigl(\lVert\boldsymbol{\eta}_k\rVert_2\bigr)}{C_d\bigl(\lVert\boldsymbol{\eta}_{\ell}\rVert_2\bigr)}.
$$

The intercept $a_{k\ell}$ and the feature-dependent linear term play
different roles. Our target concerns the latter.

Let

$$
\boldsymbol{H}_K=\boldsymbol{I}_K-K^{-1}\boldsymbol{1}_K\boldsymbol{1}_K^{\mathsf T}
$$

be the centering matrix. At coordinate $j$, collect the natural parameters as
$\boldsymbol{\eta}_{\cdot j}=(\eta_{1j},\ldots,\eta_{Kj})^{\mathsf T}$ and set

$$
\bar\eta_j=K^{-1}\sum_{k=1}^{K}\eta_{kj},\qquad \boldsymbol{c}^{(\eta)}_j=\boldsymbol{H}_K\boldsymbol{\eta}_{\cdot j}=\boldsymbol{\eta}_{\cdot j}-\bar\eta_j\boldsymbol{1}_K.
$$

The posterior-score contrast support is

$$
S_{\eta}=\left\{j:\lVert\boldsymbol{c}^{(\eta)}_j\rVert_2>0\right\}.
$$

If $j\notin S_{\eta}$, the coefficient of $x_j$ is zero in every pairwise
posterior log-odds linear term. This does not imply that the coordinate is
absent from the component densities. A nonzero common baseline $\bar\eta_j$
is retained and can contribute to the concentration norms and thus to the
pairwise intercepts.

### 2.3. Centered-$\eta$ coordinate group lasso

For fixed $K$ and tuning parameter $\lambda_{\eta}\geq0$, E-CGL maximizes

$$
\mathcal{L}_{\lambda_{\eta}}(\boldsymbol{\Theta})=\ell(\boldsymbol{\Theta})-\lambda_{\eta}\sum_{j=1}^{d}\lVert\boldsymbol{H}_K\boldsymbol{\eta}_{\cdot j}\rVert_2.
$$

Equivalently, the penalty is

$$
P_{\mathrm{E\text{-}CGL}}(\boldsymbol{\eta})=\lambda_{\eta}\sum_{j=1}^{d}\lVert\boldsymbol{c}^{(\eta)}_j\rVert_2.
$$

Each group contains one observed coordinate across all component contrasts.
The penalty therefore matches the unit of selection in $S_{\eta}$. The common
component $K^{-1}\boldsymbol{1}_K\boldsymbol{1}_K^{\mathsf T}\boldsymbol{\eta}_{\cdot j}$
lies in the null space of $\boldsymbol{H}_K$ and is not penalized. This differs
from a raw-natural-parameter group penalty because

$$
\lVert\boldsymbol{\eta}_{\cdot j}\rVert_2^2=K\bar\eta_j^2+\lVert\boldsymbol{c}^{(\eta)}_j\rVert_2^2.
$$

A raw group penalty would shrink common and component-specific effects
together. It also differs from an entrywise penalty on $c_{kj}$, which can
remove individual component-coordinate entries without selecting the
coordinate contrast as a unit. The coordinate grouping follows the group
selection principle of Yuan and Lin (2006), applied here to a centered
natural-parameter contrast.

### 2.4. Adaptive extension

E-ACGL replaces the common group weight by fixed coordinate-specific weights:

$$
P_{\mathrm{E\text{-}ACGL}}(\boldsymbol{\eta})=\lambda_{\eta}\sum_{j=1}^{d}w_j\lVert\boldsymbol{c}^{(\eta)}_j\rVert_2.
$$

From a dense initial fit, define

$$
w_j^{\mathrm{raw}}=\left(\lVert\boldsymbol{c}^{(\eta,\mathrm{init})}_j\rVert_2+\epsilon\right)^{-\gamma},\qquad w_j=\frac{w_j^{\mathrm{raw}}}{\operatorname{median}_{1\leq h\leq d}w_h^{\mathrm{raw}}}.
$$

The numerical studies use $\gamma=1$ and $\epsilon=10^{-6}$. The weights are
computed once and held fixed along the path. Median normalization separates
relative weighting from the overall scale of $\lambda_{\eta}$. E-ACGL is
treated as an empirical adaptive extension of E-CGL.

### 2.5. Prototype and directional comparators

Let $\boldsymbol{M}$ be the $K\times d$ matrix with rows
$\boldsymbol{\mu}_k^{\mathsf T}$, and let
$\boldsymbol{E}=\operatorname{diag}(\boldsymbol{\kappa})\boldsymbol{M}$ be the
matrix of natural parameters. Three coordinate supports are relevant:

$$
S_P=\left\{j:\lVert\boldsymbol{\mu}_{\cdot j}\rVert_2>0\right\},\qquad S_{\mu}=\left\{j:\lVert\boldsymbol{H}_K\boldsymbol{\mu}_{\cdot j}\rVert_2>0\right\},\qquad S_{\eta}=\left\{j:\lVert\boldsymbol{H}_K\boldsymbol{\eta}_{\cdot j}\rVert_2>0\right\}.
$$

M-L denotes the sparse vMF prototype model of Rossi and Barbaro (2022) and
targets $S_P$ through $\sum_{k,j}\lvert\mu_{kj}\rvert$. M-CGL is a matched
directional comparator for $S_{\mu}$, defined by

$$
\ell(\boldsymbol{\Theta})-\lambda_{\mu}\sum_{j=1}^{d}\lVert\boldsymbol{H}_K\boldsymbol{\mu}_{\cdot j}\rVert_2,
\qquad
\lVert\boldsymbol{\mu}_k\rVert_2=1.
$$

The unit-sphere constraints make the directional optimization nonseparable;
the implementation uses variable splitting and updates on the product of
spheres. Its computational details are reserved for the Supplement. E-CGL is
the primary estimator for $S_{\eta}$, and E-ACGL is its adaptive extension.

### 2.6. Guarded proximal generalized-EM estimation

At outer iteration $t$, the responsibilities are

$$
\tau_{ik}^{(t)}=\frac{\pi_k^{(t)}C_d\bigl(\lVert\boldsymbol{\eta}_k^{(t)}\rVert_2\bigr)\exp\bigl((\boldsymbol{\eta}_k^{(t)})^{\mathsf T}\boldsymbol{x}_i\bigr)}{\sum_{h=1}^{K}\pi_h^{(t)}C_d\bigl(\lVert\boldsymbol{\eta}_h^{(t)}\rVert_2\bigr)\exp\bigl((\boldsymbol{\eta}_h^{(t)})^{\mathsf T}\boldsymbol{x}_i\bigr)}.
$$

Define

$$
N_k^{(t)}=\sum_{i=1}^{n}\tau_{ik}^{(t)},\qquad \boldsymbol{r}_k^{(t)}=\sum_{i=1}^{n}\tau_{ik}^{(t)}\boldsymbol{x}_i.
$$

Conditional on these responsibilities, the natural-parameter block minimizes
$F_t(\boldsymbol{\eta})=f_t(\boldsymbol{\eta})+g(\boldsymbol{\eta})$, where

$$
f_t(\boldsymbol{\eta})=-\sum_{k=1}^{K}\left[(\boldsymbol{r}_k^{(t)})^{\mathsf T}\boldsymbol{\eta}_k+N_k^{(t)}\log C_d\bigl(\lVert\boldsymbol{\eta}_k\rVert_2\bigr)\right]
$$

and

$$
g(\boldsymbol{\eta})=\lambda_{\eta}\sum_{j=1}^{d}w_j\lVert\boldsymbol{H}_K\boldsymbol{\eta}_{\cdot j}\rVert_2.
$$

The smooth term is convex because $-\log C_d(\lVert\boldsymbol{\eta}_k\rVert_2)$
is the vMF log-partition function. Its gradient is

$$
\nabla_{\boldsymbol{\eta}_k}f_t(\boldsymbol{\eta})=N_k^{(t)}A_d\bigl(\lVert\boldsymbol{\eta}_k\rVert_2\bigr)\frac{\boldsymbol{\eta}_k}{\lVert\boldsymbol{\eta}_k\rVert_2}-\boldsymbol{r}_k^{(t)},
$$

with continuous value $-\boldsymbol{r}_k^{(t)}$ at the origin. Here $A_d$ is
the vMF mean-resultant-length function.

Let $\boldsymbol{J}_K=K^{-1}\boldsymbol{1}_K\boldsymbol{1}_K^{\mathsf T}$,
$\boldsymbol{H}_K=\boldsymbol{I}_K-\boldsymbol{J}_K$, and

$$
\boldsymbol{v}=\boldsymbol{\eta}^{(m)}-s\nabla f_t\bigl(\boldsymbol{\eta}^{(m)}\bigr).
$$

For coordinate $j$, the proximal-gradient update is

$$
\boldsymbol{\eta}_{\cdot j}^{(m+1)}=\boldsymbol{J}_K\boldsymbol{v}_{\cdot j}+\left(1-\frac{s\lambda_{\eta}w_j}{\lVert\boldsymbol{H}_K\boldsymbol{v}_{\cdot j}\rVert_2}\right)_+\boldsymbol{H}_K\boldsymbol{v}_{\cdot j}.
$$

The first term preserves the common baseline; the second applies block soft
thresholding only to the centered contrast (Parikh and Boyd, 2014). The
initial step size is $s=1/\max_kN_k^{(t)}$. It is halved until the smooth
majorization inequality

$$
f_t(\boldsymbol{\eta}^{(m+1)})\leq f_t(\boldsymbol{\eta}^{(m)})+\left\langle\nabla f_t(\boldsymbol{\eta}^{(m)}),\boldsymbol{\eta}^{(m+1)}-\boldsymbol{\eta}^{(m)}\right\rangle+\frac{\lVert\boldsymbol{\eta}^{(m+1)}-\boldsymbol{\eta}^{(m)}\rVert_F^2}{2s}
$$

is satisfied within numerical tolerance. The candidate is accepted only when
both the penalized auxiliary function and the observed penalized
log-likelihood do not decrease beyond the acceptance tolerance. A path point
that exhausts the backtracking budget is marked as failed rather than treated
as a converged zero-change update.

The complete fitting procedure is summarized below.

**Algorithm 1. Guarded path algorithm for E-CGL and E-ACGL**

1. Fit the finite-concentration dense vMF model from multiple starts and,
   among fits satisfying the numerical stopping criterion, retain the one
   with the largest observed log-likelihood.
2. Set $w_j=1$ for E-CGL. For E-ACGL, compute the adaptive weights from the
   dense fit and hold them fixed.
3. Construct a zero-augmented, KKT-scaled geometric path for
   $\lambda_{\eta}$.
4. For each path value, warm-start from the preceding accepted fit and repeat:
   compute responsibilities; update $\boldsymbol{\pi}$; solve the conditional
   natural-parameter block by guarded proximal-gradient steps; and accept the
   candidate only after the majorization and objective checks pass.
5. Store the support and numerical diagnostics for every accepted path point.
6. Refit every distinct support without a sparsity penalty under the selected
   contrast constraints in
   Section 2.7.
7. Select the refitted support by BIC based on the nominal parameter count.

The likelihood of a finite vMF mixture can be unbounded under concentration
collapse (Ng, 2023). All dense, path, and refit calculations are therefore
defined on the finite computational parameter space

$$
0\leq\kappa_k=\lVert\boldsymbol{\eta}_k\rVert_2\leq\kappa_{\max},\qquad \kappa_{\max}=10^6.
$$

Proposals outside this space are rejected in the dense fit, path fit, and
refit. This restriction defines the numerical problem and is not an
additional sparsity penalty. Boundary diagnostics are reported in the
Supplement.

### 2.7. Unpenalized refit under selected contrast constraints

For a path support $S$, the penalty is removed and the model is refitted under

$$
\eta_{kj}=b_j+c_{kj},\qquad \sum_{k=1}^{K}c_{kj}=0,qquad c_{kj}=0\ \text{for}\ j\notin S.
$$

Thus an inactive coordinate retains its common baseline $b_j$, while its
component contrast is fixed at zero. The refit imposes

$$
\left\{
j:
\left\lVert\boldsymbol{H}_K\boldsymbol{\eta}_{\cdot j}\right\rVert_2>0
\right\}
\subseteq S.
$$

An active contrast may collapse to zero during numerical refitting. Let
$\widetilde{\boldsymbol{\Theta}}_S$ denote the parameter returned by this
numerical constrained-refit routine. The sparsity penalty is removed, but a
global maximum of the nonconvex mixture likelihood is not assumed.

For $m_S=\lvert S\rvert$, E-CGL uses the nominal dimension

$$
\operatorname{df}_{\eta}^{\mathrm{nom}}(S)=d+(K-1)m_S+(K-1)\mathbf{1}(m_S>0).
$$

The formula counts $d$ common baselines, $(K-1)m_S$ free zero-sum contrasts,
and $K-1$ mixing proportions at a generic interior configuration with
pairwise-distinct component distributions. When $m_S=0$, all component
densities coincide and the mixing proportions are not separately identified,
which motivates the indicator term. Coincident components, boundary
estimates, or active contrasts that refit to zero can have smaller local
dimension. For each distinct path support,

$$
\operatorname{BIC}^{\mathrm{refit}}(S)=-2\ell\bigl(\widetilde{\boldsymbol{\Theta}}_S\bigr)+\log(n)\operatorname{df}_{\eta}^{\mathrm{nom}}(S).
$$

The selected support is

$$
\widehat S=\underset{S\in\mathcal{S}_{\mathrm{path}}}{\operatorname{argmin}}\ \operatorname{BIC}^{\mathrm{refit}}(S).
$$

This is a nominal generic parameter count for the penalty-free constrained
refit, not an exact effective degrees of freedom. It does not account for
mixture singularities, boundary solutions, or the data-dependent search over
path supports. EBIC and alternative nominal dimensions are examined as
sensitivity analyses.

## 3. Properties of the support target and algorithm

### 3.1. Posterior-score cancellation and label invariance

Let

$$
\boldsymbol{c}^{(\eta)}_j=\boldsymbol{H}_K\boldsymbol{\eta}_{\cdot j}.
$$

The following result gives the statistical interpretation of a zero group.

**Proposition 1.** For any coordinate $j$, the following statements are
equivalent:

$$
\boldsymbol{c}^{(\eta)}_j=\boldsymbol{0},
$$

$$
\eta_{1j}=\cdots=\eta_{Kj},
$$

and $x_j$ is absent from the linear term of every pairwise posterior
log-score contrast.

**Proof.** The equality
$\boldsymbol{H}_K\boldsymbol{\eta}_{\cdot j}=\boldsymbol{0}$ holds if and
only if $\boldsymbol{\eta}_{\cdot j}$ belongs to the span of
$\boldsymbol{1}_K$. For any pair $(k,\ell)$, the coefficient of $x_j$ in the
posterior log-score contrast is $\eta_{kj}-\eta_{\ell j}$, so all such
coefficients vanish exactly under the same condition. $\square$

The result concerns the covariate-dependent linear term. Differences in
$\pi_k$ or $C_d(\kappa_k)$ may remain in the intercept. The group norm also
has the pairwise representation

$$
\left\lVert\boldsymbol{c}^{(\eta)}_j\right\rVert_2^2
=
\frac{1}{K}\sum_{1\leq k<\ell\leq K}
(\eta_{kj}-\eta_{\ell j})^2.
$$

Thus E-CGL performs coordinate selection using the aggregate dispersion of
all pairwise natural-parameter contrasts. It does not identify which subset
of component pairs differs at an active coordinate; that would require a
pairwise fusion formulation.

Let $\boldsymbol{P}$ be a $K\times K$ permutation matrix. Since
$\boldsymbol{P}\boldsymbol{1}_K=\boldsymbol{1}_K$,

$$
\boldsymbol{H}_K\boldsymbol{P}
=
\boldsymbol{P}\boldsymbol{H}_K.
$$

Consequently,

$$
\left\lVert
\boldsymbol{H}_K\boldsymbol{P}\boldsymbol{\eta}_{\cdot j}
\right\rVert_2
=
\left\lVert
\boldsymbol{P}\boldsymbol{H}_K\boldsymbol{\eta}_{\cdot j}
\right\rVert_2
=
\left\lVert
\boldsymbol{H}_K\boldsymbol{\eta}_{\cdot j}
\right\rVert_2.
$$

Both the penalty value and the selected coordinate support are invariant to
component relabeling. This invariance does not itself establish parameter
identifiability. Under standard finite-mixture identifiability conditions,
including positive weights and distinct component distributions for fixed
$K$, parameters are identifiable at most up to label permutation. If two
components have identical natural parameters, their individual mixing
weights are not separately identified.

### 3.2. Directional and natural-parameter heterogeneity

Directional and posterior-score supports need not coincide. A sufficient
condition guaranteeing their equality is a common positive concentration. To
describe their relation, write

$$
\boldsymbol{\kappa}
=
\bar\kappa\boldsymbol{1}_K+\boldsymbol{\delta}_{\kappa},
\qquad
\boldsymbol{1}_K^{\mathsf T}\boldsymbol{\delta}_{\kappa}=0,
$$

and, for coordinate $j$,

$$
\boldsymbol{\mu}_{\cdot j}
=
\bar\mu_j\boldsymbol{1}_K+\boldsymbol{c}^{(\mu)}_j,
\qquad
\boldsymbol{c}^{(\mu)}_j
=
\boldsymbol{H}_K\boldsymbol{\mu}_{\cdot j}.
$$

Because $\boldsymbol{\eta}_{\cdot j}=\boldsymbol{\kappa}\odot
\boldsymbol{\mu}_{\cdot j}$,

$$
\boldsymbol{c}^{(\eta)}_j
=
\bar\kappa\boldsymbol{c}^{(\mu)}_j
+
\bar\mu_j\boldsymbol{\delta}_{\kappa}
+
\boldsymbol{H}_K
\left(
\boldsymbol{\delta}_{\kappa}\odot
\boldsymbol{c}^{(\mu)}_j
\right).
$$

The three terms represent directional variation, concentration variation
along the common direction, and their interaction. They need not be
orthogonal and can reinforce or cancel one another. Hence neither
$S_{\mu}\subseteq S_{\eta}$ nor $S_{\eta}\subseteq S_{\mu}$ holds in
general.

**Corollary 1.** If
$\kappa_1=\cdots=\kappa_K=\kappa>0$, then

$$
\boldsymbol{c}^{(\eta)}_j
=
\kappa\boldsymbol{c}^{(\mu)}_j,
\qquad
S_{\eta}=S_{\mu}.
$$

If the component directions are common but the concentrations differ, then
$\boldsymbol{c}^{(\mu)}_j=\boldsymbol{0}$ while

$$
\boldsymbol{c}^{(\eta)}_j
=
\bar\mu_j\boldsymbol{\delta}_{\kappa}.
$$

Such a coordinate is directionally common but can enter posterior-score
contrasts through concentration heterogeneity. Conversely, when every
$\kappa_k>0$, a coordinate can have a common natural-parameter coefficient
while its directional coefficients differ because
$\mu_{kj}=\eta_{kj}/\kappa_k$. These cases motivate the separate estimands
used in Section 4.

### 3.3. Conditional optimization properties

For fixed responsibilities, the negative conditional vMF criterion is
convex in each natural parameter. Let $f(\boldsymbol{E})$ denote its smooth
part, where the rows of $\boldsymbol{E}$ are
$\boldsymbol{\eta}_1^{\mathsf T},\ldots,
\boldsymbol{\eta}_K^{\mathsf T}$. For a step size $s>0$, define

$$
\boldsymbol{V}
=
\boldsymbol{E}-s\nabla f(\boldsymbol{E}).
$$

The proximal map of the centered group penalty separates by coordinate.

**Proposition 2.** For coordinate $j$,

$$
\operatorname{prox}_{s\lambda_{\eta}w_j
\lVert\boldsymbol{H}_K\cdot\rVert_2}
(\boldsymbol{V}_{\cdot j})
=
\boldsymbol{J}_K\boldsymbol{V}_{\cdot j}
+
\left(
1-
\frac{s\lambda_{\eta}w_j}
{\left\lVert\boldsymbol{H}_K\boldsymbol{V}_{\cdot j}\right\rVert_2}
\right)_+
\boldsymbol{H}_K\boldsymbol{V}_{\cdot j}.
$$

The centered term is defined as zero when
$\left\lVert\boldsymbol{H}_K\boldsymbol{V}_{\cdot j}\right\rVert_2=0$.
This is the unconstrained proximal map used before the finite-concentration
guard; it is not the proximal operator of the concentration-cap constraint.
The common coordinate mean is unchanged by this map, while the centered
contrast is group-soft-thresholded. The proof follows from the orthogonal
decomposition of $\mathbb{R}^K$ into the common subspace and its centered
complement.

Let $Q_{\lambda_{\eta}}$ denote the penalized expected complete-data
log-likelihood. A majorized proximal step decreases the conditional
minimization criterion, equivalently increasing $Q_{\lambda_{\eta}}$. With
zero acceptance tolerance, an accepted generalized-EM update satisfying

$$
Q_{\lambda_{\eta}}
\left(
\boldsymbol{\Theta}^{(t+1)}\mid\boldsymbol{\tau}^{(t)}
\right)
\geq
Q_{\lambda_{\eta}}
\left(
\boldsymbol{\Theta}^{(t)}\mid\boldsymbol{\tau}^{(t)}
\right)
$$

also has a nondecreasing penalized observed objective by the generalized-EM
inequality. The implemented positive tolerance permits numerical
near-monotonicity, which is checked from the objective trace.

These statements concern the support target and accepted updates for fixed
$K$. They do not establish global optimality, selection consistency, or an
exact effective degrees of freedom for the mixture. The finite concentration
bound defines the computational parameter space. When
$\boldsymbol{\eta}_k=\boldsymbol{0}$, the natural parameter remains defined
although its directional representation is not identified.

## 4. Numerical studies

### 4.1. Design and evaluation criteria

The numerical experiments were organized around the support estimand rather
than a single ranking of clustering algorithms. All observations were drawn
from a $K=4$ vMF mixture with equal mixing proportions. The primary design
used $d=200$ coordinates partitioned into four common, 16 posterior-score,
and 180 null coordinates. Sample sizes were $n\in\{300,1000\}$, and the
concentrations were either common,

$$
(\kappa_1,\kappa_2,\kappa_3,\kappa_4)=(45,45,45,45),
$$

or heterogeneous,

$$
(\kappa_1,\kappa_2,\kappa_3,\kappa_4)=(30,40,50,60).
$$

The clustering difficulty was calibrated by the oracle Bayes error

$$
e_B
=
\Pr_{\boldsymbol{\Theta}^{\star}}
\left[
\underset{k}{\operatorname{argmax}}
\Pr(Z=k\mid\boldsymbol{X};\boldsymbol{\Theta}^{\star})
\neq Z
\right],
$$

with targets $e_B\in\{0.025,0.05,0.10\}$. For a fixed concentration
pattern, the common-to-contrast signal allocation was varied and calibrated
by Monte Carlo bisection. This changes overlap without redefining the
concentration pattern. The resulting population parameters were checked to
preserve the intended common, active, and null coordinate sets.

Each cell contained 100 independent repetitions. A separate test sample of
5,000 observations was used for predictive evaluation. Methods fitted to the
same repetition received the same training and test data. We compared
spherical $k$-means, dense vMF mixtures with common and component-specific
concentrations, the Rossi-type entrywise prototype lasso (M-L), M-CGL,
E-CGL, and E-ACGL. E-ACGL is reported as an adaptive extension rather than
as a second primary method. The sparse fits used ten starts. M-L, E-CGL, and
E-ACGL used 240 path points. Because of its higher computational cost,
M-CGL candidates from 60- and 120-point paths were combined before support
refitting.

Distinct supports were refitted before BIC selection. A method was scored
against its stated target: M-L against prototype support $S_P$, M-CGL
against directional support $S_{\mu}$, and E-CGL or E-ACGL against
posterior-score support $S_{\eta}$. Cross-target F1 scores were also
recorded in the estimand diagnostics. ARI, NMI, test negative
log-likelihood (NLL), parameter MSE, elapsed time, and numerical failure
rates were common outcomes. A smaller test NLL is preferable; vMF densities
can exceed one, so the reported average NLL may be negative. Across the 12
primary cells, the largest Monte Carlo standard errors for E-CGL and E-ACGL
were 0.011 for $F_{1,\eta}$, 0.0055 for ARI, 0.81 coordinates for selected
support size, 0.029 for $\operatorname{MSE}_{\eta}$, 0.047 for exact-support
rates, and 0.011 for mean test NLL.

### 4.2. Posterior-score support recovery

Table 1 reports the primary E-CGL results and the adaptive extension. The
true posterior-score support size is 16 in every cell.

**Table 1. Posterior-score recovery over 100 repetitions.**

| $e_B$ | $n$ | $\kappa$ | E-CGL $q$ | E-CGL F1 | E-CGL exact | E-CGL ARI | E-ACGL $q$ | E-ACGL F1 | E-ACGL exact | E-ACGL ARI |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.025 | 300 | Common | 16.14 | 0.996 | 0.86 | 0.927 | 16.14 | 0.996 | 0.86 | 0.927 |
| 0.025 | 1000 | Common | 16.02 | 0.999 | 0.98 | 0.933 | 16.02 | 0.999 | 0.98 | 0.933 |
| 0.025 | 300 | Heterogeneous | 16.15 | 0.996 | 0.88 | 0.925 | 16.15 | 0.996 | 0.88 | 0.925 |
| 0.025 | 1000 | Heterogeneous | 16.02 | 0.999 | 0.98 | 0.928 | 16.02 | 0.999 | 0.98 | 0.928 |
| 0.050 | 300 | Common | 16.26 | 0.992 | 0.77 | 0.856 | 16.23 | 0.993 | 0.79 | 0.856 |
| 0.050 | 1000 | Common | 16.06 | 0.998 | 0.95 | 0.867 | 16.06 | 0.998 | 0.95 | 0.867 |
| 0.050 | 300 | Heterogeneous | 16.60 | 0.983 | 0.69 | 0.857 | 16.18 | 0.995 | 0.85 | 0.858 |
| 0.050 | 1000 | Heterogeneous | 16.02 | 0.999 | 0.98 | 0.869 | 16.02 | 0.999 | 0.98 | 0.869 |
| 0.100 | 300 | Common | 16.28 | 0.992 | 0.76 | 0.724 | 16.33 | 0.987 | 0.77 | 0.721 |
| 0.100 | 1000 | Common | 16.02 | 0.999 | 0.98 | 0.747 | 16.02 | 0.999 | 0.98 | 0.747 |
| 0.100 | 300 | Heterogeneous | 24.34 | 0.768 | 0.01 | 0.638 | 15.23 | 0.948 | 0.22 | 0.702 |
| 0.100 | 1000 | Heterogeneous | 16.47 | 0.986 | 0.72 | 0.753 | 16.01 | 1.000 | 0.99 | 0.753 |

*Note:* F1 denotes $F_{1,\eta}$; exact denotes the proportion selecting the
16-coordinate posterior-score support exactly.

E-CGL selected between 16.02 and 16.60 coordinates and attained
$F_{1,\eta}\geq0.983$ in 11 of the 12 cells. The exception combined the
small sample, highest overlap, and heterogeneous concentrations. There,
E-CGL selected 24.34 coordinates on average and attained
$F_{1,\eta}=0.768$. The adaptive weighting reduced the selected support to
15.23 and increased $F_{1,\eta}$ to 0.948, although exact recovery remained
0.22. At $n=1000$, both procedures were close to the 16-coordinate target
for every overlap level. ARI declined as the oracle error increased even
when support recovery remained accurate, as expected from the change in
population overlap.

Table 2 gives a representative comparison at $e_B=0.05$ with heterogeneous
concentrations. The target support sizes differ by method: 20 coordinates
belong to the directional support in this design, whereas 16 belong to the
posterior-score support.

**Table 2. Representative all-method comparison at $e_B=0.05$ and heterogeneous concentrations.**

| $n$ | Method | Selected $q$ | Target-specific F1 | ARI | Test NLL | Median sec./rep. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | Spherical $k$-means | NA | NA | 0.580 | NA | 0.06 |
| 300 | Dense vMF, common $\kappa$ | NA | NA | 0.649 | -245.860 | 1.20 |
| 300 | Dense vMF, free $\kappa_k$ | NA | NA | 0.709 | -246.082 | 2.82 |
| 300 | M-L | 199.70 | $F_{1,P}=0.182$ | 0.718 | -246.122 | 9.13 |
| 300 | M-CGL | 20.73 | $F_{1,\mu}=0.897$ | 0.812 | -247.060 | 434.47 |
| 300 | E-CGL | 16.60 | $F_{1,\eta}=0.983$ | 0.857 | -247.206 | 50.26 |
| 300 | E-ACGL | 16.18 | $F_{1,\eta}=0.995$ | 0.858 | -247.215 | 43.65 |
| 1000 | Spherical $k$-means | NA | NA | 0.773 | NA | 0.25 |
| 1000 | Dense vMF, common $\kappa$ | NA | NA | 0.765 | -247.004 | 1.59 |
| 1000 | Dense vMF, free $\kappa_k$ | NA | NA | 0.835 | -247.228 | 4.54 |
| 1000 | M-L | 199.67 | $F_{1,P}=0.182$ | 0.836 | -247.232 | 29.05 |
| 1000 | M-CGL | 20.04 | $F_{1,\mu}=0.999$ | 0.867 | -247.510 | 496.21 |
| 1000 | E-CGL | 16.02 | $F_{1,\eta}=0.999$ | 0.869 | -247.517 | 46.88 |
| 1000 | E-ACGL | 16.02 | $F_{1,\eta}=0.999$ | 0.869 | -247.517 | 42.36 |

*Note:* Support F1 values refer to different population targets and are not
cross-method ranks. Test NLL is the mean negative log-likelihood per test
observation. NA denotes a method without support selection or probabilistic
test density.

M-L retained nearly all coordinates and had low prototype-support recovery
($F_{1,P}=0.182$). M-CGL and E-CGL both matched
their own targets closely at $n=1000$. The corresponding ARI and test NLL
were similar, but this agreement in clustering performance does not make the
selected coordinate sets interchangeable.

### 4.3. Directional versus posterior-score estimands

Four diagnostics were used to separate the estimands. In the common-
$\kappa$ design, $S_{\mu}=S_{\eta}$. The pure-concentration design used a
common direction and unequal concentrations, so $S_{\mu}$ was empty while
$S_{\eta}$ contained 16 coordinates. The shared-canonical design contained
80 coordinates with common natural-parameter coefficients and 20
posterior-score coordinates; unequal concentrations can make the common
canonical coordinates directionally heterogeneous. The crossed design
contained four $\eta$-only, four $\mu$-only, eight common active, and eight
null coordinates.

**Table 3. Estimand-separation diagnostics over 100 repetitions.**

| Design | Method | Selected $q$ | $F_{1,\mu}$ | $F_{1,\eta}$ | Exact own target | ARI |
|---|---|---:|---:|---:|---:|---:|
| Common $\kappa$ | M-CGL | 16.04 | 0.999 | 0.999 | 0.96 | 0.867 |
| Common $\kappa$ | E-CGL | 16.06 | 0.998 | 0.998 | 0.95 | 0.867 |
| Pure concentration | M-CGL | 0.77 | - | 0.004 | 0.24 | 0.674 |
| Pure concentration | E-CGL | 16.33 | 0.000 | 0.990 | 0.77 | 0.635 |
| Shared canonical background | M-CGL | 21.16 | 0.348 | 0.973 | 0.00 | 0.859 |
| Shared canonical background | E-CGL | 20.02 | 0.333 | 1.000 | 0.98 | 0.870 |
| Crossed support | M-CGL | 11.67 | 0.983 | 0.677 | 0.64 | 0.996 |
| Crossed support | E-CGL | 11.87 | 0.681 | 0.980 | 0.69 | 0.996 |

For the pure-concentration cell, ``Exact own target'' is the empty-support
selection rate for M-CGL rather than an F1 score. M-CGL correctly represented
the absence of directional heterogeneity in 24% of BIC-selected fits, while
E-CGL recovered the 16 posterior-score coordinates with mean
$F_{1,\eta}=0.990$. In the crossed design, each method had high mean F1 for
its own support but exact recovery rates of 0.64 for M-CGL and 0.69 for
E-CGL; both had ARI 0.996 and scored substantially lower against the other
estimand. The shared-canonical design was more difficult for M-CGL: its
BIC-selected directional F1 was 0.348, and even the best candidate on each
path averaged 0.709. The exact directional support did not occur on the
recorded path. E-CGL had mean $F_{1,\eta}=0.9995$ for the 20-coordinate
posterior-score support, with exact recovery in 98% of repetitions. These
results support target-specific evaluation.

### 4.4. Sample size, oracle benchmarks, and selector sensitivity

The sample-size experiment fixed $e_B=0.05$ and used
$n\in\{300,600,1000,2000\}$. The path oracle denotes the candidate with the
largest target-specific F1 in each repetition. The oracle-support refit uses
the known population support. Define

$$
\Delta_{\mathrm{sel}}
=
F_1^{\mathrm{path\ oracle}}-F_1^{\mathrm{selected}},
$$

and

$$
\Delta_{\mathrm{NLL}}
=
\mathrm{NLL}(\widehat{\boldsymbol{\Theta}})
-
\mathrm{NLL}(\widehat{\boldsymbol{\Theta}}_{S^{\star}}^{\mathrm{oracle\ support}}).
$$

The latter is a paired finite-sample comparison with a known-support refit,
not a theoretical oracle property.

**Table 4. Endpoint target-specific recovery and oracle gaps.**

| Method | $\kappa$ pattern | $n$ | Target $q$ | Selected $q$ | Target F1 | Exact | $\Delta_{\mathrm{sel}}$ | $\Delta_{\mathrm{NLL}}$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E-CGL | Common | 300 | 16 | 16.26 | 0.992 | 0.77 | 0.008 | 0.0086 |
| E-CGL | Common | 2000 | 16 | 16.00 | 1.000 | 1.00 | 0.000 | 0.0000 |
| E-CGL | Heterogeneous | 300 | 16 | 16.60 | 0.983 | 0.69 | 0.007 | 0.0139 |
| E-CGL | Heterogeneous | 2000 | 16 | 16.00 | 1.000 | 1.00 | 0.000 | 0.0000 |
| M-CGL | Common | 300 | 16 | 16.23 | 0.993 | 0.79 | 0.007 | 0.0079 |
| M-CGL | Common | 2000 | 16 | 16.00 | 1.000 | 1.00 | 0.000 | 0.0000 |
| M-CGL | Heterogeneous | 300 | 20 | 20.73 | 0.897 | 0.08 | 0.014 | 0.1366 |
| M-CGL | Heterogeneous | 2000 | 20 | 20.00 | 1.000 | 1.00 | 0.000 | 0.0000 |

*Note:* $\Delta_{\mathrm{sel}}$ uses the best fitted path candidate, whereas
$\Delta_{\mathrm{NLL}}$ uses a separate refit under the known population
support. Results for $n=600$ and $n=1000$ are reported in the Supplement.

Target-specific F1 and exact recovery generally approached one with
increasing sample size. Small Monte Carlo reversals occurred between 600 and
1,000 observations in two common-concentration cells, so strict monotonicity
is not claimed. By $n=1000$, all selector gaps were at most 0.0018 and all
absolute paired test-NLL gaps were below 0.0007. The heterogeneous
directional target had the largest gaps at $n=300$.

The selector sensitivity isolated two different failure mechanisms. In the
pure-concentration M-CGL diagnostic, the empty support appeared on every
path, but its exact-empty-support rate under BIC was 0.24. EBIC with
$\gamma=1$ raised this rate to 0.97, and the path oracle selected it in every
repetition. This was primarily a selector issue. In the hard E-CGL cell
$(e_B=0.10,n=300)$ with heterogeneous concentrations, BIC, EBIC with
$\gamma=1$, and the path oracle attained mean $F_{1,\eta}$ values of 0.768,
0.789, and 0.814, respectively. The true support occurred on only about 2%
of paths, indicating that path construction and estimation error also
contributed. E-ACGL attained 0.948 under BIC in the same cell.

### 4.5. Stress conditions and computation

Table 5 examines weaker signals, higher ambient dimension, and dense
posterior-score supports. The high-dimensional design used $d=500$ with 40
active coordinates. The moderately and strongly dense designs used 80 and
160 active coordinates, respectively.

**Table 5. Stress-condition results over 100 repetitions.**

| Condition | Method | Selected $q$ | $F_{1,\eta}$ | Exact support | ARI |
|---|---|---:|---:|---:|---:|
| Weak-signal beta-min, $n=1000$ | E-CGL | 15.99 | 0.998 | 0.93 | 0.869 |
| Weak-signal beta-min, $n=1000$ | E-ACGL | 15.80 | 0.994 | 0.81 | 0.869 |
| High-dimensional, $n=300,d=500$ | E-CGL | 55.71 | 0.804 | 0.01 | 0.769 |
| High-dimensional, $n=300,d=500$ | E-ACGL | 36.71 | 0.889 | 0.00 | 0.767 |
| Moderately dense, $n=300$ | E-CGL | 88.40 | 0.792 | 0.00 | 0.553 |
| Moderately dense, $n=300$ | E-ACGL | 57.97 | 0.797 | 0.00 | 0.544 |
| Moderately dense, $n=1000$ | E-CGL | 87.84 | 0.890 | 0.00 | 0.674 |
| Moderately dense, $n=1000$ | E-ACGL | 67.47 | 0.913 | 0.00 | 0.679 |
| Strongly dense, $n=1000$ | E-CGL | 153.96 | 0.924 | 0.00 | 0.635 |
| Strongly dense, $n=1000$ | E-ACGL | 125.69 | 0.878 | 0.00 | 0.613 |

The weak-signal sparse cell remained stable. Exact recovery was rare or
absent in the $d=500$ and dense-support designs even when F1 remained
moderate to high. Adaptive weights reduced the selected support size and
increased F1 in the high-dimensional and moderately dense cells, but were
not uniformly beneficial: under the strongly dense design, E-ACGL had lower
F1 and ARI than E-CGL. These cells delimit the use of the method as a sparse
or compressible posterior-score support estimator.

Across the main simulation runs, the median recorded elapsed times per
repetition were 26.06 seconds for M-L, 45.86 for E-CGL, 41.20 for E-ACGL,
and 312.18 for M-CGL. Spherical $k$-means and the dense common- and
free-concentration vMF fits had medians of 0.18, 1.41, and 4.00 seconds,
respectively. These values compare the implementations and path budgets used
in this study rather than controlled algorithmic complexity. In particular,
M-CGL used a Riemannian ADMM implementation and fewer path points than
E-CGL. Detailed timing distributions and numerical stopping diagnostics are
reported in the Supplement.

## References

Banerjee, A., Dhillon, I.S., Ghosh, J., Sra, S., 2005. Clustering on the unit
hypersphere using von Mises-Fisher distributions. Journal of Machine Learning
Research 6, 1345-1382.

Gopal, S., Yang, Y., 2014. Von Mises-Fisher clustering models. In: Proceedings
of the 31st International Conference on Machine Learning, PMLR 32, pp.
154-162.

Guo, J., Levina, E., Michailidis, G., Zhu, J., 2010. Pairwise variable
selection for high-dimensional model-based clustering. Biometrics 66,
793-804. https://doi.org/10.1111/j.1541-0420.2009.01341.x.

Ng, T.L.J., 2023. Penalized maximum likelihood estimator for mixture of von
Mises-Fisher distributions. Metrika 86, 181-203.
https://doi.org/10.1007/s00184-022-00867-0.

Pan, W., Shen, X., 2007. Penalized model-based clustering with application to
variable selection. Journal of Machine Learning Research 8, 1145-1164.

Rossi, F., Barbaro, F., 2022. Mixture of von Mises-Fisher distribution with
sparse prototypes. Neurocomputing 501, 41-74.
https://doi.org/10.1016/j.neucom.2022.05.118.

Xie, B., Pan, W., Shen, X., 2008. Variable selection in penalized model-based
clustering via regularization on grouped parameters. Biometrics 64, 921-930.
https://doi.org/10.1111/j.1541-0420.2007.00955.x.

Yuan, M., Lin, Y., 2006. Model selection and estimation in regression with
grouped variables. Journal of the Royal Statistical Society: Series B 68,
49-67. https://doi.org/10.1111/j.1467-9868.2005.00532.x.

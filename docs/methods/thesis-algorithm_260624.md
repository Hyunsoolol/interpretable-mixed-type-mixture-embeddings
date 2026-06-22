# Eta-group Algorithm Note 260624

**Purpose.** This note states the proposed Eta-group fitting procedure in an algorithmic form, parallel in style to the algorithm descriptions used for sparse vMF mixture estimation. The estimator is a **proximal EM-type** procedure; it is not an exact EM algorithm.

## Algorithm 1. Eta-group Proximal EM-type Fit for a Fixed $\lambda_\eta$

**Input:** directional data $x_1,\ldots,x_n \in S^{d-1}$, number of components $K$, penalty parameter $\lambda_\eta$, maximum iteration $T_{\max}$, convergence tolerance $\epsilon$, minimum line-search step $s_{\min}$.

**Output:** fitted parameters $\hat{\Theta} _ {\lambda_\eta}=(\hat{\alpha}_k,\hat{\mu}_k,\hat{\kappa}_k)_{k=1}^K$, natural parameters $\hat{\eta}_k=\hat{\kappa}_k\hat{\mu}_k$, selected support $\hat{S}_{\lambda_\eta}$, convergence and objective-trace diagnostics.

1. Initialize $\Theta^{(0)}=(\alpha_k^{(0)},\mu_k^{(0)},\kappa_k^{(0)})_{k=1}^K$.
2. Set $\eta_k^{(0)}=\kappa_k^{(0)}\mu_k^{(0)}$.
3. For $t=0,1,\ldots,T_{\max}-1$, repeat Steps 4-17.
4. **E-step.** Compute the posterior responsibilities

$$ \tau_{ik}^{(t)} =
\frac{
\alpha_k^{(t)} C_d(\kappa_k^{(t)})
\exp\{(\eta_k^{(t)})^\top x_i\}
}{
\sum_{\ell=1}^K
\alpha_\ell^{(t)} C_d(\kappa_\ell^{(t)})
\exp\{(\eta_\ell^{(t)})^\top x_i\}
}.
$$

5. Compute sufficient statistics

$$ N_k^{(t)}=\sum_{i=1}^n \tau_{ik}^{(t)}, 
\qquad
r_k^{(t)}=\sum_{i=1}^n \tau_{ik}^{(t)}x_i.
$$

6. **Unpenalized vMF candidate.** Set

$$
\alpha_k^0=\frac{N_k^{(t)}}{n},
\qquad
\mu_k^0=\frac{r_k^{(t)}}{\|r_k^{(t)}\|_2}.
$$

7. Compute

$$
\rho_k^0=\frac{\|r_k^{(t)}\|_2}{N_k^{(t)}}.
$$

8. Approximate the concentration parameter by

$$
\kappa_k^0
\approx
\frac{d\rho_k^0-(\rho_k^0)^3}{1-(\rho_k^0)^2}.
$$

This is the standard vMF concentration approximation used to construct the M-step candidate. It is not an exact closed-form MLE.

9. Set the unpenalized natural parameter candidate

$$
\eta_k^0=\kappa_k^0\mu_k^0.
$$

10. **Centered eta construction.** For each coordinate $j=1,\ldots,d$, compute

$$
\bar{\eta}_j^0=\frac{1}{K}\sum_{k=1}^K \eta_{kj}^0,
\qquad
c_{kj}^0=\eta_{kj}^0-\bar{\eta}_j^0.
$$

11. **Group soft-thresholding.** Apply the Eta-group proximal step coordinatewise:

$$ c_{\cdot j}^{\mathrm{prox}} =
\left(
1-\frac{\lambda_\eta}{\|c_{\cdot j}^0\|_2}
\right)_+
c_{\cdot j}^0.
$$

12. Reconstruct the proximal eta candidate by preserving the coordinatewise mean component:

$$ \eta_{kj}^{\mathrm{prox}} =
\bar{\eta}_j^0+c_{kj}^{\mathrm{prox}}.
$$

13. Restore vMF parameters:

$$ \kappa_k^{\mathrm{prox}} = \|\eta_k^{\mathrm{prox}}\|_2, \qquad \mu_k^{\mathrm{prox}} =
\frac{\eta_k^{\mathrm{prox}}}{\|\eta_k^{\mathrm{prox}}\|_2}.
$$

Small-norm cases are handled by numerical safeguards in the implementation.

14. **Monotone line-search safeguard.** Set $s=1$ and define

$$ \eta_k^{\mathrm{cand}}(s)=
\eta_k^{(t)}
+
s\{\eta_k^{\mathrm{prox}}-\eta_k^{(t)}\}.
$$

15. Restore $\kappa_k^{\mathrm{cand}}(s)=\|\eta_k^{\mathrm{cand}}(s)\|_2$ and $\mu_k^{\mathrm{cand}}(s)=\eta_k^{\mathrm{cand}}(s)/\|\eta_k^{\mathrm{cand}}(s)\|_2$.
16. If the penalized objective decreases, replace $s$ by $s/2$ and repeat Steps 14-15 until the objective no longer decreases or $s<s_{\min}$.
17. Accept the line-search candidate as $\Theta^{(t+1)}$.
18. Stop if

$$
\left| \mathcal{L}_p(\Theta^{(t+1)}) - \mathcal{L}_p(\Theta^{(t)}) \right| < \epsilon.
$$

19. Define the selected support as
$$
\hat{S_}{\lambda_\eta}=\left\{j:\|c_{\cdot j}(\hat{\eta})\|_2>0\right\}.
$$

**Remark.** Algorithm 1 is a proximal EM-type update. The E-step is the standard mixture E-step, but the M-step is not the exact maximizer of the penalized objective. The line-search step is an implementation-level safeguard against objective decrease, not a proof of global convergence.

## Algorithm 2. Eta-group Path+BIC Tuning and Post-selection Refit

**Input:** directional data $x_1,\ldots,x_n$, number of components $K$, lambda path $\Lambda=\{\lambda_1,\ldots,\lambda_L\}$, maximum iteration, convergence tolerance, number of random starts.

**Output:** selected penalty $\hat{\lambda}_\eta$, selected support $\hat{S}$, penalized Eta-group fit, and optional post-selection refit.

1. Construct a decreasing penalty path $\Lambda=\{\lambda_1,\lambda_2,\ldots,\lambda_L\}$.
2. For each $\lambda_\ell\in\Lambda$, run Algorithm 1.
3. Store the fitted parameter $\hat{\Theta}_{\lambda_\ell}$ and selected support

$$
\hat{S}_{\lambda_\ell} =
\left\{
j:
\|c_{\cdot j}(\hat{\eta}_{\lambda_\ell})\|_2>0
\right\}.
$$

4. Let $m_{\lambda_\ell}=|\hat{S}_{\lambda_\ell}|$.
5. Compute the implementation-level degrees of freedom approximation

$$
df_{\lambda_\ell} =
(K-1)+d+(K-1)m_{\lambda_\ell}.
$$

6. Compute

$$
\mathrm{BIC}(\lambda_\ell) =
-2\ell(\hat{\Theta}_{\lambda_\ell})
+
\log(n)\,df_{\lambda_\ell}.
$$

7. Select

$$
\hat{\lambda}_\eta =
\arg\min_{\lambda_\ell\in\Lambda}
\mathrm{BIC}(\lambda_\ell).
$$

8. Set $\hat{S}=\hat{S}_{\hat{\lambda}_\eta}$.
9. **Post-selection refit.**
10. If $\hat{S}=\emptyset$, do not refit and record `zero_active_support`.
11. If $\hat{S}\neq\emptyset$, fix $\hat{S}$ and re-estimate the vMF mixture parameters without the Eta-group penalty, keeping coordinates outside $\hat{S}$ inactive.
12. Return the penalized fit, selected support, BIC-selected $\hat{\lambda}_\eta$, and refit estimator when available.

**Remark.** The BIC degrees of freedom are an implementation-level approximation, not a formal effective degrees of freedom result. Refit is a post-selection bias correction step and does not reselect variables. Positive-support tuning, adaptive refinement, stability selection, and long-path runs are diagnostic or sensitivity analyses, not the current official algorithm.

## K=2 Eta Contrast Variant

For $K=2$, Algorithm 1 replaces the centered group step with coordinatewise eta contrast soft-thresholding. Define

$$
\delta^0=\eta_2^0-\eta_1^0.
$$

Then apply

$$
\delta_j^{\mathrm{prox}} =
\left(
1-\frac{\lambda_\eta}{|\delta_j^0|}
\right)_+
\delta_j^0.
$$

The eta vectors are reconstructed by preserving the coordinatewise mean

$$
\bar{\eta}_j^0=\frac{\eta_{1j}^0+\eta_{2j}^0}{2},
$$

$$
\eta_{1j}^{\mathrm{prox}} = \bar{\eta}_j^0-\frac{1}{2}\delta_j^{\mathrm{prox}}, \qquad \eta_{2j}^{\mathrm{prox}} =
\bar{\eta}_j^0+\frac{1}{2}\delta_j^{\mathrm{prox}}.
$$

For $K>2$, the centered eta group lasso in Algorithm 1 is used instead.


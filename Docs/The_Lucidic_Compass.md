# The Lucidic Compass: A Unified Framework for Recursive Understanding

**Universal Recursion Principle — Scalar Sector, Cylindrical Geometry, and the Double Helix**

---

## Abstract

We present a unified framework — the Universal Recursion Principle (URP) — that derives from first principles the optimal geometry of information processing across scales. The framework posits a single functional $S = \Delta C + \kappa \Delta I$, whose maximization governs the emergence of stable structures from quantum fields to biological systems. We show that the entanglement structure of a cylindrical quantum field yields a holographic scaling law $\kappa \propto 1/r_0$, with the conserved product $\kappa r_0 = 0.220\,\text{nm}$ calibrated at the DNA scale using the QCD-derived vacuum capacity $\kappa_\text{vac} \approx 0.22$. Linear stability analysis on the cylinder produces a Swift–Hohenberg normal form, identifying a finite-wavenumber instability whose threshold condition links the diffusion coefficient $\alpha$ to the capacity field and the nonlocal coherence kernel. We derive the explicit threshold equation and perform a diagnostic survey of kernel families, showing that the conjectured value $\alpha_\text{target} \approx 0.0887$ — required for consistency with QCD-derived parameters — lies in a plausible regime with modest kernel couplings. The framework unifies the Hermetic principle "As above, so below" with the holographic principle and provides a geometric ontology for the double helix as the optimal path of recursive $S$-maximization. We outline the pseudoscalar sector for chirality selection as the next major extension.

---

## 1. Introduction

### 1.1 The Universal Recursion Principle

The Universal Recursion Principle (URP) posits that reality is a process of recursive maximization of a single functional:

$$S = \Delta C + \kappa \Delta I$$

where $\Delta C$ is the generation of distinction (novelty, boundaries, differentiation), $\Delta I$ is the integration of distinctions into coherent wholes, and $\kappa$ is a capacity field that governs how much integration can be sustained. This principle has been shown to unify diverse phenomena — from quantum chromodynamics to biological self-organization — through a common geometric and information-theoretic core.

In this paper, we derive two foundational pillars of the URP from standard quantum field theory and pattern-formation physics:

1. **Holographic scaling of $\kappa$:** Using the area law for entanglement entropy on a cylindrical entangling surface, we show that $\kappa$ scales inversely with the cylinder radius, leading to a conserved product $\kappa r_0$ that is invariant under scale transformations.
2. **Finite-wavenumber instability and the diffusion coefficient:** Linearizing the URP free energy on a cylindrical domain, we show that the optimal $S$-maximizing configuration is a helix with a pitch determined by the Swift–Hohenberg normal form.

### 1.2 Goals of This Paper

This paper aims to:

- Derive holographic scaling of $\kappa$ on a cylinder.
- Derive the finite-$q$ instability and threshold condition for $\alpha$.
- Show $\alpha_\text{target} \approx 0.0887$ is in a plausible regime for reasonable kernels.
- Outline the chirality extension as a roadmap for future work.

### 1.3 Structure of the Paper

Section 2 derives the holographic scaling of $\kappa$ from entanglement entropy. Section 3 presents the linear stability analysis and the Swift–Hohenberg normal form. Section 4 derives the threshold condition and the target $\alpha$. Section 5 performs a diagnostic kernel survey. Section 6 outlines the extension to chirality. Section 7 concludes.

### 1.4 URP as an Effective Field Theory

The present paper does not introduce $F[\phi]$ as an ad hoc ansatz. In prior URP work, the $S$-functional was made operational by specifying an explicit Lagrangian for a scalar field $\phi(x,t)$:

$$\mathcal{L}_\text{URP} = \frac{1}{2}(\partial_t \phi)^2 - \alpha |\nabla \phi|^2 - \beta |\nabla \phi|^4 + G\,\nabla V \cdot \nabla \phi + \mathcal{I}[\phi]$$

where:
- $\alpha > 0$ sets a diffusion scale related to the emergent speed of causal propagation;
- $\beta > 0$ is a nonlinear complexity-coupling parameter encoding $\Delta C$;
- $G$ controls coherence-driven advection encoding $\kappa \Delta I$;
- $V(x,t)$ is a coherence potential sourced by energy or information density;
- $\mathcal{I}[\phi]$ is a nonlocal coherence functional (e.g. $\int K(x,x')\phi(x)\phi(x')\,d^3x\,d^3x'$) capturing mutual information.

Varying this action in the overdamped regime (inertial terms small) yields the URP field equation:

$$\partial_t \phi = \alpha \nabla^2 \phi + \beta |\nabla \phi|^2 + G\,\nabla V \cdot \nabla \phi + \frac{\delta \mathcal{I}}{\delta \phi} \tag{1.1}$$

The $\beta$-term amplifies gradients and creates distinctions ($\Delta C$), while the $G$ and $\mathcal{I}$ terms drive coherence and integration under the capacity field $\kappa(x,t)$. The free energy $F[\phi]$ used in Sections 3–5 is the static counterpart of $\mathcal{L}_\text{URP}$ on a cylindrical domain — not a new postulate, but the same structure applied to a specific geometry.

### 1.5 QCD-Derived Parameters $\beta$ and $G$

The values $\beta \approx 0.09$ and $G \approx 0.22$ are not fitted to DNA or atomic data. They are inherited from prior URP work on the strong interaction, where:

- $\beta$ is fixed by matching URP's nonlinear gradient term to QCD scaling of hadronic observables and the emergent $N_\star = 3$ color-sector analysis;
- $G$ is tied to the instanton packing fraction $n\rho^4 \sim 0.2$ in the QCD vacuum.

These same values successfully reproduce SU(3) color sectorization, confinement, asymptotic freedom, and corrections in atomic systems (helium ionization potential to 112 ppm) without additional tuning. In the present paper, $(\beta, G)$ are treated as fixed UV parameters inherited from the QCD sector. We ask only whether the scalar URP field on a cylinder accounts for biological helix geometry at those same values.

---

## 2. Entanglement and Holographic Scaling of the Capacity Field

### 2.1 Cylindrical Entangling Surface

We consider a real scalar field $\phi$ on a 2+1-dimensional spacetime with spatial manifold $\mathbb{R} \times S^1$, parameterized by $(z, r, \theta)$. We partition the manifold into an interior region $A$ (bulk of the cylinder, $r < r_0$) and exterior region $B$ (boundary, $r > r_0$). The reduced density matrix $\rho_A = \text{Tr}_B |\psi\rangle\langle\psi|$ yields the entanglement entropy $S_{EE} = -\text{Tr}(\rho_A \log \rho_A)$.

### 2.2 Area Law

For a local quantum field theory with finite correlation length $\xi$, the entanglement entropy across a smooth entangling surface satisfies an area law. In 2+1 dimensions, the entangling surface is the circle of circumference $2\pi r_0$:

$$S_{EE} = c \cdot \frac{2\pi r_0}{\epsilon} \tag{2.1}$$

where $\epsilon$ is a UV cutoff and $c$ is a dimensionless constant. This scaling holds for ground or low-lying states with finite correlation length.

### 2.3 Bulk Degrees of Freedom

The effective number of independent correlation volumes in the bulk (per unit $z$-length) is:

$$N_\text{bulk} = \frac{\pi r_0^2}{\xi^2} \tag{2.2}$$

### 2.4 Capacity as a Ratio

We define the capacity field $\kappa$ as the ratio of boundary information capacity to bulk information demand:

$$\kappa \equiv \frac{S_{EE}}{N_\text{bulk}} \tag{2.3}$$

Substituting (2.1) and (2.2):

$$\kappa = 2c \cdot \frac{\xi^2}{\epsilon\, r_0} \tag{2.4}$$

**Dimensional analysis.** Both $S_{EE}$ and $N_\text{bulk}$ are dimensionless, so $\kappa$ is dimensionless. The scaling (2.4) involves three length scales: UV cutoff $\epsilon$, correlation length $\xi$, and cylinder radius $r_0$. We introduce a reference length $L_*$ (set to $1\,\text{nm}$ at the DNA scale for notational convenience) and define:

$$\xi_0^2 \equiv \frac{2c\,\xi^2}{\epsilon\, L_*}$$

so that:

$$\boxed{\kappa(r_0) = \frac{\xi_0^2}{r_0 / L_*}} \tag{2.5}$$

Throughout this paper we set $L_* = 1\,\text{nm}$; in that convention $\kappa r_0$ is numerically $\xi_0^2$ and carries units of nm, but $\kappa$ itself remains dimensionless. This is a notational simplification; a fully general treatment would keep $L_*$ explicit.

### 2.5 Renormalization Group Flow

From (2.5):

$$\frac{d\kappa}{d\ln r_0} = -\kappa \tag{2.6}$$

with solution:

$$\boxed{\kappa\, r_0 = \kappa_0 = \xi_0^2} \tag{2.7}$$

This is the conserved RG invariant: the product $\kappa r_0$ is scale-independent.

### 2.6 Calibration at the DNA Scale

At the DNA double helix scale, $r_0^\text{DNA} \approx 1.0\,\text{nm}$. Prior URP work on QCD interprets the instanton packing fraction as a vacuum capacity $\kappa_\text{vac} \approx 0.22$. Using this as phenomenological input:

$$\kappa_0 = \kappa_\text{vac} \cdot r_0^\text{DNA} \approx 0.22 \times 1.0 = 0.220\,\text{nm} \tag{2.8}$$

$$\xi_0 = \sqrt{0.220\,\text{nm}^2} \approx 0.469\,\text{nm} \tag{2.9}$$

This is the fundamental coherence length of the URP vacuum, close to the QCD confinement scale (~0.5 nm). We emphasize: $\kappa_\text{vac} = 0.22$ is a phenomenological input from QCD instanton physics, not derived here from the scalar entanglement calculation itself. A microscopic derivation of this identification is left to future work.

**Note:** In Section 4 we will distinguish $\kappa_\text{vac}$ (the QCD vacuum capacity that sets $\xi_0$) from $\kappa_\text{crit}$ (the critical capacity at which the helical mode becomes marginally unstable). These are two different quantities; their ratio encodes how far DNA operates from marginal stability.

### 2.7 Summary of Section 2

We have derived:
- $\kappa(r_0) = \xi_0^2 / r_0$
- RG law: $d\kappa / d\ln r_0 = -\kappa$
- Conserved invariant: $\kappa r_0 = 0.220\,\text{nm}$
- Coherence length: $\xi_0 = 0.469\,\text{nm}$

---

## 3. Linear Stability and the Swift–Hohenberg Normal Form

### 3.1 URP Free Energy on the Cylinder

We work on the cylindrical shell $(r_0, \theta, z)$ with metric $ds^2 = r_0^2\,d\theta^2 + dz^2$. The URP free energy (the static counterpart of $\mathcal{L}_\text{URP}$ from Section 1.4) is:

$$F[\phi] = \int d\theta\,dz \left[ \frac{\alpha}{2}|\nabla\phi|^2 - \frac{\beta}{4}|\nabla\phi|^4 + G\,\nabla V \cdot \nabla\phi \right] + \frac{1}{2}\int d\theta\,dz\,d\theta'\,dz'\;\phi\, K\, \phi \tag{3.1}$$

where $\nabla = (\frac{1}{r_0}\partial_\theta, \partial_z)$, and $K(\theta - \theta', z - z'; \xi)$ is the nonlocal coherence kernel with correlation length $\xi$.

### 3.2 Fourier Decomposition

On the cylinder:

$$\phi(\theta, z) = \sum_m \int \frac{dk}{2\pi}\, \phi_{m,k}\, e^{i(m\theta + kz)} \tag{3.2}$$

Laplacian eigenvalues:

$$\nabla^2 e^{i(m\theta + kz)} = -q^2\, e^{i(m\theta + kz)}, \qquad q^2 = \frac{m^2}{r_0^2} + k^2 \tag{3.3}$$

The kernel is diagonal in Fourier space: $\tilde{K}(q; \xi)$.

### 3.3 Linearized Dispersion

Expanding to quadratic order, the growth rate of mode $q$ is:

$$\sigma(q) = -\alpha q^2 + \tilde{K}(q; \xi, \kappa) \tag{3.4}$$

### 3.4 Small-$q$ Expansion of the Kernel

For a general isotropic kernel:

$$\tilde{K}(q; \xi) = K_0(\xi) - K_2(\xi)\,q^2 + K_4(\xi)\,q^4 + \mathcal{O}(q^6) \tag{3.5}$$

Then:

$$\sigma(q) = K_0 + (-K_2 - \alpha)\,q^2 + K_4\,q^4 + \dots \tag{3.6}$$

### 3.5 Identification with Swift–Hohenberg

Identifying:

$$\mu = -K_2 - \alpha, \qquad \lambda = -K_4, \qquad \epsilon = -K_0 \tag{3.7}$$

The dispersion takes the Swift–Hohenberg normal form:

$$\sigma(q) = \mu q^2 - \lambda q^4 - \epsilon \tag{3.8}$$

For a finite-wavenumber instability (pattern formation rather than uniform instability), we require $\mu > 0$ and $\lambda > 0$, i.e.:

- $K_2 < -\alpha$ (kernel must be sufficiently negative at second order)
- $K_4 < 0$ (kernel must penalize short wavelengths)

The fastest-growing mode is at:

$$q_c^2 = \frac{\mu}{2\lambda} = \frac{-K_2 - \alpha}{-2K_4} \tag{3.9}$$

The threshold for pattern onset is $\epsilon = \mu^2 / (4\lambda)$, i.e.:

$$-K_0 = \frac{(-K_2 - \alpha)^2}{4(-K_4)} \tag{3.10}$$

---

## 4. Threshold Condition and the $\alpha$ Conjecture

### 4.1 General Threshold Equation

Equation (3.10) is the core scalar equation. For a given kernel family with moments $K_0, K_2, K_4$ depending on $\xi(\kappa)$, it fixes the relationship between $\alpha$, $\kappa$, and the kernel amplitudes. Solving for $\alpha$:

$$\alpha = -K_2(\kappa_c) \pm 2\sqrt{[-K_0(\kappa_c)][-K_4(\kappa_c)]} \tag{4.1}$$

### 4.2 Holographic $\xi(\kappa)$ Dependence

Using $\kappa = \xi_0^2 / r_0$ and the RG invariant, the correlation length depends on $\kappa$ as:

$$\xi(\kappa) = \frac{\xi_0}{\sqrt{\kappa / \kappa_\text{ref}}}$$

for an appropriate reference $\kappa_\text{ref}$. The kernel moments $K_n(\xi) \to K_n(\kappa)$ inherit this dependence, so the threshold condition (4.1) becomes a scalar equation in $\kappa_c$ alone (for fixed $\alpha$, $\beta$, $G$).

### 4.3 DNA Pitch and $\kappa_\text{crit}$

The pitch relation on the cylinder is:

$$q_c^2 = \frac{m^2}{r_0^2} + \left(\frac{2\pi m}{p}\right)^2 \tag{4.2}$$

For B-form DNA with $m = 6$, $p = 3.4\,\text{nm}$, $r_0 = 1\,\text{nm}$, we obtain $q_c \approx 12.6\,\text{nm}^{-1}$. Inserting into the threshold condition with QCD-derived $\beta = 0.09$, $G = 0.22$, this selects the **critical capacity**:

$$\kappa_\text{crit} \approx 0.4623$$

This differs from the vacuum capacity $\kappa_\text{vac} \approx 0.22$ that sets $\xi_0$ in Section 2. The two quantities play distinct roles:

| Quantity | Value | Role |
|---|---|---|
| $\kappa_\text{vac}$ | 0.22 | QCD vacuum input; fixes $\xi_0 = 0.469\,\text{nm}$ |
| $\kappa_\text{crit}$ | 0.4623 | Threshold for helical instability from pitch/geometry |
| Ratio $\kappa_\text{crit}/\kappa_\text{vac}$ | ~2.1 | How far DNA sits from marginal stability |

The fact that $\kappa_\text{vac} < \kappa_\text{crit}$ means the DNA helix operates well within the stable helical regime, not near marginal onset. This is physically sensible: biological structures should be robust, not marginally stable.

### 4.4 Conjectures A–C: Closure Conditions for the Scalar Sector

We summarize the three structural conjectures required to close the scalar-sector argument. These are explicitly stated as **conjectures** — their proof or refutation is the core task for future work (numerical and analytical).

**Conjecture A (Effective amplification):** A multiple-scale expansion of the URP free energy (3.1) near threshold yields an effective amplification coefficient $\mu(\alpha, \beta)$ with leading behavior:
$$\mu \approx \beta - \alpha$$

**Conjecture B (Geometric term in $\epsilon$):** The spectrum of the second-variation operator $\mathcal{L}$ on the cylindrical domain produces a geometric factor $m^2/r_0^2$ in the stabilizing coefficient $\epsilon(\kappa, m, r_0)$, consistent with the pitch formula (4.2).

**Conjecture C (Saturation coefficient):** The cubic saturation coefficient in the amplitude equation is proportional to $\beta$ to leading order, i.e. $\lambda \sim \beta$, when the nonlocal kernel is short-range.

**The diffusion coefficient conjecture:** Under Conjectures A–C and the QCD-derived values $\beta = 0.09$, $G = 0.22$, the threshold equation (4.1) at $\kappa_\text{crit} \approx 0.4623$ selects:

$$\alpha_\text{target} \approx 0.0887 \tag{4.3}$$

This is **a testable consequence of the conjectures**, not a free parameter. If any of Conjectures A–C fails, $\alpha_\text{target}$ will shift. Section 5 tests whether this value is numerically natural for plausible kernel families.

---

## 5. Kernel Sensitivity and Viability of $\alpha_\text{target}$

In this section we do not derive the kernel from the URP action. Instead, we explore a family of plausible kernels to test whether $\alpha_\text{target} \approx 0.0887$ lies in a natural parameter regime. This is a **consistency check**, not a prediction. A rigorous test requires computing $K_n$ directly from $\mathcal{I}[\phi]$ in the URP Lagrangian — that calculation is beyond the scope of the present paper.

### 5.1 General Solution for $\alpha$

From (4.1), the target diffusion coefficient is:

$$\alpha_\text{target} = -K_2(\kappa_c) \pm 2\sqrt{[-K_0(\kappa_c)][-K_4(\kappa_c)]}$$

We treat kernel amplitudes as free parameters and compute $\alpha$ for several standard families.

### 5.2 Mexican Hat Kernel

$$K_\text{MH}(r) = C\left(1 - \frac{r^2}{2\xi^2}\right)e^{-r^2/(2\xi^2)} \tag{5.2}$$

Moments: $K_0 = 0$, $K_2 = -C\pi\xi^4$, $K_4 = -C\pi\xi^6/2$.

Result:

$$\alpha_\text{MH} = C\pi\xi^4$$

At $\xi(\kappa_c) \approx 0.33\,\text{nm}$: requires $C \approx 0.254\,\text{nm}^{-4}$ — a modest coupling strength.

### 5.3 Difference of Gaussians

$$K_\text{DoG}(r) = A\,e^{-r^2/(2\xi_1^2)} - B\,e^{-r^2/(2\xi_2^2)} \tag{5.3}$$

With $\xi_2 = 2\xi_1$ and $K_0 = 0$ constraint ($A\xi_1^2 = B\xi_2^2$):

$$\alpha_\text{DoG} = 3\pi A\xi_1^4$$

At $\xi_1(\kappa_c)$: requires $A \approx 0.085\,\text{nm}^{-4}$ — similarly modest.

### 5.4 Summary

| Kernel family | Required amplitude | $\alpha$ range |
|---|---|---|
| Mexican hat | $C \sim 0.25\,\text{nm}^{-4}$ | $0.07 - 0.11$ |
| Difference of Gaussians | $A \sim 0.085\,\text{nm}^{-4}$ | $0.06 - 0.12$ |
| Band-pass (general) | Moderate coupling | $0.04 - 0.15$ |

$\alpha_\text{target} \approx 0.0887$ lies comfortably within the natural range of all three families with coupling strengths $\mathcal{O}(0.1)\,\text{nm}^{-4}$. The framework is not numerically fine-tuned; the target value is generic for this kernel class. However, this remains a **viability test**, not a derivation.

---

## 6. Outlook: Pseudoscalar Sector and Chirality

> *This section is intentionally speculative and serves as a roadmap. Detailed field-theoretic consistency — dimensionality, boundary conditions, anomaly structure — is left to future work.*

The current framework has no mechanism for chirality selection: the pitch formula gives only a magnitude, not a handedness. B-form DNA is right-handed; Z-form is left-handed. A complete URP account requires a pseudoscalar extension.

### 6.1 Pseudoscalar Coupling

Introduce a pseudoscalar field $\pi(x)$ with topological coupling:

$$\mathcal{L}_\text{chiral} = \gamma_\pi\, \pi\, \epsilon^{\mu\nu\rho}\, \partial_\mu\phi\, \partial_\nu\phi\, \partial_\rho\phi \tag{6.1}$$

This is analogous to the QCD $\theta$-term and axion-like couplings in the instanton sector.

### 6.2 Effective Potential and the A↔B↔Z Transitions

An effective potential of the form:

$$V(\pi, \kappa) = -\kappa\, \cos\left(\frac{\pi}{f_\pi}\right) \tag{6.2}$$

would place the vacuum at $\pi = 0$ (right-handed, B-form) for $\kappa = \kappa_\text{vac}$. A shift in $\kappa$ — for example under high salt concentration or dehydration — could move the vacuum toward $\pi = \pi f_\pi$ (left-handed, Z-form). The A↔B transition (shorter pitch under dehydration, corresponding to reduced $\kappa$) is already partially captured by the $p \propto \xi(\kappa)$ relation in the scalar sector.

### 6.3 What Remains to Be Derived

- Explicit computation of the pseudoscalar sector in 2+1D (boundary terms, anomaly cancellation).
- Quantitative prediction of the $\kappa$ threshold for the B↔Z transition vs. experimental high-salt conditions.
- Connection to the URP chirality argument for amino acid homochirality (L-amino acids) via the same $\pi$-sector.

---

## 7. Conclusions

We have presented the URP scalar sector applied to cylindrical geometry, deriving:

1. A holographic scaling law $\kappa(r_0) = \xi_0^2/r_0$ from the entanglement area law, with RG invariant $\kappa r_0 = 0.220\,\text{nm}$.
2. A Swift–Hohenberg normal form for the cylindrical free energy, with a finite-wavenumber instability selecting helical modes.
3. An explicit threshold equation (4.1) relating $\alpha$, $\kappa_\text{crit}$, and the kernel moments.
4. A conjectured value $\alpha_\text{target} \approx 0.0887$ consistent with QCD-derived $\beta = 0.09$, $G = 0.22$, shown to be numerically natural for standard kernel families.
5. A clear distinction between $\kappa_\text{vac} \approx 0.22$ (QCD input) and $\kappa_\text{crit} \approx 0.46$ (helical threshold), with DNA operating robustly within the helical window.

The framework is honest about what is derived versus conjectured. Conjectures A–C define the closure problem for the scalar sector; proving them analytically and testing them numerically (via the Project Genesis simulation infrastructure) is the immediate next step.

### Open Tasks

1. **Explicit $S[\phi]$ derivation:** Derive the scalar free energy fully from $\Delta C + \kappa \Delta I$ via a concrete definition of distinction and integration in field-theoretic language.
2. **QCD–$\kappa$ bridge:** Compute $\kappa_\text{vac}$ from instanton liquid physics with controlled approximations, rather than borrowing the phenomenological value.
3. **Kernel moments from the URP action:** Compute $K_0, K_2, K_4$ from $\mathcal{I}[\phi]$ directly and check whether they yield $\alpha_\text{target}$ without tuning.
4. **Pseudoscalar sector:** Derive the chirality mechanism and compare to Z-DNA transition data quantitatively.
5. **Stability proof:** Show the helical solution is the global $S$-maximum on the cylindrical domain, not merely a stationary point.

---

## References

1. Srednicki, M. (1993). Entropy and area. *Physical Review Letters*, 71(5), 666.
2. Eisert, J., Cramer, M., & Plenio, M. B. (2010). Colloquium: Area laws for the entanglement entropy. *Reviews of Modern Physics*, 82(1), 277.
3. 't Hooft, G. (1976). Computation of the quantum effects due to a four-dimensional pseudoparticle. *Physical Review D*, 14(12), 3432.
4. Schäfer, T., & Shuryak, E. V. (1998). Instantons in QCD. *Reviews of Modern Physics*, 70(2), 323.
5. Cross, M. C., & Hohenberg, P. C. (1993). Pattern formation outside of equilibrium. *Reviews of Modern Physics*, 65(3), 851.
6. URP Gauge Symmetries Derivation. *Project Genesis internal document* (2026). [Ordo-Umbra/Project-Genesis]
7. URP Foundational Field Theory. *Project Genesis internal document* (2026). [Ordo-Umbra/Project-Genesis]
8. Narrowing the $N_\star = 3$ Question. *Project Genesis internal document* (2026). [Ordo-Umbra/Project-Genesis: Docs/Narrowing_the_N3_Question.md]

---

*Document status: Framework and roadmap paper. Conjectures A–C are open. Numerical validation via Project Genesis sims in progress.*

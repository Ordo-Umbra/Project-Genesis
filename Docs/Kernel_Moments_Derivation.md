# Kernel Moments Derivation
## URP Scalar Sector — Nonlocal Coherence Functional on the Cylinder

*Status: Working note. These derivations feed directly into `The_Lucidic_Compass.md` Sections 3–5 and are a prerequisite for Conjectures A–C.*

---

## 1. Setup: Coherence Functional and Kernel

The nonlocal coherence term in the URP free energy (static form on the cylindrical shell) is:

$$\mathcal{I}[\phi] = \frac{1}{2}\int d\theta\,dz\,d\theta'\,dz'\;\phi(\theta,z)\,K(\theta-\theta',z-z';\xi)\,\phi(\theta',z')$$

where $K$ is translationally invariant and isotropic in the $(\theta,z)$-plane, written as $K(r;\xi)$ with $r = |(\Delta\theta, \Delta z)|$.

In Fourier space (using the cylindrical decomposition from `The_Lucidic_Compass.md` Section 3.2):

$$\mathcal{I}[\phi] = \frac{1}{2}\sum_m \int \frac{dk}{2\pi}\,\tilde{K}(q;\xi)\,|\phi_{m,k}|^2$$

with $q^2 = m^2/r_0^2 + k^2$ and:

$$\tilde{K}(q;\xi) = \int d^2r\,K(r;\xi)\,e^{i\mathbf{q}\cdot\mathbf{r}} = 2\pi\int_0^\infty r\,dr\,K(r;\xi)\,J_0(qr)$$

---

## 2. General Moment Formulas

Expanding $J_0(qr)$ for small $q$:

$$J_0(qr) = 1 - \frac{(qr)^2}{4} + \frac{(qr)^4}{64} + \mathcal{O}(q^6)$$

gives the **general moment formulas**:

$$\boxed{K_0(\xi) = 2\pi \int_0^\infty r\,K(r;\xi)\,dr}$$

$$\boxed{K_2(\xi) = \frac{\pi}{2} \int_0^\infty r^3\,K(r;\xi)\,dr}$$

$$\boxed{K_4(\xi) = \frac{\pi}{32} \int_0^\infty r^5\,K(r;\xi)\,dr}$$

so that the small-$q$ expansion is:

$$\tilde{K}(q;\xi) = K_0(\xi) - K_2(\xi)\,q^2 + K_4(\xi)\,q^4 + \mathcal{O}(q^6)$$

These feed directly into the Swift–Hohenberg coefficients (Lucidic Compass, Section 3.5):

$$\mu = -K_2 - \alpha,\qquad \lambda = -K_4,\qquad \epsilon = -K_0$$

**Conditions for finite-wavenumber instability (pattern formation):**
- $K_2 < -\alpha$ (i.e. $\mu > 0$): kernel must have sufficiently negative second moment
- $K_4 < 0$ (i.e. $\lambda > 0$): kernel penalizes short wavelengths

---

## 3. Key Radial Integrals

These Gaussian radial integrals are needed below:

| Integral | Result |
|---|---|
| $\int_0^\infty r\,e^{-r^2/(2\xi^2)}\,dr$ | $\xi^2$ |
| $\int_0^\infty r^3 e^{-r^2/(2\xi^2)}\,dr$ | $2\xi^4$ |
| $\int_0^\infty r^5 e^{-r^2/(2\xi^2)}\,dr$ | $8\xi^6$ |
| $\int_0^\infty r^7 e^{-r^2/(2\xi^2)}\,dr$ | $48\xi^8$ |

General formula: $\int_0^\infty r^{2n+1} e^{-r^2/(2\xi^2)}\,dr = n!\,(2\xi^2)^{n+1}/2 = n!\cdot 2^n \xi^{2n+2}$

---

## 4. Explicit Moments by Kernel Family

### 4.1 Pure Gaussian

$$K_\text{G}(r;\xi) = A\,e^{-r^2/(2\xi^2)}$$

| Moment | Value |
|---|---|
| $K_0$ | $2\pi A\xi^2$ |
| $K_2$ | $\pi A\xi^4$ |
| $K_4$ | $\frac{\pi A}{8}\xi^6$ |

Full expansion:

$$\tilde{K}_\text{G}(q;\xi) = 2\pi A\xi^2 - \pi A\xi^4\,q^2 + \frac{\pi A}{8}\xi^6\,q^4 + \dots$$

Note: All moments positive → $K_2 > 0$, so $\mu = -K_2 - \alpha < 0$ for all $\alpha > 0$. **A pure Gaussian kernel alone cannot produce a Swift–Hohenberg instability.** It provides the background coherence but requires modification (subtraction term) to select a finite wavenumber.

---

### 4.2 Mexican Hat

$$K_\text{MH}(r;\xi) = C\left(1 - \frac{r^2}{2\xi^2}\right)e^{-r^2/(2\xi^2)}$$

This is the natural URP coherence kernel: rewards short-range correlation, penalizes medium-range redundancy.

Derivation (writing $K_\text{MH} = C\,e^{-r^2/(2\xi^2)} - \frac{C}{2\xi^2}\,r^2\,e^{-r^2/(2\xi^2)}$):

$$K_0^\text{MH} = 2\pi C\xi^2 - \frac{C}{2\xi^2}\cdot 2\pi\cdot 2\xi^4 = 2\pi C\xi^2 - 2\pi C\xi^2 = 0$$

$$K_2^\text{MH} = \pi C \cdot 2\xi^4 \cdot \frac{1}{2} - \frac{C}{2\xi^2}\cdot\pi\cdot 8\xi^6\cdot\frac{1}{2} = \pi C\xi^4 - 2\pi C\xi^4 = -\pi C\xi^4$$

$$K_4^\text{MH} = \frac{\pi C}{8}\xi^6 - \frac{C}{2\xi^2}\cdot\frac{\pi}{8}\cdot 48\xi^8\cdot\frac{1}{1} = \frac{\pi C\xi^6}{8} - \frac{6\pi C\xi^6}{8} = -\frac{3\pi C\xi^6}{4}$$

**Summary:**

| Moment | Value |
|---|---|
| $K_0$ | $0$ |
| $K_2$ | $-\pi C\xi^4$ |
| $K_4$ | $-\frac{3\pi C}{4}\xi^6$ |

Swift–Hohenberg coefficients from Mexican hat:

$$\mu = \pi C\xi^4 - \alpha,\qquad \lambda = \frac{3\pi C}{4}\xi^6,\qquad \epsilon = 0$$

The $\epsilon = 0$ condition means the Mexican hat sits exactly at the onset of pattern formation. To produce a genuine threshold (DNA-scale $\kappa_\text{crit}$), a small positive $\epsilon$ must come from either a correction term in the kernel or from the $G\nabla V \cdot \nabla\phi$ coherence potential coupling. This is a meaningful constraint.

Fastest-growing wavenumber:

$$q_c^2 = \frac{\mu}{2\lambda} = \frac{\pi C\xi^4 - \alpha}{\frac{3\pi C}{2}\xi^6} = \frac{1}{\xi^2}\cdot\frac{1 - \alpha/(\pi C\xi^4)}{3/2}$$

For $\alpha \ll \pi C\xi^4$: $q_c \approx \sqrt{2/3}/\xi$, i.e. the selected wavenumber is set by the coherence length.

---

### 4.3 Difference of Gaussians (DoG)

$$K_\text{DoG}(r;\xi_1,\xi_2) = A\,e^{-r^2/(2\xi_1^2)} - B\,e^{-r^2/(2\xi_2^2)},\qquad \xi_2 > \xi_1$$

General moments (by linearity):

| Moment | Value |
|---|---|
| $K_0$ | $2\pi(A\xi_1^2 - B\xi_2^2)$ |
| $K_2$ | $\pi(A\xi_1^4 - B\xi_2^4)$ |
| $K_4$ | $\frac{\pi}{8}(A\xi_1^6 - B\xi_2^6)$ |

**Constraint $K_0 = 0$** (no net uniform bias): $B = A\xi_1^2/\xi_2^2$. Then:

$$K_2^\text{DoG} = \pi A\xi_1^4\left(1 - \frac{\xi_1^2}{\xi_2^2}\right) > 0 \quad (\xi_2 > \xi_1)$$

$$K_4^\text{DoG} = \frac{\pi A}{8}\xi_1^6\left(1 - \frac{\xi_1^4}{\xi_2^4}\right) > 0 \quad (\xi_2 > \xi_1)$$

**Note:** With $K_0 = 0$ constraint and $\xi_2 > \xi_1$, both $K_2$ and $K_4$ are positive → same problem as pure Gaussian. **A standard DoG with $K_0=0$ cannot produce a Swift–Hohenberg instability either.**

To get negative $K_2,K_4$ (required for $\mu,\lambda > 0$), one of:
1. Allow $K_0 \neq 0$ and take the net sign of $K_2, K_4$ negative, i.e. let the inhibitory Gaussian dominate at the $r^3,r^5$ level.
2. Use $\xi_1 > \xi_2$ (inhibitory component is shorter-range than excitatory) — this reverses the signs.
3. Use the Mexican hat form (Section 4.2), which naturally achieves $K_2 < 0$, $K_4 < 0$.

**Interpretation:** For the URP coherence kernel to produce biological helix geometry via Swift–Hohenberg, it must be of **Mexican-hat type** (excitatory at short range, inhibitory at medium range), not a pure Gaussian or standard DoG. This is a **non-trivial structural constraint on $\mathcal{I}[\phi]$**.

---

## 5. Connection to $\alpha_\text{target}$

With the Mexican hat kernel at $\xi(\kappa_\text{crit})$, the threshold condition (Lucidic Compass eq. 4.1) becomes:

$$\alpha_\text{target} = \pi C\xi^4(\kappa_\text{crit})$$

where $C$ is the kernel amplitude and $\xi(\kappa_\text{crit})$ is determined by the holographic scaling (Section 2.6 of Lucidic Compass). At $\kappa_\text{crit} \approx 0.4623$ and $\xi_0 = 0.469\,\text{nm}$:

$$\xi(\kappa_\text{crit}) = \frac{\xi_0}{\sqrt{\kappa_\text{crit}}} \approx \frac{0.469}{\sqrt{0.4623}} \approx 0.690\,\text{nm}$$

Then:

$$\alpha_\text{target} = \pi C \times (0.690)^4 \approx 0.716\,C\,\text{nm}^4$$

For $\alpha_\text{target} = 0.0887$:

$$C \approx \frac{0.0887}{0.716} \approx 0.124\,\text{nm}^{-4}$$

This is the **required Mexican hat amplitude** to make the URP scalar sector consistent with QCD-derived parameters and DNA geometry — a modest coupling strength, consistent with the kernel survey in Lucidic Compass Section 5.

---

## 6. Simulation Protocol

To extract kernel moments from Project-Genesis dynamics:

1. **Run URP field evolution** on the cylindrical domain with parameters $(\alpha, \beta, G, \kappa)$ from the QCD sector.
2. **Measure the two-point function**:
   $$C(\mathbf{r}) = \langle \phi(\mathbf{x})\,\phi(\mathbf{x}+\mathbf{r})\rangle$$
3. **Identify the effective kernel**: in the linearized regime, $K_\text{eff}(r) \propto C(r)$ (the field's correlation function is set by the kernel).
4. **Compute moments numerically**:
   $$K_0 = 2\pi\int_0^\infty r\,K_\text{eff}(r)\,dr$$
   $$K_2 = \frac{\pi}{2}\int_0^\infty r^3\,K_\text{eff}(r)\,dr$$
   $$K_4 = \frac{\pi}{32}\int_0^\infty r^5\,K_\text{eff}(r)\,dr$$
5. **Check against threshold**: plug $K_0,K_2,K_4$ into the threshold condition and verify consistency with $\alpha_\text{target}$.

This is the bridge from analytic moments to sim-derived moments.

---

## 7. Open Questions

1. **$\epsilon \neq 0$ from Mexican hat alone?** The pure Mexican hat gives $\epsilon = K_0 = 0$ — exactly at threshold. The finite $\epsilon$ needed for a genuine stability window must come from the $G\nabla V \cdot \nabla\phi$ term or from a small correction to the kernel. Deriving this is Conjecture B from the Lucidic Compass.

2. **URP derivation of $K_\text{MH}$:** Why should $\mathcal{I}[\phi]$ produce a Mexican-hat kernel? The physical argument is that URP rewards local coherence (short-range positive) but penalizes redundancy (medium-range inhibitory). Formalizing this via information-theoretic arguments is the next analytic step.

3. **$\kappa$-dependence of $C$:** The amplitude $C(\kappa)$ of the kernel should itself depend on $\kappa$ via the holographic scaling. Deriving $C(\kappa)$ from the URP dynamics is part of Conjecture A.

---

*Cross-references:*
- *`Docs/The_Lucidic_Compass.md` — Sections 3–5 (uses these moments)*
- *`Docs/Narrowing_the_N3_Question.md` — QCD sector parameters*
- *`sims/` — simulation infrastructure for step 6 above*

*Document status: Working derivation note. Analytic results in Sections 3–5 are exact for the stated kernel families. Section 5 numerical values are estimates pending full holographic $\xi(\kappa)$ computation.*

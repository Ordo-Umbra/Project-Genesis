# GitHub Actions Workflows

## `mc_confinement.yml` — Monte Carlo Confinement Analysis

Runs automatically on every push/PR to `main`, and weekly on Mondays.

### What it does

| Job | What runs | Time |
|---|---|---|
| **tests** | `pytest tests/test_gauge_mc.py` + `pytest tests/test_gauge_mc_confinement.py` | ~5 min |
| **analysis** | Quick smoke test + full 7-point beta scan + 4 figures | ~10 min |

### Viewing results on your phone

1. Go to the **Actions** tab on GitHub
2. Click the most recent **Monte Carlo Confinement Analysis** run
3. The **job summary** shows the theory-comparison table directly on screen — no download needed
4. To view the figures, click **confinement-results-`<sha>`** under *Artifacts* and download the zip

### Theory predictions checked

| Label | Prediction | URP anchor |
|---|---|---|
| P1 | σ > 0 at β_low (area law active) | β-sectorisation §4.A |
| P2 | σ(β_low) ≥ σ(β_high) | Asymptotic-freedom direction |
| P3 | \|⟨P⟩\| < 0.3 at β_low (Z_N unbroken) | Confined phase |
| P4 | \|⟨P⟩\| increases with β | URP crossover to deconfined sector |
| P5 | W(3,3) < W(1,1) at β_low | Wilson-loop area-law ordering |
| P6 | χ(2,2) > 0 at β_low | Perimeter-subtracted string tension > 0 |

### Manual trigger

You can trigger a run at any time from the Actions tab → **Monte Carlo Confinement Analysis** → **Run workflow**.

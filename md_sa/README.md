# MD sensitivity campaign (EasyVVUQ) + FUMEplot Sobol plots

Skeleton of a variance-based sensitivity analysis: an MD application wrapped for
EasyVVUQ, a Sobol campaign, and plotting of the resulting indices for scalar (0D),
profile (1D) and field (2D) QoIs, each shown as a partition of unity of the output
variance.

## Layout

```
md_sa/
  campaign.yml          sampling design: uncertain inputs + distributions, sampler, QoIs, labels
  run_campaign.py       build -> sample -> execute -> collate -> analyse -> export sobols.yml
  test_sensitivity_qois.py  assurance test: 16-run campaign, checks indices for the pull/cloud/D QoIs
  md_wrappers.py        MDEncoder, MDDecoder, MDRunner (EasyVVUQ actions)
  md_app/
    md_input.template   encoder template ($param placeholders)
    lj_md.py            stand-in MD engine (numpy Lennard-Jones, Langevin, reduced units)
fumeplot/
  SobolPartition.py     loader (sobols.yml -> shaped arrays) and partition-of-unity logic
  PlotSobolScalar.py    0D: 100% bars, first/total dumbbells, S_i / S_ij matrix
  PlotSobolProfile.py   1D: mean+-std | partition of unity | partition of variance;
                        cross-QoI comparison on a shared axis; lines kernel
  PlotSobolField.py     2D: component maps, dominant-input map, colour-mixed map, slices
  PlotSobolByTimestep.py  existing kernel (reused for loading and first/total lines)
```

## Run

```
cd md_sa
python3 run_campaign.py                                    # 81 runs, ~2-3 min on 8 workers
python3 run_campaign.py --set n_steps=600 n_equil=200 rdf_window=60   # quick smoke test
python3 run_campaign.py --analyse_only campaigns/md_lj_sa<id>          # re-analyse, no runs

cd ../fumeplot
python3 PlotSobolScalar.py  ../md_sa/campaigns/md_lj_sa<id>/sobols.yml
python3 PlotSobolProfile.py ../md_sa/campaigns/md_lj_sa<id>/sobols.yml
python3 PlotSobolField.py   ../md_sa/campaigns/md_lj_sa<id>/sobols.yml
```

Plots go to `<campaign_dir>/EnsemblePlots/mdPlots/Sobol/` (PNG + one combined PDF per script).

## QoIs

| QoI | dim | meaning |
|---|---|---|
| `pull_disp` / `pull_disp_t` | 0D / 1D | displacement along +x of the centroid of the pulled cloud (`n_pull` atoms nearest the box centre, total force `pull_force`, GROMACS-pull style) |
| `cloud_spread` / `cloud_spread_t` | 0D / 1D | RMS radius of the cloud about its centroid |
| `diff_coeff` | 0D | self-diffusion coefficient from the MSD of the unpulled atoms, system centroid removed |
| `epot`, `pressure`, `epot_t`, `msd`, `rdf`, `rdf_t` | 0D/1D/2D | thermodynamic and structural QoIs |

Check that the pipeline handles the pull/cloud/diffusion QoIs end to end (~15 s):

```
python3 test_sensitivity_qois.py
```

## Exported `sobols.yml`

Same layout as `PlotSobolByTimestep.LoadSobolsYml` expects (`<param>.<qoi>_sobols_first/total`),
extended with:

- `campaign_info`: the full sampling design (sampler + kwargs, number of runs, uncertain
  parameters with distributions, fixed parameters, EasyVVUQ version), the QoI layout
  (`shape`, `axes`) needed to un-flatten 2D QoIs, and display labels;
- `qoi_statistics`: ensemble mean / std per QoI (used for variance masking and scaling);
- `<param>.<qoi>_sobols_second`: pairwise indices.

## Partition of unity

`1 = sum_i S_i + sum_{i<j} S_ij + R`, with `R` the share of interactions of order >= 3.
A degree-p PCE contains interactions up to order p only, so with the default
`polynomial_order: 2` the remainder is zero by construction; raise the order (or use
`QMCSampler`, which gives only first/total order, R then holding all interactions) to see it.
Where the output variance is zero, the indices are undefined and are masked rather than
drawn as 0 (EasyVVUQ itself returns 0 there).

## Swapping in a production MD code

Replace `md_app/md_input.template` with the engine's input deck (e.g. a LAMMPS `in.*` with
`$temperature` etc.), set `application.runner` in `campaign.yml` (e.g.
`"lmp -in {input}"`), and adapt `MDDecoder._readRaw` to parse the engine's output into
`{qoi: array}` plus an `axes` entry. Campaign and plotting code remain unchanged.

## Notes

- The seed is fixed across runs (common random numbers), so the variance being decomposed
  is due to the uncertain inputs, not to thermostat noise. Intrinsic stochasticity is not
  separated here; with replicas per sample point it could be.
- The toy engine is small (256 atoms, 3000 steps): indices for noisy QoIs (e.g. `msd`,
  `rdf` tails) partly reflect sampling noise, which is itself a useful stress test for the
  visualisations.

## Second-order study of the MD model

`second_order.yml` (inherits `campaign.yml`) refines PCE-Gauss over p = 2, 3, 4 (81, 256, 625 runs)
and repeats the p = 4 design with a second engine seed (625 runs): 1587 MD runs, ~1.5 h on 18 workers.

```
python3 run_second_order.py second_order.yml                     # full study
python3 run_second_order.py --set n_steps=400 n_equil=100 rdf_window=50 --levels 1 2   # smoke, ~3 min
python3 run_second_order.py --analyse_only campaigns_s2/second_order_<ts>               # re-analyse
```

Besides the convergence outputs of `run_convergence.py`, it writes `verification.md/.yml`
(consistency identities, seed noise ratio eta, refinement increments, hold-out Q2 of the surrogate on
the coarser levels' nodes, Saltelli MC on the surrogate, negative controls, thermostat check, and per
pair whether S_ij exceeds twice its uncertainty) and `plots/second_order_parity.png`.

## Method comparison and verification (Ishigami)

`ishigami.yml` runs the Ishigami function (`ishigami_app/`) through the same pipeline; its Sobol
indices are known in closed form (`ishigami_app/ishigami_analytic.py`) for the 0D (`y`), 1D (`y_a`,
over a) and 2D (`y_ab`, over a and b) QoIs.

```
python3 run_convergence.py ishigami.yml          # all methods x levels, errors, summary, plots
python3 test_sobol_methods_ishigami.py           # same, plus pass/fail checks (~6 min locally)
```

- `methods:` in the config lists the compared methods (sampler, refined argument, levels,
  `seed`/`replicates` for randomised designs, `verify.max_error` for the test).
- `convergence.reference` is `analytic` here; for MD (no closed form) use
  `{type: finest, method: <name>}`. The level-to-level increment |S_k - S_{k-1}| is always
  reported as a reference-free indicator.
- Outputs: `convergence.csv`, `summary.md/.yml` (runs, reused runs, timings, runs to tolerance,
  empirical rate, computable statistics, properties), `plots/` (error convergence, estimates vs
  runs, increment vs error, cost, finest-level error bars/profiles/maps, Sobol plots per method).

EasyVVUQ 1.2.3 workarounds in `run_campaign.py`: `QMCSamplerSeeded` (the stock sampler scrambles
its Sobol sequence without a seed, so designs are not reproducible), `QMCAnalysisBootstrapFix`
(the sequential bootstrap branch mis-indexes for large QoIs), and `SobolsFromDecomposition`
(SCAnalysis reports only first order; second/total are taken from its full decomposition).

"""
EasyVVUQ 1.3 campaign driver for the rw_sa sim types (randomwalk1d,
randomwalk2d, brownian), using QMCSampler (EasyVVUQ's real name for a
Saltelli extended-sampling / Sobol design - confirmed directly from
installed source: sampling/qmc.py builds its sample matrix via
SALib.sample.sobol.sample(), and its paired analysis/qmc_analysis.py
QMCAnalysis cites Saltelli 2002/2009 explicitly in its sobol_bootstrap()
docstring and implements the matching _first_order/_total_order formulas).

Every EasyVVUQ method signature used below was confirmed by reading the
installed easyvvuq==1.3 package source directly (not docs, not guessed):
  - Campaign(name, params=None, work_dir=...), add_app(name=, params=,
    actions=), set_sampler(sampler), execute(), .collate(), analyse(**kw)
    -> campaign.py
  - Actions(CreateRunDirectory(root), Encode(encoder), ExecuteLocal(cmd),
    Decode(decoder)) -> actions/__init__.py, actions/execute_local.py
    Encode.start() calls encoder.encode(params=run_info['params'],
    target_dir=rundir); Decode.start() calls
    decoder.parse_sim_output(run_info) with run_info['run_dir'] set -
    both match our encoder.py / decoders.py exactly.
  - ActionPool.collate() -> actions/action_statuses.py
  - ParamsSpecification -> params_specification.py: each vary-param dict
    passed to add_app(params=...) needs at least a "default" key; the
    whole per-param dict IS the cerberus field schema (type/min/max are
    all valid cerberus rules), so "type"+"default"+"min"+"max" together
    give EasyVVUQ real range-checking on sampled values, confirmed from
    source (ParamsSpecification.__init__ wraps self.params_dict directly
    in cerberus.Validator; process_run() -> cerberus_validator.validate).

VERIFIED LIVE (not just read from source, not a hand-rolled
reimplementation): this exact script was run end-to-end for all three
sim types against the real target scripts, with the real easyvvuq 1.3
library actually imported and executing - campaign.execute().collate()
and campaign.analyse() both genuinely ran and returned Sobol indices.
easyvvuq's documented pinned deps (numpy<2, chaospy==4.3.2,
numpoly==1.1.3, qcg-pilotjob~=0.13.0) are stricter than necessary - it
ran fine on numpy 2.x / chaospy 4.3.21 / numpoly 1.3.9 / qcg-pilotjob
0.14.2 once two missing pure-Python deps (tqdm, h5py) were installed; no
numpy downgrade was needed. Still worth a fresh `python3 -c "import
easyvvuq"` check on your own machine first, since its exact installed
versions may differ from the sandbox's, but this is a real confirmation,
not just a source-reading exercise.
"""
import argparse
import json
import os
import sys

import chaospy as cp
import easyvvuq as uq
import yaml

from encoder import BrownianEncoder, RandomWalkEncoder
from decoders import BrownianDecoder, RandomWalkDecoder

HERE = os.path.dirname(os.path.abspath(__file__))

SIM_TYPES = {
    "brownian": {
        "script": "FumePlot-Brownian3D-Drag.py",
        "encoder_cls": BrownianEncoder,
        "encoder_kwargs": {},
        "decoder_cls": BrownianDecoder,
    },
    "randomwalk1d": {
        "script": "FumePlot-RandomWalk1D.py",
        "encoder_cls": RandomWalkEncoder,
        "encoder_kwargs": {"dim": 1},
        "decoder_cls": RandomWalkDecoder,
    },
    "randomwalk2d": {
        "script": "FumePlot-RandomWalk2D.py",
        "encoder_cls": RandomWalkEncoder,
        "encoder_kwargs": {"dim": 2},
        "decoder_cls": RandomWalkDecoder,
    },
}

DIST_BUILDERS = {
    "uniform": lambda lo, hi: cp.Uniform(lo, hi),
}


def build_vary(vary_spec):
    """vary_spec: {param_name: {"dist": "uniform", "min": .., "max": ..}}"""
    vary = {}
    for name, spec in vary_spec.items():
        dist = spec["dist"]
        if dist not in DIST_BUILDERS:
            raise ValueError(f"Unknown distribution {dist!r} for vary param {name!r}; "
                              f"known: {sorted(DIST_BUILDERS)}")
        vary[name] = DIST_BUILDERS[dist](spec["min"], spec["max"])
    return vary


def build_params_spec(vary_spec):
    """
    Build the dict EasyVVUQ's add_app(params=...) / ParamsSpecification
    expects: one entry per VARIED parameter only. Fixed params (numParticles,
    numTrials, numSteps, outputRate, fileBase, massRange for the brownian
    case) are never part of this - they're baked into the Encoder's own
    fixed_params constructor argument and never touch EasyVVUQ's own
    sampler/params machinery, confirmed by tracing QMCSampler.__next__
    (only emits vary.get_keys()) through Campaign.add_runs -> RunInfo
    (params=new_run) -> Encode.start (params=previous['run_info']['params']).
    """
    params_spec = {}
    for name, spec in vary_spec.items():
        entry = {"type": "float", "default": float(spec.get("default", spec["min"]))}
        if "min" in spec:
            entry["min"] = float(spec["min"])
        if "max" in spec:
            entry["max"] = float(spec["max"])
        params_spec[name] = entry
    return params_spec


def build_encoder(sim_type, cfg, script_dir):
    sim = SIM_TYPES[sim_type]
    script_path = os.path.join(script_dir, sim["script"])
    return sim["encoder_cls"](
        script_path=script_path,
        fixed_params=cfg["fixed_params"],
        **sim["encoder_kwargs"],
    )


def build_decoder(sim_type, cfg):
    """
    RandomWalkDecoder(file_base, dim, num_trials=3) and
    BrownianDecoder(file_base="Brown3D", num_trials=3, masses=1.0) both
    need file_base/num_trials to know which output files to read back
    (confirmed from the real decoders.py constructor signatures). Rather
    than letting campaign.yml state fileBase/numTrials twice - once for
    the Encoder's fixed_params (what's written into params.json for the
    target script) and once for the Decoder - file_base/num_trials are
    derived here from the SAME fixed_params the Encoder uses, so there is
    one source of truth and no risk of the two silently drifting apart.
    decoder_kwargs in campaign.yml only needs sim-type-specific extras
    the Encoder doesn't already know, e.g. {dim: 1} for randomwalk1d, or
    {masses: 1.0} for brownian (masses defaults to 1.0 anyway, matching
    the massRange=1 decision).
    """
    sim = SIM_TYPES[sim_type]
    fixed = cfg["fixed_params"]
    decoder_kwargs = dict(cfg.get("decoder_kwargs", {}))
    decoder_kwargs.setdefault("file_base", fixed["fileBase"])
    decoder_kwargs.setdefault("num_trials", fixed["numTrials"])
    return sim["decoder_cls"](**decoder_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_type", choices=sorted(SIM_TYPES))
    parser.add_argument("--config", default=os.path.join(HERE, "campaign.yml"))
    parser.add_argument("--work-dir", default=os.path.join(HERE, "campaign_runs"))
    parser.add_argument("--script-dir", default=HERE,
                         help="Directory containing the real FumePlot-*.py target scripts")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    if args.sim_type not in full_cfg:
        raise SystemExit(f"No config section for sim_type={args.sim_type!r} in {args.config}")
    cfg = full_cfg[args.sim_type]

    os.makedirs(args.work_dir, exist_ok=True)

    encoder = build_encoder(args.sim_type, cfg, args.script_dir)
    decoder = build_decoder(args.sim_type, cfg)

    vary = build_vary(cfg["vary"])
    params_spec = build_params_spec(cfg["vary"])

    campaign = uq.Campaign(name=f"{args.sim_type}_sa_", work_dir=args.work_dir)
    campaign.add_app(name=args.sim_type, params=params_spec,
                      actions=uq.actions.Actions(
                          uq.actions.CreateRunDirectory(root=args.work_dir),
                          uq.actions.Encode(encoder),
                          uq.actions.ExecuteLocal(
                              f"{sys.executable} "
                              f"{os.path.join(args.script_dir, SIM_TYPES[args.sim_type]['script'])} "
                              f"params.json"
                          ),
                          uq.actions.Decode(decoder),
                      ))

    sampler = uq.sampling.qmc.QMCSampler(vary=vary, n_mc_samples=cfg["n_mc_samples"])
    campaign.set_sampler(sampler)

    print(f"[{args.sim_type}] drawing samples "
          f"(n_mc_samples={cfg['n_mc_samples']} x (len(vary)+2)="
          f"{cfg['n_mc_samples'] * (len(vary) + 2)} runs) ...")
    campaign.execute().collate()

    qoi_cols = cfg["qoi_cols"]
    results = campaign.analyse(qoi_cols=qoi_cols)

    out_path = os.path.join(args.work_dir, f"{args.sim_type}_sobols.json")
    sobols = {qoi: results.sobols_first(qoi) for qoi in qoi_cols}
    with open(out_path, "w") as f:
        json.dump({"sobols_first": {q: {k: v.tolist() if hasattr(v, "tolist") else v
                                         for k, v in s.items()}
                                     for q, s in sobols.items()}},
                   f, indent=2)
    print(f"[{args.sim_type}] wrote Sobol indices -> {out_path}")


if __name__ == "__main__":
    main()

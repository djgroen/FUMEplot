"""
EasyVVUQ Encoders for the rw_sa random walk / Brownian motion scripts.

Each encoder's job: take the dict of sampled parameter values EasyVVUQ's
sampler passes in (keyed by whatever names campaign.yml defines) plus a
fixed set of non-sampled parameters (numParticles, numTrials, numSteps,
outputRate, fileBase - decided once per campaign, not resampled per run),
translate them into the actual global-variable names the rw_sa/*.py
scripts' config-override mechanism expects (see FumePlot-*.py's
_PARAMS_FILE / globals()[_key] = _value block, added in PR #18), and
write that as the JSON file the script reads as its one CLI argument.

Matches EasyVVUQ's own encoder convention exactly (verified against the
installed easyvvuq 1.3 package source,
easyvvuq/encoders/generic_template.py and copy_encoder.py): any plain
class with
    encode(self, params={}, target_dir='')
works as an encoder - no base/abstract class exists to inherit from.
get_restart_dict() and element_version() are only needed if the Campaign
itself gets saved/restarted through EasyVVUQ's database (CopyEncoder in
the real package implements exactly these two, same pattern used here).

CONFIRMED against the real rw_sa/*.py file content (both override
mechanisms seen first-hand): FumePlot-Brownian3D-Drag.py uses
`globals()[_key] = _value` (overrides ANY global by exact name, silently
creates an unused new global on a typo rather than erroring).
FumePlot-RandomWalk1D.py/RandomWalk2D.py use a different mechanism - a
`params` dict that gets `params.update(json.load(f))`'d - which only
ever overrides pre-declared `params` dict keys, but those keys happen to
match the exact same global names used below (flowVelocity unsuffixed
for RW1D, flowVelocityX/Y for RW2D, diffSpread for both). Both
mechanisms were end-to-end tested against the real scripts via
subprocess (not fixtures) - see fumeplot-technical project notes.
validate_against_script() remains in place as a safety net for both
mechanisms regardless (catches a typo'd override key before a campaign
run wastes time on it).
"""
import ast
import json
import os


def validate_against_script(script_path, override_keys):
    """
    Fail loudly, before a campaign run wastes time, if any override key
    doesn't match a top-level assignment target in the target script -
    rather than silently doing nothing via globals()[key] = value on a
    name the script never reads.

    Parses the script's AST rather than importing it, since importing
    one of these scripts directly would otherwise run its whole
    simulation (and, for the no-args default path, try to plt.show()).
    """
    with open(script_path) as f:
        tree = ast.parse(f.read(), filename=script_path)

    assigned_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)

    missing = [k for k in override_keys if k not in assigned_names]
    if missing:
        raise ValueError(
            f"{script_path}: override key(s) {missing} don't match any "
            f"top-level assignment in this script - they would be "
            f"silently ignored by the globals()[key]=value override "
            f"mechanism, not applied. Known top-level names: "
            f"{sorted(assigned_names)}"
        )


class _JSONOverrideEncoder:
    """
    Shared write logic for both encoders below: merges fixed_params with
    the sampler-provided params (sampler values win on a clash), maps
    sampler-facing names to the target script's actual global-variable
    names via SAMPLER_TO_SCRIPT_NAME, validates the result against the
    real script, and writes it as the JSON overrides file.

    script_path: path to the rw_sa/*.py script this encoder targets -
    used only for validate_against_script(), never executed/imported.
    """

    SAMPLER_TO_SCRIPT_NAME = {}  # overridden per subclass

    def __init__(self, script_path, fixed_params=None, target_filename="params.json"):
        self.script_path = script_path
        self.fixed_params = dict(fixed_params or {})
        self.target_filename = target_filename

    def _translate(self, sampler_params):
        translated = {}
        for key, value in sampler_params.items():
            if key not in self.SAMPLER_TO_SCRIPT_NAME:
                raise ValueError(
                    f"{type(self).__name__}: unrecognised sampler "
                    f"parameter {key!r}; known sampler-facing names are "
                    f"{sorted(self.SAMPLER_TO_SCRIPT_NAME)}"
                )
            translated[self.SAMPLER_TO_SCRIPT_NAME[key]] = value
        return translated

    def encode(self, params=None, target_dir=''):
        params = params or {}
        if not target_dir:
            raise RuntimeError('No target directory specified to encoder')

        overrides = dict(self.fixed_params)
        overrides.update(self._translate(params))

        validate_against_script(self.script_path, overrides.keys())

        target_path = os.path.join(target_dir, self.target_filename)
        with open(target_path, 'w') as f:
            json.dump(overrides, f)

    def get_restart_dict(self):
        return {
            "script_path": self.script_path,
            "fixed_params": self.fixed_params,
            "target_filename": self.target_filename,
        }

    def element_version(self):
        return "0.1"


class RandomWalkEncoder(_JSONOverrideEncoder):
    """
    Encoder for FumePlot-RandomWalk1D.py / FumePlot-RandomWalk2D.py.

    dim=1: sampler params "advection_x" (-> flowVelocity) and
           "diffusivity" (-> diffSpread). RW1D genuinely has a single
           unsuffixed flowVelocity param (no X suffix), confirmed
           against the real script - see module docstring.
    dim=2: sampler params "advection_x"/"advection_y"
           (-> flowVelocityX/flowVelocityY) and "diffusivity"
           (-> diffSpread).
    """

    def __init__(self, dim, script_path, fixed_params=None, target_filename="params.json"):
        if dim not in (1, 2):
            raise ValueError(f"dim must be 1 or 2, got {dim!r}")
        self.dim = dim
        if dim == 1:
            self.SAMPLER_TO_SCRIPT_NAME = {
                "advection_x": "flowVelocity",
                "diffusivity": "diffSpread",
            }
        else:
            self.SAMPLER_TO_SCRIPT_NAME = {
                "advection_x": "flowVelocityX",
                "advection_y": "flowVelocityY",
                "diffusivity": "diffSpread",
            }
        super().__init__(script_path, fixed_params, target_filename)


class BrownianEncoder(_JSONOverrideEncoder):
    """
    Encoder for FumePlot-Brownian3D-Drag.py.

    Sampler params: "advection_x"/"advection_y"/"advection_z"
    (-> flowVelocityX/Y/Z), "fluidDrag" (-> fluidDrag, name unchanged),
    "mass" (-> massRange).

    Does NOT expose brownianForce as a sampler parameter: it's derived
    (brownianForce = sqrt(2*fluidDrag)), recomputed by the script itself
    after the override loop applies - see FumePlot-Brownian3D-Drag.py.
    """

    SAMPLER_TO_SCRIPT_NAME = {
        "advection_x": "flowVelocityX",
        "advection_y": "flowVelocityY",
        "advection_z": "flowVelocityZ",
        "fluidDrag": "fluidDrag",
        "mass": "massRange",
    }

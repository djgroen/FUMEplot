"""
Shared helpers for Sobol plots of 0D, 1D and 2D QoIs
(PlotSobolScalar.py, PlotSobolProfile.py, PlotSobolField.py).

Input is a `sobols.yml` in the layout read by PlotSobolByTimestep.LoadSobolsYml,
optionally extended (as written by md_sa/run_campaign.py) with

    campaign_info.qoi_layout:  {qoi: {shape: [...], axes: {name: [...]}}}
    campaign_info.param_order, campaign_info.labels
    qoi_statistics:            {qoi: {mean: [...], std: [...]}}
    <param>.<qoi>_sobols_second: {<other_param>: [...]}

Flattened arrays are restored to their QoI shape. Files without a layout
(e.g. FabHomecoming reports) are read as 1D QoIs over the array index.

The indices of each QoI are turned into a partition of unity of its variance

    1 = sum_i S_i  +  sum_{i<j} S_ij  +  R

where R = 1 - sum_i S_i - sum_{i<j} S_ij is the share of interactions of
order >= 3 (zero for a degree-2 PCE). Without second-order indices, all
interactions are reported together as R = 1 - sum_i S_i. Where the QoI
variance vanishes the indices are undefined and are masked as NaN.
"""

from collections import namedtuple
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
import yaml

from PlotSobolByTimestep import (LoadSobolsYml, PARAM_COLOURS, INTERACTION_COLOUR,
                                 _formatLabel, _fileName)

HIGHER_ORDER_COLOUR = '#7a7974'
MASK_COLOUR = '#f0efec'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'

# texture: secondary encoding for interaction components (CVD / print)
PAIR_HATCH = '///'
HIGHER_HATCH = 'xxx'

# blue sequential ramp, light -> dark, for index magnitudes
SEQUENTIAL_BLUE = mcolors.LinearSegmentedColormap.from_list(
    'sobol_blue', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#2a78d6', '#1c5cab', '#104281', '#0d366b'])

Component = namedtuple('Component', ['key', 'label', 'colour', 'hatch', 'values'])


class SobolQoi:
    """Sobol indices of one QoI, restored to its shape."""

    def __init__(self, name, shape, axes, first, total, second, mean, std):
        self.name = name
        self.shape = tuple(shape)
        self.axes = axes            # {axis_name: coords}, in array-axis order
        self.first = first          # {param: ndarray(shape)}
        self.total = total          # {param: ndarray(shape)}
        self.second = second        # {(p, q): ndarray(shape)}, p before q in param order
        self.mean = mean            # ndarray(shape) or None
        self.std = std              # ndarray(shape) or None

    @property
    def ndim(self):
        return len(self.shape)


class SobolSet:
    """All QoIs of one sensitivity campaign plus the campaign description."""

    def __init__(self, info, params, qois):
        self.info = info
        self.params = params        # fixed order -> fixed colours
        self.qois = qois            # {name: SobolQoi}
        self.labels = info.get('labels', {}) or {}

    def label(self, key):
        return self.labels.get(key, _formatLabel(str(key)))

    def colour(self, param):
        return PARAM_COLOURS[self.params.index(param) % len(PARAM_COLOURS)]

    def byDim(self, ndim, names=None):
        return [q for n, q in self.qois.items() if q.ndim == ndim and (names is None or n in names)]

    def describeDesign(self):
        """One-line description of the sampling design, for figure captions."""
        s = self.info.get('sampler', {})
        if not s:
            return ''
        kw = ', '.join(f"{k}={v}" for k, v in (s.get('kwargs') or {}).items())
        return f"{s.get('type', '?')}({kw}), {s.get('n_samples', '?')} runs, {len(self.params)} uncertain inputs"


def LoadSobolSet(yml_path, param_order=None):
    with open(yml_path) as f:
        report = yaml.safe_load(f)
    info = report.get('campaign_info', {}) or {}
    layout = info.get('qoi_layout', {}) or {}
    stats = report.get('qoi_statistics', {}) or {}
    flat = LoadSobolsYml(yml_path)

    found = sorted({p for q in flat.values() for o in ('first', 'total') for p in q[o]})
    order = param_order or info.get('param_order') or []
    params = [p for p in order if p in found] + [p for p in found if p not in order]

    qois = {}
    for name, entry in flat.items():
        n = max(len(v) for o in ('first', 'total') for v in entry[o].values())
        shape = layout.get(name, {}).get('shape', [n])
        axes = {k: np.asarray(v) for k, v in (layout.get(name, {}).get('axes') or {}).items()}
        if len(shape) and not axes:
            axes = {'index': np.arange(shape[0])}

        def restore(v):
            return np.asarray(v, dtype=float).reshape(shape)

        second = {}
        for p, q in combinations(params, 2):
            v = (report.get(p, {}).get(f"{name}_sobols_second") or {}).get(q)
            if v is not None:
                second[(p, q)] = restore(v)

        st = stats.get(name, {})
        qois[name] = SobolQoi(
            name, shape, axes,
            first={p: restore(v) for p, v in entry['first'].items()},
            total={p: restore(v) for p, v in entry['total'].items()},
            second=second,
            mean=restore(st['mean']) if 'mean' in st else None,
            std=restore(st['std']) if 'std' in st else None)
    return SobolSet(info, params, qois)


def VarianceMask(qoi, rel_tol=1e-6):
    """True where the QoI variance is (numerically) zero and Sobol indices are undefined."""
    if qoi.std is None:
        return np.zeros(qoi.shape, dtype=bool)
    std = np.nan_to_num(qoi.std)
    return std <= rel_tol * max(np.max(std), np.finfo(float).tiny)


def Partition(sset, qoi, split_pairs=False, rel_tol=1e-6, warn_tol=0.05):
    """
    Ordered list of Components whose values sum to 1 wherever defined (NaN elsewhere):
    first-order per parameter, then pairwise interactions (summed, or one per pair if
    `split_pairs`), then the higher-order remainder.
    """
    mask = VarianceMask(qoi, rel_tol)
    comps = [Component(p, sset.label(p), sset.colour(p), None, np.clip(qoi.first[p], 0, 1))
             for p in sset.params if p in qoi.first]

    if qoi.second:
        pairs = [Component(pq, ' × '.join(sset.label(x) for x in pq),
                           MixColours(np.array([[0.5], [0.5]]), [sset.colour(x) for x in pq])[0],
                           PAIR_HATCH, np.clip(v, 0, 1))
                 for pq, v in qoi.second.items()]
        if split_pairs:
            comps += pairs
        else:
            comps.append(Component('pairwise', 'Pairwise interactions', INTERACTION_COLOUR, PAIR_HATCH,
                                   np.sum([c.values for c in pairs], axis=0)))
        rest = ('higher', 'Higher-order interactions', HIGHER_ORDER_COLOUR, HIGHER_HATCH)
    else:
        rest = ('interactions', 'Interactions', INTERACTION_COLOUR, PAIR_HATCH)

    total = np.sum([c.values for c in comps], axis=0)
    residual = 1.0 - total
    worst = np.max(np.abs(np.where(mask, 0, np.minimum(residual, 0))))
    if worst > warn_tol:
        print(f"[WARNING] {qoi.name}: Sobol indices sum to up to {1 + worst:.3f} (> 1); renormalised")
    comps.append(Component(rest[0], rest[1], rest[2], rest[3], np.clip(residual, 0, 1)))

    norm = np.maximum(np.sum([c.values for c in comps], axis=0), 1.0)
    return [c._replace(values=np.where(mask, np.nan, c.values / norm)) for c in comps], mask


# ------------------------------------------------------------------ colour helpers

def _toLinear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _toSrgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def MixColours(weights, colours):
    """
    Convex combination of colours in linear RGB.
    weights: (K, ...) summing to 1 along axis 0; colours: K colour specs.
    Returns (..., 3) sRGB.
    """
    lin = _toLinear(np.array([mcolors.to_rgb(c) for c in colours]))        # (K, 3)
    w = np.nan_to_num(np.asarray(weights, dtype=float))
    mixed = np.tensordot(np.moveaxis(w, 0, -1), lin, axes=([-1], [0]))      # (..., 3)
    return _toSrgb(mixed)


def FadeToSurface(rgb, strength, surface='#ffffff'):
    """Blend towards the surface colour: strength 1 keeps rgb, 0 gives the surface."""
    s = np.clip(np.nan_to_num(strength), 0, 1)[..., None]
    return s * rgb + (1 - s) * np.array(mcolors.to_rgb(surface))


def InkOn(colour):
    """Readable text colour on a filled mark."""
    r, g, b = _toLinear(np.array(mcolors.to_rgb(colour)))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.25 else '#ffffff'


# ------------------------------------------------------------------ output helpers

def PlotFolder(plot_path, code):
    folder = f"{plot_path}/EnsemblePlots/{code}Plots/Sobol"
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def SaveFigure(fig, name, plot_folder, combine_plots_pdf=None):
    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig, bbox_inches='tight')
    fig.savefig(f"{plot_folder}/{_fileName(name)}.png", bbox_inches='tight', dpi=200)


def DrawColourKey(ax, comps, x, rgb_rows, xlabel, title=None, xfmt=None):
    """
    2D colour key: one row per component, colours along x exactly as drawn on the map.
    rgb_rows: (n_comps, len(x), 3). Used where lightness encodes a quantity (share, std).
    """
    dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
    ax.imshow(np.asarray(rgb_rows), aspect='auto', interpolation='nearest', origin='upper',
              extent=(x[0] - dx / 2, x[-1] + dx / 2, len(comps) - 0.5, -0.5))
    ax.set_yticks(range(len(comps)), [c.label for c in comps], fontsize='x-small')
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', labelsize='x-small')
    ax.set_xlabel(xlabel, fontsize='x-small')
    if xfmt is not None:
        ax.xaxis.set_major_formatter(xfmt)
    if title:
        ax.set_title(title, fontsize='x-small', loc='left')
    for s in ax.spines.values():
        s.set_visible(False)


def ComponentLegendHandles(comps):
    import matplotlib.patches as mpatches
    return [mpatches.Patch(facecolor=c.colour, hatch=c.hatch, edgecolor='white', label=c.label)
            for c in comps]

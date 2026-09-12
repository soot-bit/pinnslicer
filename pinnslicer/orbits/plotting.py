# ═══════════════════════════════════════════════════════════════
# Description: photon-orbit plotting.
#
# Classes:
#   OrbitPlotXY    — Cartesian (x, y) in units of r_s
#   OrbitPlotUPhi  — (phi, u) with u = r_s / r
#
# Both take an existing matplotlib `ax`, and can be used standalone.
#
# Functions:
#   set_paper_style()              — apply the figure style used in the paper
#   plot_xy_uphi()                 — the paired (x, y) / (phi, u) figure
#   plot_uphi_with_segment_panels()— (phi, u) plus per-orbit segment panels
#   split_phi_u_segments()         — chop an orbit into dphi-wide segments
#
# The two solvers return OrbitSolution objects (see orbits/solvers.py); the
# figure builders here take dictionaries of those, keyed by (r0, delta).
# ═══════════════════════════════════════════════════════════════
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Palette (for orbits) ───────────────────────────────────────
CRIMSON      = '#A00020'  # dark red           — inner spiralling orbit
GOLDENROD    = '#E5A400'  # golden yellow      - photon sphere
ORANGE       = '#D2451E'  # burnt orange       - spiral in
YELLOW_GREEN = '#4C9A2A'  # yellow-green       - boomerang
COBALT       = '#1F5FA8'  # clean mid-blue     - deflected, u-turn

# One line style per solver, so that FD (solid) and PINN (dotted) are told
# apart by style and the orbit is identified by colour, in every figure.
LINESTYLE = {
    'fd':   dict(linestyle='-',  linewidth=1.0),
    'pinn': dict(linestyle=':',  linewidth=1.8),
}

# Cycled through when an orbit is drawn segment by segment, so that
# consecutive dphi-slices can be told apart.
SEGMENT_STYLES = [
    '--',
    '-.',
    (0, (1, 1)),        # fine dots
    (0, (5, 2, 1, 2)),  # long dash-dot
    (0, (3, 5, 1, 5)),  # sparse dash-dot
    (0, (5, 5)),        # long dash
]

# The style of every figure in the paper. Applied by set_paper_style(), not on
# import: importing a module should not silently reconfigure matplotlib.
PAPER_STYLE = {
    'font.family':         'serif',
    'mathtext.fontset':    'stix',
    'font.size':           11,
    'axes.linewidth':      0.8,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.major.size':    4,
    'ytick.major.size':    4,
    'xtick.minor.size':    2,
    'ytick.minor.size':    2,
    'xtick.top':           True,
    'ytick.right':         True,
    'grid.color':          '#e0e0e0',
    'grid.linewidth':      0.5,
    'legend.frameon':      False,
    'legend.fontsize':     11,
    'legend.labelspacing': 0.8,
    'figure.dpi':          150,
    'savefig.dpi':         300,
}
# ══════════════════════════════════════════════════════════════
def set_paper_style(**overrides):
    """Apply the paper's matplotlib style; keyword arguments override it."""
    mpl.rcParams.update({**PAPER_STYLE, **overrides})
# ══════════════════════════════════════════════════════════════
def to_numpy(x):
    """Convert a tensor to a numpy array; pass anything else through."""
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return x
# ══════════════════════════════════════════════════════════════
def save_figure(fig, name, savedir='figures', formats=('pdf',)):
    """Write `fig` to <savedir>/<name>.<fmt> for each requested format."""
    os.makedirs(savedir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(savedir, f'{name}.{fmt}')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
# ══════════════════════════════════════════════════════════════
class OrbitPlotXY:
# ══════════════════════════════════════════════════════════════
    """Photon orbits in Cartesian (x, y), units of r_s.

    Draws onto a supplied `ax`; does not create a figure.

    Usage:
        fig, ax = plt.subplots()
        OrbitPlotXY(ax).add_orbit(phi, u, color)
    """

    def __init__(self, ax, xlim=(-3, 8), ylim=(-4, 4)):
        self.ax        = ax
        self._has_pinn = False

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x/r_s$')
        ax.set_ylabel(r'$y/r_s$')
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
        ax.grid(True)
        # the photon sphere at r = 1.5 r_s (dashed grey) and the black hole
        # itself, out to the event horizon at r = r_s (filled black)
        ax.add_patch(plt.Circle((0, 0), 1.5, color='#aaaaaa', fill=False,
                                linewidth=0.9, linestyle='--', zorder=3))
        ax.add_patch(plt.Circle((0, 0), 1.0, color='#111111', zorder=4))

    def add_orbit(self, phi, u, color, style='fd', lw=None):
        s    = LINESTYLE[style]
        phi_ = to_numpy(phi)
        u_   = to_numpy(u)
        # polar to Cartesian, in units of r_s: r/r_s = 1/u, so
        # x = cos(phi)/u and y = sin(phi)/u
        self.ax.plot(np.cos(phi_)/u_, np.sin(phi_)/u_,
                     color=color, zorder=2,
                     linestyle=s['linestyle'],
                     linewidth=lw or s['linewidth'])
        if style == 'pinn':
            self._has_pinn = True
# ══════════════════════════════════════════════════════════════
class OrbitPlotUPhi:
# ══════════════════════════════════════════════════════════════
    """Photon orbits in (phi, u) coordinates, u = r_s / r.
    Draws onto a supplied `ax`; does not create a figure.
    Usage:
        fig, ax = plt.subplots()
        OrbitPlotUPhi(ax).add_orbit(phi, u, color, r0=r0, delta=delta)
    """
    def __init__(self, ax, phi_max=9.0, show_yaxis=True):
        self.ax              = ax
        self.phi_max         = phi_max
        self._has_pinn       = False
        self._legend_handles = []
        ax.set_xlim(0, phi_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r'$\phi$ (rad)')
        # Not showing left axis (for segment panels)
        if show_yaxis:
            ax.set_ylabel(r'$u(\phi) = r_s/r$')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        # x ticks: major at π, minor at π/2 with labels
        n_major = int(phi_max / np.pi)
        n_minor = int(phi_max / (np.pi/2))
        major_ticks = [k*np.pi   for k in range(n_major+1)
                       if k*np.pi <= phi_max+0.01]
        minor_ticks = [k*np.pi/2 for k in range(1, n_minor+1)
                       if k*np.pi/2 <= phi_max+0.01
                       and not np.isclose(k*np.pi/2 % np.pi, 0)]
        major_labels = ['$0$'] + [r'$\pi$' if k==1 else rf'${k}\pi$'
                                   for k in range(1, n_major+1)]
        minor_labels = [r'$\frac{\pi}{2}$' if k==1
                        else rf'$\frac{{{k}\pi}}{{2}}$'
                        for k in range(1, n_minor+1)
                        if not np.isclose(k*np.pi/2 % np.pi, 0)]
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_labels)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticklabels(minor_labels, minor=True,
                           fontsize=mpl.rcParams['font.size']-0.5)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        ax.grid(True, which='major', linewidth=0.5)
        ax.grid(True, which='minor', linewidth=0.25, alpha=0.6)
        # the photon sphere, r = 1.5 r_s, is u = 2/3 in these coordinates
        ax.axhline(2/3, color=GOLDENROD, linewidth=0.8,
                   linestyle='--', zorder=1)

    def add_orbit(self, phi, u, color, r0=None, delta=None,
                  style='fd', lw=None, alpha=1.0, show_v0=False):
        s    = LINESTYLE[style]
        phi_ = to_numpy(phi)
        u_   = to_numpy(u)
        self.ax.plot(phi_, u_, color=color, zorder=2,
                     linestyle=s['linestyle'],
                     linewidth=lw or s['linewidth'],
                     alpha=alpha)
        if style == 'fd' and r0 is not None and delta is not None:
            label = rf'$r_0={r0:.1f}\,r_s$, $\delta_0={delta:.1f}°$'
            self._legend_handles.append(
                mlines.Line2D([], [], color=color,
                              linewidth=LINESTYLE['fd']['linewidth'],
                              label=label))
        if style == 'pinn':
            self._has_pinn = True
        if show_v0 and style == 'fd':
            # arrow showing the initial slope v0 = du/dphi at phi = 0,
            # estimated from the first two points of the orbit
            slope = (u_[1]-u_[0]) / (phi_[1]-phi_[0])
            al    = 0.35
            self.ax.annotate('',
                xy=(al, u_[0]+slope*al), xytext=(0, u_[0]),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.0, mutation_scale=12),
                zorder=5)
            dy = -0.04 if np.isclose(slope, 0.0, atol=1e-3) else slope*al - 0.02
            self.ax.text(al+0.02, u_[0]+dy, r'$v_0$',
                        color=color, fontsize=11, va='center', zorder=5)

    def add_orbit_segments(self, segments, color, dphi, styles=SEGMENT_STYLES):
        """Plot a PINN orbit split into its dphi-wide segments.

        segments : list of (phi_seg, u_seg) tuples, indexed by segment number
        color    : line color (same for all segments of this orbit).
        dphi     : segment width, used to shift segments back to [0, dphi].
        styles   : cycling linestyles for segments k=1,2,... (k=0 uses the
                   nominal PINN style).
        """
        pinn_style = LINESTYLE['pinn']['linestyle']

        for k, (phi_seg, u_seg) in enumerate(segments):
            if k == 0:
                self.ax.plot(phi_seg, u_seg, color=color,
                             linestyle=pinn_style, zorder=2)
            else:
                ls = styles[(k - 1) % len(styles)]
                # true location
                self.ax.plot(phi_seg, u_seg, color=color,
                             linestyle=ls, zorder=2)
                # shifted back to [0, dphi] for overlay comparison
                self.ax.plot(phi_seg - k*dphi, u_seg, color=color,
                             linestyle=ls, zorder=2)

        self._has_pinn = True


    def overlay_segments(self, segments, color, dphi, alpha_min=0.3):
        """Overlay all segments of one orbit onto [0, dphi], shifted back.
        All segments solid; alpha fades from 1.0 (segment 0) to alpha_min (last).
        """
        n = len(segments)
        for k, (phi_seg, u_seg) in enumerate(segments):
            alpha = 1.0 if n == 1 else 1.0 - (1.0 - alpha_min) * k / (n - 1)
            self.ax.plot(phi_seg - k*dphi, u_seg, color=color,
                        linestyle='-', alpha=alpha, zorder=2)

    def add_dphi_lines(self, dphi, color='#555555', lw=0.8):
        """Vertical lines at each k*dphi boundary."""
        k = 1
        while k * dphi <= self.phi_max:
            self.ax.axvline(k * dphi, color=color, linewidth=lw,
                            linestyle='-', zorder=1)
            k += 1

    def finalize(self, show_labels=True, show_legend=True,
                legend_anchor=(1.05, 0.57)):
        many_orbits = len(self._legend_handles) > 5

        if show_labels:
            labels = [(0.98, 'event horizon')]
            if not many_orbits:
                labels.append((2/3, 'photon sphere'))
            labels.append((0.02, 'escape to infinity'))

            for u_val, txt in labels:
                self.ax.annotate(txt,
                    xy=(1.0, u_val), xycoords='axes fraction',
                    xytext=(6, 0),   textcoords='offset points',
                    va='center', ha='left',
                    fontsize=mpl.rcParams['font.size'],
                    color='#111111', annotation_clip=False)

        if show_legend:
            if self._has_pinn:
                style_handles = [
                    mlines.Line2D([], [], color='#666666',
                                linestyle=LINESTYLE['fd']['linestyle'],
                                linewidth=LINESTYLE['fd']['linewidth'],
                                label='FD'),
                    mlines.Line2D([], [], color='#666666',
                                linestyle=LINESTYLE['pinn']['linestyle'],
                                linewidth=LINESTYLE['pinn']['linewidth'],
                                label='PINN'),
                ]
                leg1 = self.ax.legend(
                    handles=style_handles,
                    loc='upper left',
                    bbox_to_anchor=(legend_anchor[0], 0.87),
                    borderaxespad=0, ncol=2,
                    fontsize=mpl.rcParams['font.size'])
                self.ax.add_artist(leg1)

            if self._legend_handles:
                self.ax.legend(
                    handles=self._legend_handles,
                    loc='upper left',
                    bbox_to_anchor=legend_anchor,
                    borderaxespad=0,
                    labelspacing=0.5 if many_orbits else mpl.rcParams['legend.labelspacing'],
                    fontsize=mpl.rcParams['font.size'])
# ══════════════════════════════════════════════════════════════
def split_phi_u_segments(phi, u, dphi):
    """Split (phi, u) into dphi-wide segments.

    phi, u : arrays of equal length.
    dphi   : segment width.

    Returns a list of (phi_seg, u_seg) tuples, indexed by segment
    number (segments[k] covers phi in [k*dphi, (k+1)*dphi)).
    """
    # floor with a tolerance, as in the solvers: a phi that is an exact
    # multiple of dphi belongs to the segment that starts there.
    from .solvers import PHI_TOL

    seg_id = np.floor(np.asarray(phi) / dphi + PHI_TOL).astype(np.int64)
    segments = []
    for k in range(seg_id.max() + 1):
        mask = seg_id == k
        segments.append((phi[mask], u[mask]))
    return segments
# ══════════════════════════════════════════════════════════════
def plot_xy_uphi(orbits, phi_max, fd_data, pinn_data=None,
                 name='', save=False, savedir='figures',
                 show_v0=False, show_labels=True,
                 legend_anchor=(1.05, 0.57),
                 xlim=(-3, 8), ylim=(-4, 4), total_width=11.0):
    """Build photon-orbit figure: Cartesian (x, y) and (phi, u).

    Parameters
    ----------
    orbits : list of dict
        Orbit definitions, each with keys r0, delta, color, u0, v0.
    phi_max : float
        Upper phi limit of the (phi, u) panel.
    fd_data : dict
        Maps (r0, delta) to the FD solver's OrbitSolution.
    pinn_data : dict, optional
        Same mapping from the PINN; omit for an FD-only figure.
    name : str
        Base filename, used when save is True.
    save : bool
        Write the figure to `savedir` as a PDF.
    savedir : str
        Output folder for the saved figure.
    show_v0 : bool
        Draw initial-velocity arrows on the (phi, u) panel.
    show_labels : bool
        Annotate event horizon, photon sphere, and escape on the right margin.
    legend_anchor : tuple of float
        Legend position in axes fraction.
    xlim, ylim : tuple of float
        Cartesian panel limits.
    total_width : float
        Figure width in inches.

    Returns
    -------
    fig : matplotlib Figure
    """
    # The (x, y) panel is drawn with an equal aspect ratio, so its height is
    # fixed by its width and the axis ranges; the whole figure follows. The
    # two panels are given widths proportional to their x ranges so that the
    # scales look comparable.
    xrange    = xlim[1] - xlim[0]
    yrange    = ylim[1] - ylim[0]
    figheight = total_width * 0.42 * yrange / xrange

    fig, (ax_xy, ax_uphi) = plt.subplots(
        1, 2,
        figsize=(total_width, figheight),
        gridspec_kw={'width_ratios': [xrange, phi_max],'wspace': 0.1}
    )

    plot_xy   = OrbitPlotXY(ax_xy,    xlim=xlim, ylim=ylim)
    plot_uphi = OrbitPlotUPhi(ax_uphi, phi_max=phi_max)

    for orb in orbits:
        key = (orb['r0'], orb['delta'])
        fd  = fd_data[key]

        plot_xy.add_orbit(fd.phi, fd.u, color=orb['color'], style='fd')
        plot_uphi.add_orbit(fd.phi, fd.u, color=orb['color'],
                            r0=orb['r0'], delta=orb['delta'],
                            style='fd', show_v0=show_v0)

        if pinn_data is not None:
            # the two solvers can stop at different points (they hit the
            # spacetime boundary at slightly different phi), so compare them
            # only over the range both of them covered
            pinn = pinn_data[key]
            k    = min(len(fd.u), len(pinn.u))
            plot_xy.add_orbit(pinn.phi[:k], pinn.u[:k],
                              color=orb['color'], style='pinn')
            plot_uphi.add_orbit(pinn.phi[:k], pinn.u[:k],
                                color=orb['color'],
                                r0=orb['r0'], delta=orb['delta'],
                                style='pinn')

    plot_uphi.finalize(show_labels=show_labels, show_legend=True,
                       legend_anchor=legend_anchor)

    if save:
        save_figure(fig, name, savedir=savedir)

    return fig
# ══════════════════════════════════════════════════════════════
def plot_uphi_with_segment_panels(orbits_main, orbits_panels, dphi, phi_max,
                                  fd_solver, n_points=400,
                                  name='', save=False, savedir='figures',
                                  legend_anchor=(1.88, 0.34)):
    """Main u-phi curve (FD) plus narrow panels, each overlaying one
    orbit's segments onto [0, dphi].

    Parameters
    ----------
    orbits_main, orbits_panels : list of dict
        Orbits of the main panel and of the narrow side panels.
    dphi : float
        Segment width the side panels fold the orbits onto.
    phi_max : float
        Upper phi limit of the main panel.
    fd_solver : callable
        The FD solver used for every curve, e.g. FDPhotonOrbitSolver().
    n_points : int
        Number of phi points at which the orbits are evaluated.

    Returns
    -------
    fig : matplotlib Figure
    """
    # one wide main panel plus one narrow panel per orbit; the widths are
    # proportional to the phi range each panel covers, so a dphi-wide panel
    # is drawn at the same scale as the main axis
    width_ratios = [phi_max] + [dphi] * len(orbits_panels)
    wspace = 0.2

    fig, axes = plt.subplots(
        1, len(width_ratios), figsize=(11, 4),
        gridspec_kw={'width_ratios': width_ratios, 'wspace': wspace}
    )
    ax_main, *ax_panels = axes

    # phi array
    phi = np.linspace(0, phi_max, n_points)

    # main panel: FD, all orbits
    plot_main = OrbitPlotUPhi(ax_main, phi_max=phi_max)
    for orb in orbits_main:
        fd = fd_solver((orb['u0'], orb['v0']), phi)
        plot_main.add_orbit(fd.phi, fd.u, color=orb['color'], style='fd',
                            r0=orb['r0'], delta=orb['delta'], lw=1.5)
    plot_main.add_dphi_lines(dphi, color='#888888', lw=0.8)

    ax_main.legend(
        handles=plot_main._legend_handles,
        loc='center left',
        bbox_to_anchor=legend_anchor,
        borderaxespad=0,
        fontsize=mpl.rcParams['font.size'])

    # narrow panels: one orbit each, segments overlaid on [0, dphi]
    for i, (ax_i, orb) in enumerate(zip(ax_panels, orbits_panels)):
        plot_i = OrbitPlotUPhi(ax_i, phi_max=dphi, show_yaxis=False)
        fd = fd_solver((orb['u0'], orb['v0']), phi)
        segments = split_phi_u_segments(fd.phi, fd.u, dphi)
        plot_i.overlay_segments(segments, color=orb['color'], dphi=dphi)

        is_last = (i == len(orbits_panels) - 1)
        plot_i.finalize(show_labels=is_last, show_legend=False)

    if save:
        save_figure(fig, name, savedir=savedir)

    return fig

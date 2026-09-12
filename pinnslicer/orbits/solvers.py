# ----------------------------------------------------------------------------
# Description: photon-orbit solvers.
#
#   FDPhotonOrbitSolver : O(h^4) finite-difference reference solver
#   PinnSlicer          : recursive deployment of a PINN trained on one
#                         phi-slice [0, dphi]
#
# Both solvers are callables with the same signature,
#
#       solution = solver(y0, phi)
#
# and both return an OrbitSolution, which carries the solution (phi, u, v),
# the status code saying what stopped the loop, and the number of iterations
# the loop performed.
# ----------------------------------------------------------------------------
import warnings
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import torch

# Tolerance used when a phi value is divided by the slice width dphi and the
# quotient is floored: without it, a phi that is an exact decimal multiple of
# dphi (0.6 / 0.1 = 5.999999999999999) is assigned to the previous segment.
PHI_TOL = 1e-9
# ----------------------------------------------------------------------------
class SolverStatus(IntEnum):
    """
    Why a solver's iterative (FD) or recursive (PINN) loop terminated.

    COMPLETED   the last requested phi value was reached; the orbit is
                returned over the full requested range.
    ESCAPED     u dropped below u_min = 0, that is r = 1/u went to infinity:
                the photon escaped.
    ABSORBED    u rose above u_max = 1, that is r fell below the Schwarzschild
                radius: the photon crossed the event horizon.
    MAX_STEPS   the FD solver used up its budget of K steps.
    PHI_GUARD   the FD solver reached its phi_max_guard safety cap without
                hitting either a spacetime boundary or the requested phi.
    """

    COMPLETED = 0
    ESCAPED   = 1
    ABSORBED  = 2
    MAX_STEPS = 3
    PHI_GUARD = 4

    @property
    def description(self):
        """One-line, human-readable form of the status, for tables and logs."""
        return {
            SolverStatus.COMPLETED: "reached requested phi",
            SolverStatus.ESCAPED:   "escaped to infinity",
            SolverStatus.ABSORBED:  "absorbed at event horizon",
            SolverStatus.MAX_STEPS: "maximum number of steps reached",
            SolverStatus.PHI_GUARD: "phi safety cap reached",
        }[self]
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OrbitSolution:
    """
    The result of a call to either solver.

    Attributes
    ----------
    phi : ndarray
        The phi values at which the solution is given. This is the requested
        phi array truncated to what the solver actually computed, so phi, u
        and v always have the same length.
    u : ndarray
        u(phi) = r_s / r.
    v : ndarray
        v(phi) = du/dphi.
    status : SolverStatus
        What stopped the loop; see SolverStatus.
    n_iterations : int
        Number of iterations performed: integration steps for the FD solver,
        phi-segments visited for the PINN solver.
    solver : str
        'fd' or 'pinn', the solver that produced this solution.
    info : dict
        Solver-specific extras (step size h, slice width dphi, ...).
    """

    phi: np.ndarray
    u: np.ndarray
    v: np.ndarray
    status: SolverStatus
    n_iterations: int
    solver: str
    info: dict = field(default_factory=dict)

    @property
    def complete(self):
        """True if the solver reached the last requested phi value."""
        return self.status is SolverStatus.COMPLETED

    @property
    def phi_reached(self):
        """The largest phi actually reached (nan for an empty solution)."""
        return float(self.phi[-1]) if len(self.phi) else float("nan")

    def __len__(self):
        return len(self.phi)

    def __str__(self):
        return (
            f"OrbitSolution({self.solver}): {len(self)} points, "
            f"phi_reached = {self.phi_reached:.3f}, "
            f"{self.n_iterations} iterations, "
            f"status = {self.status.name} ({self.status.description})"
        )
# ----------------------------------------------------------------------------
def validate_phi(phi):
    """
    Check and normalise the phi array requested from a solver.

    The solvers assume phi is non-negative and strictly increasing: the FD
    solver interpolates on it and the PINN solver splits it at segment
    boundaries, and neither is meaningful otherwise.

    Returns:
        phi (ndarray of float64)
    """
    phi = np.asarray(phi, dtype=np.float64)

    if phi.ndim != 1:
        raise ValueError(f"/!\\ phi must be a 1-D array, got shape {phi.shape}")

    if len(phi) < 2:
        raise ValueError(f"/!\\ phi must contain at least 2 points, got {len(phi)}")

    if phi[0] < 0:
        raise ValueError(f"/!\\ phi must be non-negative, got phi[0] = {phi[0]}")

    if np.any(np.diff(phi) <= 0):
        raise ValueError("/!\\ phi must be strictly increasing")

    return phi
# ----------------------------------------------------------------------------
def check_phi_resolution(phi, dphi, strict=False):
    """
    Guard against a phi grid that is too coarse for the slice width dphi.

    The PINN is applied one phi-segment at a time, and the segments are read
    off the requested phi values: the recursion visits one segment per group
    of phi values that share a segment index. A phi grid whose spacing exceeds
    dphi therefore leaves some segments with no requested point at all; those
    segments are skipped, the recursion advances the initial conditions by
    less than the orbit actually travelled, and every point past the first gap
    is silently wrong.

    Parameters:
        phi (ndarray): validated phi array (see `validate_phi`).
        dphi (float): slice width the PINN was trained on.
        strict (bool): raise ValueError instead of warning.

    Returns:
        n_missing (int): number of segments containing no requested phi value.
    """
    segments  = np.floor(phi / dphi + PHI_TOL).astype(np.int64)
    nsegments = int(segments[-1]) + 1

    n_missing = nsegments - len(np.unique(segments))
    if n_missing == 0:
        return 0

    max_gap = float(np.max(np.diff(phi)))
    n_min   = int(np.ceil(phi[-1] / dphi)) + 1

    message = (
        f"/!\\ the phi points are too coarse for dphi = {dphi:g}: "
        f"{n_missing} of {nsegments} phi-segments contain no requested point "
        f"(largest gap between consecutive phi values is {max_gap:g} > {dphi:g}). "
        f"Those segments are skipped by the recursion, so the orbit past the "
        f"first gap is wrong. Supply at least {n_min} points over "
        f"[0, {phi[-1]:g}], for example "
        f"np.linspace(0, {phi[-1]:g}, {max(n_min, 2)})."
    )

    if strict:
        raise ValueError(message)

    warnings.warn(message, stacklevel=3)
    return n_missing
# ----------------------------------------------------------------------------
def _first_true(mask):
    """
    Index of the first True in a 1-D boolean tensor, or len(mask) if there
    is none. Used to decide which spacetime boundary was crossed first.
    """
    hits = torch.nonzero(mask)
    return int(hits[0]) if len(hits) else len(mask)
# ----------------------------------------------------------------------------
class FDPhotonOrbitSolver:
    '''
    Finite-difference reference solver for the photon orbit equation.

    We roughly follow the semantics of scipy.integrate.odeint(func, y0, t, ...),
    which in addition to the initial conditions y0 requires the user to provide
    an array t (called phi here) of points at which the solution is returned.

    We write the 2nd order photon orbit equation

        D^2 u = G(u), with G(u) = 1.5*u**2 - u

    as two 1st order equations

        Du = v
        Dv = G

    where D is the derivative operator, and step both forward with a Taylor
    expansion truncated at O(h^4).

    Parameters:
        h (float): step size.
        K (int): maximum number of steps.
        phi_max_guard (float): safety cap on phi if no boundary is hit.

    Usage:
        fd_solver = FDPhotonOrbitSolver()
        solution  = fd_solver((u0, v0), phi)
    '''
    def __init__(self, h=np.pi/10_000, K=1_000_000, phi_max_guard=10*np.pi):
        self.h = h                          # step size
        self.K = K                          # maximum number of steps
        self.phi_max_guard = phi_max_guard  # safety cap if no boundary is hit

    def __call__(self, y0, phi):
        '''
        y0:  initial conditions (u0, v0)
             u0 = u(0)
             v0 = Du(0)

        phi: numpy array of points at which the solution should be returned.
             phi[-1] sets the requested upper limit of the orbit.

        Returns:
            solution (OrbitSolution)
        '''
        try:
            u0, v0 = y0[0, :]
        except (IndexError, TypeError):
            u0, v0 = y0

        phi = validate_phi(phi)

        # The two spacetime boundaries. Outside [umin, umax] the orbit has
        # left the region in which the equation is being solved, and the
        # integration stops.
        umin = 0.0  # r = 1/u = infinity
        umax = 1.0  # r = 1/u = Schwarzschild radius

        K             = self.K            # step budget
        phi_max_guard = self.phi_max_guard
        phi_max_user  = phi[-1]           # the orbit the caller asked for

        # powers of the step size, computed once outside the loop
        h1 = self.h
        h2 = h1*self.h
        h3 = h2*self.h
        h4 = h3*self.h

        # the integrated orbit, accumulated step by step:
        # x[i] = phi at step i, u[i] = u(x[i]), v[i] = du/dphi at x[i]
        x = [0]
        u = [u0]
        v = [v0]

        # if the loop below runs to exhaustion, the step budget is what
        # stopped it
        status = SolverStatus.MAX_STEPS
        n_iterations = K - 1

        for i in range(1, K):

            # Successive derivatives of G(u) = 1.5 u^2 - u along the orbit,
            # obtained by differentiating G with respect to phi and using
            # du/dphi = v and dv/dphi = G:
            #
            #   G0 = G(u)                    = 1.5 u^2 - u
            #   G1 = dG/dphi                 = (3u - 1) v
            #   G2 = d^2G/dphi^2             = 3 v^2 + (3u - 1) G0
            #   G3 = d^3G/dphi^3             = (9 G0 + (3u - 1)^2) v
            #
            # Q = 3u - 1 = dG/du is shared by three of them.
            Q  = 3*u0 - 1

            G0 = 1.5 * u0**2 - u0
            G1 = Q*v0
            G2 = 3*v0**2 + Q*G0
            G3 = (9*G0 + Q**2) * v0

            # Taylor expansions truncated after the h^4 term, hence O(h^4):
            #   u(x+h) = u + v h + G0 h^2/2! + G1 h^3/3! + G2 h^4/4!
            #   v(x+h) = v + G0 h + G1 h^2/2! + G2 h^3/3! + G3 h^4/4!

            # compute u(x+h)
            u1 = u0 + v0*h1 + G0*h2/2 + G1*h3/6 + G2*h4/24

            # compute v(x+h)
            v1 = v0 + G0*h1 + G1*h2/2 + G2*h3/6 + G3*h4/24

            # ---------------------------------------------------------
            # Termination conditions, recorded in the status code
            # ---------------------------------------------------------
            if u1 < umin:
                status = SolverStatus.ESCAPED     # escaped to infinity
            elif u1 > umax:
                status = SolverStatus.ABSORBED    # absorbed at event horizon
            elif x[-1] > phi_max_user:
                status = SolverStatus.COMPLETED   # reached user-requested phi
            elif x[-1] > phi_max_guard:
                status = SolverStatus.PHI_GUARD   # safety cap
            else:
                # accumulate solution
                x.append(i*h1)
                u.append(u1)
                v.append(v1)

                # apply recursion
                u0 = u1
                v0 = v1
                continue

            # one of the four termination conditions fired: record how many
            # steps were taken and stop. Note the point that triggered it is
            # deliberately not accumulated -- the orbit stops just short of
            # the boundary it would have crossed.
            n_iterations = i
            break

        xa = np.array(x)
        ua = np.array(u)
        va = np.array(v)

        # Actual extent of the computed orbit (boundary, user limit, or guard)
        phi_max_orbit = xa[-1]

        # Return only the requested points that fall within what was computed:
        # an orbit that ended early (escape, absorption, guard) covers less
        # than the caller asked for, and the rest would be extrapolation.
        select  = phi < phi_max_orbit
        phi_out = phi[select]

        u_out = np.interp(phi_out, xa, ua)
        v_out = np.interp(phi_out, xa, va)

        return OrbitSolution(
            phi=phi_out,
            u=u_out,
            v=v_out,
            status=status,
            n_iterations=n_iterations,
            solver="fd",
            info=dict(h=self.h, K=self.K, phi_max_guard=self.phi_max_guard),
        )
# ----------------------------------------------------------------------------
class PinnSlicer:
    '''
    Deploy a PINN trained on a single phi-slice [0, dphi] over an arbitrary
    phi range, by applying it recursively slice by slice.

    The PINN is evaluated at the points of the array phi. To do so it must
    also be evaluated at the segment boundaries x_i = (i+1)*dphi,
    i = 0, 1, 2, ... because the (u_i, v_i) pair at each of these points
    provides the initial conditions of segment i+1. A boundary point is
    therefore injected into each segment and dropped from the output.

    Parameters:
        pinn (nn.Module): a trained pinnslicer.nn.Solution.
        dphi (float): the slice width the PINN was trained on.
        strict (bool): raise instead of warning when the requested phi grid is
            too coarse for dphi (see `check_phi_resolution`).
        debug (bool): print the segment bookkeeping.

    Usage:
        pinn_solver = PinnSlicer(pinn, dphi=0.1)
        solution    = pinn_solver((u0, v0), phi)

    Notes:
        1. The default types in PyTorch are float32 and int64.
        2. torch.tensor inherits the fundamental type of its argument.
        3. All tensors are built from, and therefore live on, the same device
           as the PINN's own parameters.
    '''
    # u outside [u_min, u_max] means the photon left the region in which the
    # orbit equation is being solved.
    U_MIN = 0.0   # r = 1/u = infinity
    U_MAX = 1.0   # r = 1/u = Schwarzschild radius

    def __init__(self, pinn, dphi, strict=False, debug=False):
        if dphi <= 0:
            raise ValueError(f"/!\\ dphi must be positive, got {dphi}")

        self.pinn   = pinn
        self.dphi   = float(dphi)
        self.strict = strict
        self.debug  = debug

    @property
    def device(self):
        """The device the PINN's parameters live on."""
        return next(self.pinn.parameters()).device

    def __call__(self, y0, phi):
        '''
        y0:  initial conditions (u0, v0)
        phi: numpy array of points at which the solution should be returned.

        Returns:
            solution (OrbitSolution)
        '''
        dphi   = self.dphi
        debug  = self.debug
        device = self.device

        phi = validate_phi(phi)

        # Guard: a phi grid coarser than dphi leaves segments unvisited
        check_phi_resolution(phi, dphi, strict=self.strict)

        # 1. Find into which phi segment the specified phi values land.
        #    Floor with a tolerance rather than truncate: truncation puts a
        #    phi that is an exact multiple of dphi in the previous segment.
        segments  = np.floor(phi / dphi + PHI_TOL).astype(np.int64)
        nsegments = int(segments[-1]) + 1  # number of phi segments
        if debug:
            print(f'segments:     {segments[:15]}...{segments[-4:-1]}')

        # 2. Compute phi values modulo dphi, clipped to the trained domain
        #    [0, dphi], and convert to the PyTorch default float type
        phi_mod_dphis = np.clip(
            phi - segments * dphi, 0.0, dphi
        ).astype(np.float32)
        if debug:
            print(f'phi mod dphi: {phi_mod_dphis[:12]}...')

        # 3. Get segment boundary markers: markers[i] = segments[i+1] - segments[i]
        markers = np.diff(segments)
        if debug:
            print(f'markers:      {markers[:15]}...{markers[-4:-1]}')

        # 4. Find 1 + ordinal values of boundary markers, that is, the places
        #    (plus 1) at which we need to chop up phi_mod_dphis
        cuts = np.flatnonzero(markers) + 1
        if debug:
            print(f'cuts:         {cuts[:15]}...{cuts[-4:-1]}')

        # 5. Convert to tensors on the PINN's device. new_tensor inherits both
        #    the dtype and the device of its parent.
        phi_mod_dphis = torch.from_numpy(phi_mod_dphis).view(-1, 1).to(device)
        cuts_t = torch.from_numpy(cuts)
        dphi_t = phi_mod_dphis.new_tensor([dphi]).view(-1, 1)

        # 6. Change type of y0 to float32 to be consistent with PyTorch default
        y0 = torch.as_tensor(
            np.asarray(y0, dtype=np.float32).reshape(-1)
        ).to(device)

        # -------------------------------------------------------------
        # Apply the PINN recursively
        # -------------------------------------------------------------
        # U, V collect one chunk of the orbit per phi-segment; they are
        # concatenated at the end. The loop runs to completion unless a
        # spacetime boundary is reached, which is the only thing that can
        # stop it early -- hence COMPLETED as the starting assumption.
        U = []
        V = []
        status       = SolverStatus.COMPLETED
        n_iterations = 0

        # tensor_split cuts phi (mod dphi) at the segment boundaries, so each
        # `p` holds the requested points of one segment, all of them in the
        # [0, dphi] domain the PINN was trained on. The guard above has
        # already established that no segment is missing from this sequence.
        for p in torch.tensor_split(phi_mod_dphis, cuts_t):

            n_iterations += 1

            # For the phi (mod dphi) values of the current segment,
            # inject a point at phi = dphi
            phi_t = torch.cat([p, dphi_t]).requires_grad_(True)

            # Compute PINN solution at every point of segment
            u = self.pinn(phi_t, y0)      # u(phi)
            v = self.pinn.diff(u, phi_t)  # v(phi) = du/dphi

            # Squeeze away extraneous dimension
            u = u.squeeze()               # shape: (-1, 1) => (-1, )
            v = v.squeeze()

            # Check if we've reached infinity or the event horizon
            escaped  = u < self.U_MIN
            absorbed = u > self.U_MAX
            spacetime_boundary_reached = bool((escaped | absorbed).any())

            if spacetime_boundary_reached:
                # whichever boundary is crossed first is the one that
                # terminated the orbit; the PINN, being a smooth function,
                # happily continues past both, so u is clamped to the
                # boundary and v recomputed from the clamped u.
                status = (
                    SolverStatus.ESCAPED
                    if _first_true(escaped) <= _first_true(absorbed)
                    else SolverStatus.ABSORBED
                )

                u = torch.where(u < self.U_MIN, 1e-10, u)  # boundary at "infinity"
                u = torch.where(u > self.U_MAX, 1.0, u)    # boundary at horizon
                v = self.pinn.diff(u.view(-1, 1), phi_t).squeeze()  # update v(phi)

            # Concatenate orbital segments, but drop injected point
            U.append(u[:-1])
            V.append(v[:-1])

            if spacetime_boundary_reached:
                break

            # -------------------------------------------------------------
            # Spacetime boundary not reached, so apply recursion.
            # -------------------------------------------------------------
            # The right boundary values of the current phi segment become
            # the initial conditions y0 for the next phi segment. This is
            # why a right boundary point is injected into each phi segment.
            # u[-1], v[-1] are the values at the injected boundary point
            # phi = dphi, which is the start of the next segment. detach()
            # cuts the autograd graph here: the segments are solved one after
            # another, not backpropagated through as one long chain.
            y0 = torch.stack([u[-1], v[-1]]).detach()

        # Detach from computation tree, send to CPU, and convert to numpy array.
        u_out = torch.cat(U).detach().cpu().numpy()
        v_out = torch.cat(V).detach().cpu().numpy()

        # The orbit may have stopped early, in which case fewer points than
        # requested were computed; phi is trimmed to match.
        phi_out = phi[:len(u_out)]

        return OrbitSolution(
            phi=phi_out,
            u=u_out,
            v=v_out,
            status=status,
            n_iterations=n_iterations,
            solver="pinn",
            info=dict(dphi=dphi, n_segments=nsegments, device=str(device)),
        )

"""Mission animation (PyVista, offscreen): allocation, docking, transport.

Visual style follows the single-payload eight-agent demo: black starfield,
grey polyhedral payloads (convex hull of each payload's candidate attachment
points), orange cube agents with a nozzle and a blue exhaust plume whose
length follows the thrust, and a white wireframe of each payload's goal pose.
Everything is driven by the computed solution: docking-phase MPC trajectories
and the verified transport rollout (positions and attitudes).
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

from spacecraft_libraries.new_opts import so3_exp, so3_log, state_attitude_to_phi
from spacecraft_libraries.solvers.socp_inner import attitude_rollout

from .verify import rollout

ORANGE = "#d9822b"
NOZZLE = "#e8c87a"
PLUME = "#7fb2ff"
GREY = "#c8c8c8"


# ------------------------------------------------------------------ geometry
def hull_mesh(points):
    pts = np.asarray(points, float)
    hull = ConvexHull(pts)
    faces = np.hstack([[3, *s] for s in hull.simplices])
    return pv.PolyData(pts, faces)


def _frame(z):
    """Rotation whose third column is the unit vector z."""
    z = z / (np.linalg.norm(z) + 1e-12)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _tf(R=np.eye(3), t=np.zeros(3), scale=(1, 1, 1)):
    M = np.eye(4)
    M[:3, :3] = R @ np.diag(scale)
    M[:3, 3] = t
    return M


def _set_text(actor, text, corner=0):
    """Update a PyVista text actor (vtkTextActor or vtkCornerAnnotation)."""
    if hasattr(actor, "SetInput"):
        actor.SetInput(text)
    else:
        actor.SetText(corner, text)


def _slerp(R0, R1, s):
    return R0 @ so3_exp(s * so3_log(R0.T @ R1))


# ------------------------------------------------------------------ timeline
class Timeline:
    """Continuous-time access to the computed mission."""

    def __init__(self, sc, out, dp, dock, assignment):
        self.sc, self.dp, self.dock, self.asg = sc, dp, dock, assignment
        # transport: verified rollout positions, attitude from the final torques
        self.tr_r, self.tr_R, self.tr_U = [], [], []
        for j, res in enumerate(out["oracle"]["payloads"]):
            sp, bc = sc.sys_params_for(j, out["allocation"]["occupied"][j])
            self.tr_r.append(rollout(sp, bc, res["control"])["r"])
            Rs, _ = attitude_rollout(sp, bc, res["tau"])
            self.tr_R.append(Rs)
            self.tr_U.append(res["control"])           # (A_j, N, 3) body frame
        self.occ = out["allocation"]["occupied"]
        # index of each agent's thrust row within its payload's control array
        self.row = [self.occ[j].index(p) for (j, p) in assignment]
        self.u_dock_max = max(float(np.max(np.linalg.norm(a["u"], axis=1))) for a in dock["agents"])
        self.u_tr_max = max(float(np.max(np.linalg.norm(U, axis=2))) for U in self.tr_U)
        self.last_dir = [None] * len(assignment)

    @staticmethod
    def _interp(arr, s):
        """arr (K+1, ...) sampled at s in [0, K]."""
        k = int(np.clip(np.floor(s), 0, len(arr) - 2))
        a = s - k
        return (1 - a) * arr[k] + a * arr[k + 1], k

    def payload(self, j, phase, s):
        """(position, rotation) of payload j at step coordinate s of a phase."""
        if phase == "dock":
            r, _ = self._interp(self.dp.drift[j][0], s)
            return r, self.dp.R0[j]
        r, k = self._interp(self.tr_r[j], s)
        return r, _slerp(self.tr_R[j][k], self.tr_R[j][k + 1], s - k)

    def agent(self, a, phase, s):
        """(position, exhaust direction, thrust fraction) of agent a."""
        j, p = self.asg[a]
        if phase == "dock":
            ag = self.dock["agents"][a]
            r, k = self._interp(ag["r"], s)
            u = ag["u"][min(k, len(ag["u"]) - 1)]
            mag = float(np.linalg.norm(u))
            if mag > 1e-6:
                self.last_dir[a] = -u / mag        # exhaust opposes thrust
            d = self.last_dir[a]
            if d is None:                            # never fired yet: face the port
                pr, _, _ = self.dp.port(j, p)
                d = r - pr[-1]
                d /= np.linalg.norm(d) + 1e-12
            return r, d, mag / self.u_dock_max
        rp, R = self.payload(j, phase, s)
        q = self.sc.payloads[j].points[p]
        k = int(np.clip(np.floor(s), 0, self.sc.N - 1))
        mag = float(np.linalg.norm(self.tr_U[j][self.row[a], k]))
        return rp + R @ q, R @ (q / np.linalg.norm(q)), mag / self.u_tr_max


# ------------------------------------------------------------------ render
def render(sc, out, dp, dock, assignment, path, fps=30, size=(1920, 1080),
           durations=(3.0, 4.0, 9.0, 14.0, 4.0), agent_size=None, seed=0, title=None):
    """Write the mission MP4. durations = (intro, allocation, docking,
    transport, outro) in seconds."""
    import imageio_ffmpeg

    tl = Timeline(sc, out, dp, dock, assignment)
    rng = np.random.default_rng(seed)
    ext = max(P.extent for P in sc.payloads)
    s_ag = agent_size or 0.16 * ext

    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("black")
    pl.enable_anti_aliasing("ssaa")

    # starfield on a large sphere
    d = rng.standard_normal((4000, 3))
    stars = pv.PolyData(d / np.linalg.norm(d, axis=1, keepdims=True) * 2.0e4)
    pl.add_mesh(stars, color="white", point_size=2.0, render_points_as_spheres=False,
                lighting=False)

    # payloads and goal wireframes
    pay_actors, hulls = [], []
    for j, P in enumerate(sc.payloads):
        h = hull_mesh(P.points)
        hulls.append(h)
        pay_actors.append(pl.add_mesh(h.copy(), color=GREY, smooth_shading=False,
                                      ambient=0.25, diffuse=0.8, specular=0.1))
        goal = pl.add_mesh(h.copy(), style="wireframe", color="white", line_width=1.5,
                           lighting=False)
        goal.user_matrix = _tf(so3_exp(state_attitude_to_phi(P.xf)), P.xf.r)

    # open-point markers (allocation phase)
    open_actors = []
    for j, P in enumerate(sc.payloads):
        acts = []
        for p, q in enumerate(P.points):
            occupied = p in tl.occ[j]
            m = pv.Sphere(radius=0.22 * s_ag, center=(0, 0, 0))
            acts.append((p, q, pl.add_mesh(m, color=ORANGE if occupied else "white",
                                           lighting=False)))
        open_actors.append(acts)

    # agents: body cube, nozzle (exhaust along +z), plume of unit length
    cube = pv.Cube(x_length=s_ag, y_length=s_ag, z_length=s_ag)
    nozzle = pv.Cone(center=(0, 0, 0.62 * s_ag), direction=(0, 0, -1), height=0.35 * s_ag,
                     radius=0.32 * s_ag, resolution=24)
    plume = pv.Cone(center=(0, 0, 0.5), direction=(0, 0, -1), height=1.0,
                    radius=0.12, resolution=24)
    agents = []
    for _ in assignment:
        agents.append(dict(
            body=pl.add_mesh(cube.copy(), color=ORANGE, ambient=0.3, diffuse=0.8),
            nozzle=pl.add_mesh(nozzle.copy(), color=NOZZLE, ambient=0.3),
            plume=pl.add_mesh(plume.copy(), color=PLUME, opacity=0.85, lighting=False)))
    lines = [pl.add_mesh(pv.Line((0, 0, 0), (1, 0, 0)), color=ORANGE, line_width=1.5,
                         opacity=0.6, lighting=False) for _ in assignment]

    title_actor = pl.add_text("", position="upper_left", font_size=16, color="white",
                              font="times")
    caption = pl.add_text("", position="lower_edge", font_size=13, color="white", font="times")

    view = np.array([0.55, -0.75, 0.45])
    view /= np.linalg.norm(view)
    pl.camera.up = (0, 0, 1)
    pl.camera.view_angle = 30

    def place_payload(j, r, R):
        pay_actors[j].user_matrix = _tf(R, r)
        for p, q, act in open_actors[j]:
            act.user_matrix = _tf(np.eye(3), r + R @ q)

    def place_agent(a, r, d, frac, visible=True):
        F = _frame(d)
        for key in ("body", "nozzle"):
            agents[a][key].user_matrix = _tf(F, r)
            agents[a][key].visibility = visible
        L = max(frac, 0.0) * 6.0 * s_ag
        agents[a]["plume"].visibility = visible and L > 0.05 * s_ag
        agents[a]["plume"].user_matrix = _tf(F, r + F[:, 2] * 0.8 * s_ag,
                                             (L / 1.0 * 0.6 + 1e-6, L / 1.0 * 0.6 + 1e-6, L))

    n_intro, n_alloc, n_dock, n_tr, n_out = (int(round(x * fps)) for x in durations)
    total = n_intro + n_alloc + n_dock + n_tr + n_out
    head = title or f"{sc.M} payloads, {len(assignment)} agents"
    cap_text = {
        "intro": "Agents start free-flying; payloads must reach their goal poses (white wireframes).",
        "alloc": "Allocation: agents choose attachment points (orange) minimizing surrogate fuel.",
        "dock": "Docking: each agent runs fuel-optimal MPC with a line-of-sight approach cone.",
        "tr": "Transport: payloads follow collision-free corridors; plumes show agent thrust.",
        "out": "All payloads delivered: collision-free, verified with the full orbital dynamics.",
    }

    def when(f):
        if f < n_intro + n_alloc + n_dock:
            s = 0.0 if f < n_intro + n_alloc else (f - n_intro - n_alloc) / max(n_dock - 1, 1) * dp.N
            return "dock", s, ("intro" if f < n_intro else ("alloc" if f < n_intro + n_alloc else "dock"))
        s = min((f - n_intro - n_alloc - n_dock) / max(n_tr - 1, 1), 1.0) * sc.N
        return "tr", s, ("tr" if f < total - n_out else "out")

    # camera path: frame the payloads and agents of each frame, smoothed in time
    from scipy.ndimage import gaussian_filter1d
    centers, radii = [], []
    for f in range(total):
        phase, s, stage = when(f)
        pts = [tl.payload(j, phase, s)[0] for j in range(sc.M)]
        pts += [Timeline._interp(dock["agents"][a]["r"], s)[0] if phase == "dock"
                else pts[assignment[a][0]] for a in range(len(assignment))]
        if stage == "out":
            pts += [P.xf.r for P in sc.payloads]
        pts = np.asarray(pts)
        c = 0.5 * (pts.min(0) + pts.max(0))
        centers.append(c)
        radii.append(np.max(np.linalg.norm(pts - c, axis=1)) + 2.0 * ext)
    sig = 1.2 * fps
    centers = gaussian_filter1d(np.asarray(centers), sig, axis=0, mode="nearest")
    radii = gaussian_filter1d(np.asarray(radii), sig, mode="nearest")
    tl.last_dir = [None] * len(assignment)

    writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, quality=8,
                                         macro_block_size=8, pix_fmt_out="yuv420p")
    writer.send(None)
    try:
        for f in range(total):
            phase, s, stage = when(f)
            for j in range(sc.M):
                place_payload(j, *tl.payload(j, phase, s))
                for _, _, act in open_actors[j]:
                    act.visibility = stage in ("alloc", "dock")
            for a in range(len(assignment)):
                r, dvec, frac = tl.agent(a, phase, s)
                if stage in ("intro", "alloc"):
                    frac = 0.0
                if stage == "out":
                    frac = 0.0
                place_agent(a, r, dvec, frac)
                show_line = stage == "alloc"
                lines[a].visibility = show_line
                if show_line:
                    j, p = assignment[a]
                    pr, _, _ = dp.port(j, p)
                    lines[a].mapper.SetInputData(pv.Line(r, pr[0]))
            t_phys = (-dp.T + s * dp.dt) if phase == "dock" else s * sc.dt
            _set_text(title_actor, f"{head}\nt = {t_phys:+.0f} s", corner=2)
            _set_text(caption, cap_text[stage], corner=0)
            az = 20.0 * f / total
            vr = so3_exp(np.array([0, 0, np.deg2rad(az)])) @ view
            dist = radii[f] / np.tan(np.deg2rad(15)) * 1.1
            pl.camera.focal_point = centers[f]
            pl.camera.position = centers[f] + dist * vr
            pl.camera.clipping_range = (max(dist - 3 * radii[f], 1.0), dist + 3.0e4)
            pl.render()
            writer.send(np.ascontiguousarray(pl.screenshot(return_img=True)[:, :, :3]))
    finally:
        writer.close()
        pl.close()
    return path

"""
Shallow Water Equations solver using Taichi.

Uses a *partial* Lax-Friedrichs scheme with the η-gradient (water surface
elevation) formulation.  This is naturally *well-balanced* – a lake-at-rest
(u=v=0, η=h+b=const) is preserved exactly regardless of bed topography.

The momentum equation:
    ∂(hu)/∂t + ∂(hu²/h)/∂x + g h ∂η/∂x + ∂(huv/h)/∂y = 0
    where  η = h + b  (water surface elevation)

Standard Lax-Friedrichs replaces the centre value with the full neighbour
average, which is very stable but introduces huge numerical diffusion that
kills vortices.  We use a *partial* average:

    q_avg = (1 − α) · q_centre  +  α · ¼(q_e + q_w + q_n + q_s)

With α < 1 the numerical diffusion is reduced by factor α, allowing wake
vortices and recirculation zones to survive.  Stability requires
  c_max · dt / dx  ≤  √(α / 2)     (2-D CFL for partial L-F)

Supports both CPU and GPU backends via Taichi.
"""
import taichi as ti
import numpy as np

# Cells with depth below this are treated as dry (zero state)
DRY_THRESH = 1e-3
# Hard cap on velocity magnitude (m/s)
MAX_SPEED = 4.0
# Lax-Friedrichs blending factor (1.0 = full L-F, lower = less diffusion)
# CFL stability limit: c_max*dt/dx ≤ sqrt(LF_ALPHA/2)
# With MAX_SPEED=4, h0=0.5: c_max≈6.2, dt=0.002, dx=0.039 → 0.318, √(0.4/2)=0.447 ✓
LF_ALPHA = 0.4


def init_taichi(use_gpu: bool = False) -> str:
    """Initialize (or re-initialize) Taichi with the requested backend."""
    try:
        ti.reset()
    except Exception:
        pass

    if use_gpu:
        try:
            ti.init(arch=ti.gpu)
            return "gpu"
        except Exception:
            ti.init(arch=ti.cpu)
            return "cpu (GPU unavailable)"
    else:
        ti.init(arch=ti.cpu)
        return "cpu"


@ti.data_oriented
class SWESolver:
    """Lax-Friedrichs SWE solver with η-gradient well-balancing.

    State variables (conserved):
        h  – water depth above bed
        hu – x-momentum  (h · u)
        hv – y-momentum  (h · v)

    Bed elevation *b* encodes obstacles.  Cells with b >= h0 are solid.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 dt: float, g: float, nu: float, h0: float, u0: float):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.dt = dt
        self.g = g
        self.nu = nu
        self.h0_val = h0
        self.u0_val = u0

        # conserved variables (double-buffered)
        self.h  = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.hu = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.hv = ti.field(dtype=ti.f32, shape=(nx, ny))

        self.h_new  = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.hu_new = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.hv_new = ti.field(dtype=ti.f32, shape=(nx, ny))

        # bed elevation
        self.b = ti.field(dtype=ti.f32, shape=(nx, ny))

        # derived (diagnostic) quantities
        self.vx       = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vy       = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.speed    = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vorticity = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.pressure = ti.field(dtype=ti.f32, shape=(nx, ny))

    # ------------------------------------------------------------------ #
    #  public helpers                                                      #
    # ------------------------------------------------------------------ #
    def set_bed(self, b_np: np.ndarray):
        self.b.from_numpy(b_np.astype(np.float32))

    def step(self):
        """Advance one time-step (Lax-Friedrichs + η-gradient)."""
        self._lax_friedrichs_step()
        self._apply_limiters()
        self._apply_bc_inflow()
        self._apply_bc_outflow()
        self._apply_bc_walls()
        self._swap()

    def get_frame_data(self) -> dict:
        self._compute_derived()
        return {
            "h": self.h.to_numpy(),
            "vx": self.vx.to_numpy(),
            "vy": self.vy.to_numpy(),
            "speed": self.speed.to_numpy(),
            "vorticity": self.vorticity.to_numpy(),
            "pressure": self.pressure.to_numpy(),
        }

    # ------------------------------------------------------------------ #
    #  safe velocity                                                       #
    # ------------------------------------------------------------------ #
    @ti.func
    def _safe_vel(self, mom: ti.f32, h_val: ti.f32) -> ti.f32:
        result = 0.0
        if h_val > DRY_THRESH:
            result = mom / h_val
            result = ti.min(ti.max(result, -MAX_SPEED), MAX_SPEED)
        return result

    # ------------------------------------------------------------------ #
    #  initialization                                                      #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def initialize(self):
        h0 = self.h0_val
        u0 = self.u0_val
        for i, j in self.h:
            bed = self.b[i, j]
            if bed < h0:
                depth = h0 - bed
                self.h[i, j] = depth
                self.hu[i, j] = depth * u0
                self.hv[i, j] = 0.0
            else:
                self.h[i, j] = 0.0
                self.hu[i, j] = 0.0
                self.hv[i, j] = 0.0

    # ------------------------------------------------------------------ #
    #  main Lax-Friedrichs step with η-gradient                            #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def _lax_friedrichs_step(self):
        g  = self.g
        dx = self.dx
        dy = self.dy
        dt = self.dt
        nu = self.nu
        h0 = self.h0_val
        nx = self.nx
        ny = self.ny

        for i, j in self.h:
            if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1:
                continue
            if self.b[i, j] >= h0:
                self.h_new[i, j] = 0.0
                self.hu_new[i, j] = 0.0
                self.hv_new[i, j] = 0.0
                continue

            # ---------- gather neighbours ----------------------------- #
            h_c  = self.h[i, j];     hu_c = self.hu[i, j];     hv_c = self.hv[i, j]
            h_e  = self.h[i+1, j];   hu_e = self.hu[i+1, j];   hv_e = self.hv[i+1, j]
            h_w  = self.h[i-1, j];   hu_w = self.hu[i-1, j];   hv_w = self.hv[i-1, j]
            h_n  = self.h[i, j+1];   hu_n = self.hu[i, j+1];   hv_n = self.hv[i, j+1]
            h_s  = self.h[i, j-1];   hu_s = self.hu[i, j-1];   hv_s = self.hv[i, j-1]

            # neighbour velocities (safe)
            u_e = self._safe_vel(hu_e, h_e)
            u_w = self._safe_vel(hu_w, h_w)
            v_n = self._safe_vel(hv_n, h_n)
            v_s = self._safe_vel(hv_s, h_s)

            # ---------- Partial Lax-Friedrichs average ----------------- #
            # Blend centre value with neighbour average to control
            # numerical diffusion.  alpha=1 is pure L-F (max diffusion),
            # alpha<1 preserves more of the centre → sharper vortices.
            alpha = LF_ALPHA
            nbr_h  = 0.25 * (h_e + h_w + h_n + h_s)
            nbr_hu = 0.25 * (hu_e + hu_w + hu_n + hu_s)
            nbr_hv = 0.25 * (hv_e + hv_w + hv_n + hv_s)
            h_avg  = (1.0 - alpha) * h_c  + alpha * nbr_h
            hu_avg = (1.0 - alpha) * hu_c + alpha * nbr_hu
            hv_avg = (1.0 - alpha) * hv_c + alpha * nbr_hv

            # ---------- mass flux (standard) -------------------------- #
            self.h_new[i, j] = h_avg \
                - dt / (2.0 * dx) * (hu_e - hu_w) \
                - dt / (2.0 * dy) * (hv_n - hv_s)

            # ---------- water-surface elevation η = h + b ------------- #
            eta_e = h_e + self.b[i+1, j]
            eta_w = h_w + self.b[i-1, j]
            eta_n = h_n + self.b[i, j+1]
            eta_s = h_s + self.b[i, j-1]

            deta_dx = (eta_e - eta_w) / (2.0 * dx)
            deta_dy = (eta_n - eta_s) / (2.0 * dy)

            # ---------- advection-only momentum fluxes ---------------- #
            dFhu_adv = hu_e * u_e - hu_w * u_w
            dGhu_adv = hu_n * v_n - hu_s * v_s
            dFhv_adv = hv_e * u_e - hv_w * u_w
            dGhv_adv = hv_n * v_n - hv_s * v_s

            # ---------- viscous diffusion ----------------------------- #
            lap_hu = (hu_e + hu_w + hu_n + hu_s - 4.0 * hu_c) / (dx * dx)
            lap_hv = (hv_e + hv_w + hv_n + hv_s - 4.0 * hv_c) / (dx * dx)

            # ---------- momentum update ------------------------------- #
            # L-F average  –  advection flux  –  g h ∂η/∂x  +  viscosity
            hu_val = hu_avg \
                - dt / (2.0 * dx) * dFhu_adv \
                - dt / (2.0 * dy) * dGhu_adv \
                - dt * g * h_c * deta_dx \
                + dt * nu * lap_hu

            hv_val = hv_avg \
                - dt / (2.0 * dx) * dFhv_adv \
                - dt / (2.0 * dy) * dGhv_adv \
                - dt * g * h_c * deta_dy \
                + dt * nu * lap_hv

            h_val = self.h_new[i, j]

            # ---------- sponge layers (absorb waves near boundaries) --- #
            # Scale sponge width to ~6% of grid dimension (min 8 cells)
            sponge_w = ti.max(ny // 16, 8)
            r = 0.0
            if j < sponge_w:
                r = ti.cast(sponge_w - j, ti.f32) / ti.cast(sponge_w, ti.f32)
            elif j >= ny - 1 - sponge_w:
                r = ti.cast(j - (ny - 1 - sponge_w), ti.f32) / ti.cast(sponge_w, ti.f32)
            # right (outflow) sponge – wider to absorb downstream waves
            sponge_r = ti.max(nx // 16, 12)
            if i >= nx - 1 - sponge_r:
                r2 = ti.cast(i - (nx - 1 - sponge_r), ti.f32) / ti.cast(sponge_r, ti.f32)
                r = ti.max(r, r2)
            if r > 0.0:
                r = r * r * 0.5       # quadratic ramp, strong damping
                h_val  = (1.0 - r) * h_val  + r * h0
                hu_val = (1.0 - r) * hu_val + r * h0 * self.u0_val
                hv_val = (1.0 - r) * hv_val
                self.h_new[i, j] = h_val

            self.hu_new[i, j] = hu_val
            self.hv_new[i, j] = hv_val

    # ------------------------------------------------------------------ #
    #  limiters                                                            #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def _apply_limiters(self):
        h0 = self.h0_val
        u0 = self.u0_val
        for i, j in self.h_new:
            h_val  = self.h_new[i, j]
            hu_val = self.hu_new[i, j]
            hv_val = self.hv_new[i, j]
            # NaN guard
            is_bad = (h_val != h_val) or (hu_val != hu_val) or (hv_val != hv_val)
            if is_bad or h_val < DRY_THRESH:
                self.h_new[i, j]  = 0.0
                self.hu_new[i, j] = 0.0
                self.hv_new[i, j] = 0.0
            else:
                # emergency depth clamp (prevents runaway feedback)
                if h_val > 4.0 * h0:
                    self.h_new[i, j]  = h0
                    self.hu_new[i, j] = h0 * u0
                    self.hv_new[i, j] = 0.0
                else:
                    u = hu_val / h_val
                    v = hv_val / h_val
                    spd = ti.sqrt(u * u + v * v)
                    if spd > MAX_SPEED:
                        factor = MAX_SPEED / spd
                        self.hu_new[i, j] *= factor
                        self.hv_new[i, j] *= factor

    # ------------------------------------------------------------------ #
    #  boundary conditions (split to avoid corner race)                    #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def _apply_bc_inflow(self):
        h0 = self.h0_val
        u0 = self.u0_val
        for j in range(self.ny):
            self.h_new[0, j]  = h0
            self.hu_new[0, j] = h0 * u0
            self.hv_new[0, j] = 0.0

    @ti.kernel
    def _apply_bc_outflow(self):
        nx = self.nx
        for j in range(self.ny):
            self.h_new[nx - 1, j]  = self.h_new[nx - 2, j]
            self.hu_new[nx - 1, j] = self.hu_new[nx - 2, j]
            self.hv_new[nx - 1, j] = self.hv_new[nx - 2, j]

    @ti.kernel
    def _apply_bc_walls(self):
        nx = self.nx
        ny = self.ny
        for i in range(1, nx - 1):
            self.h_new[i, 0]  = self.h_new[i, 1]
            self.hu_new[i, 0] = self.hu_new[i, 1]
            self.hv_new[i, 0] = -self.hv_new[i, 1]
        for i in range(1, nx - 1):
            self.h_new[i, ny - 1]  = self.h_new[i, ny - 2]
            self.hu_new[i, ny - 1] = self.hu_new[i, ny - 2]
            self.hv_new[i, ny - 1] = -self.hv_new[i, ny - 2]

    # ------------------------------------------------------------------ #
    #  swap                                                                #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def _swap(self):
        for i, j in self.h:
            self.h[i, j]  = self.h_new[i, j]
            self.hu[i, j] = self.hu_new[i, j]
            self.hv[i, j] = self.hv_new[i, j]

    # ------------------------------------------------------------------ #
    #  derived quantities                                                  #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def _compute_derived(self):
        g  = self.g
        dx = self.dx
        dy = self.dy
        nx = self.nx
        ny = self.ny

        for i, j in self.h:
            h_val = self.h[i, j]
            u = 0.0
            v = 0.0
            if h_val > DRY_THRESH:
                u = ti.min(ti.max(self.hu[i, j] / h_val, -MAX_SPEED), MAX_SPEED)
                v = ti.min(ti.max(self.hv[i, j] / h_val, -MAX_SPEED), MAX_SPEED)
            self.vx[i, j] = u
            self.vy[i, j] = v
            self.speed[i, j] = ti.sqrt(u * u + v * v)
            self.pressure[i, j] = 0.5 * g * h_val * h_val

        for i, j in self.vorticity:
            if 0 < i < nx - 1 and 0 < j < ny - 1:
                dvdx = (self.vy[i+1, j] - self.vy[i-1, j]) / (2.0 * dx)
                dudy = (self.vx[i, j+1] - self.vx[i, j-1]) / (2.0 * dy)
                self.vorticity[i, j] = dvdx - dudy
            else:
                self.vorticity[i, j] = 0.0
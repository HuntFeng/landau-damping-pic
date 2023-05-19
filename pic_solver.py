import shutil
import os

import numpy as np
import taichi as ti
from tqdm import tqdm
from scipy.stats import rv_continuous


@ti.dataclass
class Particle:
    w: ti.f64 # weight
    x: ti.f64 # position
    v: ti.f64 # velocity
    q: ti.f64 # charge
    m: ti.f64 # mass

    @ti.func
    def advance(self, E:ti.f64,B:ti.f64,dt:ti.f64):
        q, m = self.q, self.m
        x, v = self.x, self.v

        v += dt * (q/m)*(E+v*B)
        x += dt * v

        self.v = v
        self.x = x
        

@ti.data_oriented
class Fields:
    def __init__(self, x: np.array) -> None:
        """ 
        Initialize fields 
        
        Input:
            x: mesh
        """
        nx = x.shape[0]
        dx = x[1] - x[0]

        self.nx = nx

        self.E = ti.field(dtype=float, shape=(nx,)) # electric field
        self.B = ti.field(dtype=float, shape=(nx,)) # magnetic field

        self.n = ti.field(dtype=float, shape=(nx,)) # number density field
        self.rho = ti.field(dtype=float, shape=(nx,)) # charge density field
        self.u = ti.field(dtype=float, shape=(nx,)) # bulk velocity field
        self.p = ti.field(dtype=float, shape=(nx,)) # pressure field
        self.q = ti.field(dtype=float, shape=(nx,)) # heat flux field

        self.v_sqr_avg = ti.field(dtype=float, shape=(nx,)) # intermediate field for p and q
        self.v_cube_avg = ti.field(dtype=float, shape=(nx,)) # intermediate field for p and q

        # useful matrices
        self.Laplacian = np.diag(np.ones(nx-1), k=-1) \
            + np.diag(np.ones(nx-1), k=1) \
            - 2*np.diag(np.ones(nx), k=0)
        self.Laplacian[0,-1] = 1
        self.Laplacian[-1,0] = 1
        self.Laplacian /= dx**2

        self.grad = np.diag(-np.ones(nx-1), k=-1) + np.diag(np.ones(nx-1), k=1)
        self.grad[0,-1] = -1
        self.grad[-1,0] = 1
        self.grad /= 2*dx

    def compute_E(self):
        """ Compute E field """
        # we need to solve potential, then differentiate it to get E
        # since the div operator is singular with periodic bounary
        # so we cannot directly sove divE = rho
        V = np.linalg.solve(self.Laplacian, -self.rho.to_numpy())
        E = -self.grad@V
        self.E.from_numpy(E)

    def compute_B(self):
        """ Compute B field. In 1D, no need to compute B field since vxB=0 """
        B = np.zeros(self.nx)
        self.B.from_numpy(B)

@ti.data_oriented
class Solver:
    def __init__(self) -> None:
        print("Preparing initial conditions >>>>>>>>>>>>>>>>>>")
        """ Initial data, mesh """
        # parameters (TABLE I Qin. et al 2022)
        k1 = 0.6
        k2 = 1.2
        A1 = 0.05
        A2 = 0.4
        phi = 0.38716
        L = 2*np.pi/k1
        
        n0 = 1
        N_sim = int(8e4)
        N_real = n0*L
        mpw = N_real/N_sim # macroparticle weight

        self.L = L
        self.n0 = n0

        # mesh and fields
        nx = 256 # number of cells
        x = np.linspace(0, L, nx, endpoint=False) # periodic grid points
        self.x = ti.field(dtype=float, shape=(nx,))

        self.nx = nx
        self.dx = x[1]-x[0]
        self.x.from_numpy(x)
        self.F = Fields(x)

        # particles
        class Density(rv_continuous):
            def _pdf(self, x):
                """ Density distribution (Normalized) """
                return (1 + A1*np.cos(k1*x) + A2*np.cos(k2*x+phi))/L
        density = Density(a=0, b=L)
        xe = density.rvs(size=N_sim) # draw position from the density distribution
        ve = np.random.normal(0, 6, N_sim)

        self.p = Particle.field(shape=(N_sim,))
        self.p.w.from_numpy(mpw*np.ones(N_sim))
        self.p.x.from_numpy(xe)
        self.p.v.from_numpy(ve)
        self.p.q.from_numpy(-np.ones(N_sim))
        self.p.m.from_numpy(np.ones(N_sim))

    @ti.func
    def gather(self, field: ti.template(), xp:float) -> float:
        """ 
        Gatter mesh quantities to particle position. Interpolation.

        Input:
            xp: particle position
        """
        x = ti.static(self.x)
        nx, dx = self.nx, self.dx

        i = int(xp/dx)
        ip1 = (i+1)%nx
        Dx = (xp - x[i])/dx

        return field[i]*(1.0-Dx)  + field[ip1]*Dx
    
    @ti.func
    def scatter(self, field: ti.template(), value: float, xp: float):
        """ 
        Scatter the particle quantities to mesh position. Extrapolation.
        
        Input:
            field: the field we need to compute
            value: the field value
            xp: particle position
        """
        x = ti.static(self.x)
        nx, dx = self.nx, self.dx

        i = int(xp/dx)
        ip1 = (i+1)%nx
        Dx = (xp - x[i])/dx

        field[i] += value * (1.0-Dx)
        field[ip1] += value * Dx
    
    @ti.kernel
    def compute_fields(self):
        """ Compute all fields except E and B using the positions and velocities of particles  """
        p = ti.static(self.p)
        dx = self.dx
        n0 = self.n0

        # clear fields
        for i in self.F.n:
            self.F.n[i] = 0.0
            self.F.u[i] = 0.0
            self.F.p[i] = 0.0
            self.F.q[i] = 0.0

            self.F.rho[i] = 0.0

        # number density field
        for k in p:
            # n = num particles at each mesh point / cell width
            self.scatter(self.F.n, p[k].w / dx, p[k].x)
        # charge density
        for i in self.F.rho:
            self.F.rho[i] = n0 - self.F.n[i]

        # fluid velocity field
        for k in p:
            # velocity sum at each mesh point
            self.scatter(self.F.u, p[k].w*p[k].v, p[k].x)
        for i in self.F.u:
            self.F.u[i] /= self.F.n[i]

        sum_w = 0.0 # sum of all weights
        for k in p:
            sum_w += p[k].w
            self.scatter(self.F.v_sqr_avg, p[k].m*p[k].w*p[k].v**2, p[k].x)
            self.scatter(self.F.v_cube_avg, p[k].m*p[k].w*p[k].v**3, p[k].x)
        for i in self.F.u:
            self.F.p[i] = self.F.v_sqr_avg[i]/sum_w - self.F.u[i]**2
            self.F.q[i] = self.F.v_cube_avg[i]/sum_w - 3*self.F.v_sqr_avg[i]*self.F.u[i] + 2*self.F.u[i]**3

    @ti.kernel
    def advance_particles(self, dt:float):
        """ Update particle positions and velocities """
        p = ti.static(self.p)
        L = self.L

        # scatter fields to particle position
        for k in p:
            E = self.gather(self.F.E, p[k].x)
            B = self.gather(self.F.B, p[k].x)
            p[k].advance(E,B,dt)

            # periodic boundary
            if (p[k].x > L or p[k].x < 0):
                p[k].x = p[k].x % L

    def update(self, dt:float):
        """ Update particle position and velocities """
        self.compute_fields()
        self.F.compute_E()
        self.F.compute_B()
        self.advance_particles(dt)

    @ti.kernel
    def rewind_velocity(self, dt:float):
        p = ti.static(self.p)
        for k in p:
            E = self.gather(self.F.E, p[k].x)
            p[k].v -= (p[k].q/p[k].m)*E*dt/2

    def run(self):
        dt = 0.001 # normalized to w_{pe}^{-1}
        tf = 2 # normalized to w_{pe}^{-1}
        total_frame = int(tf/dt)
        print("Rewinding Velocity")
        self.compute_fields()
        self.F.compute_E()
        self.rewind_velocity(dt)
        print("Start >>>>>>>>>>>>>>>>>>")
        for frame in tqdm(range(total_frame)):
            self.update(dt)
            save_fields(f"{datadir}/{frame:04d}", self.F)


def save_fields(filename:str, F: Fields):
        """ Save fields """
        np.savez(
            filename, 
            n=F.n.to_numpy(),
            u=F.u.to_numpy(),
            p=F.u.to_numpy(),
            q=F.u.to_numpy(),
            E=F.E.to_numpy(),
        )

if __name__ == '__main__':
    # create/clear folder to store data
    datadir = "data_pic"
    if (os.path.exists(datadir)):
        shutil.rmtree(datadir)
    os.mkdir(datadir)

    # initialize GPU, if no GPU available, fallback to CPU
    ti.init(ti.gpu, default_fp=ti.f64)

    # initialize PIC solver and run
    solver = Solver()
    solver.run()

# This is a model library for calcium signaling models used when
# collaborating with the Zartman Lab. The purpose of this library is to
# provide a common interface for loading and managing models in the repository
# instead of having each model be hardcoded in each script.

# Import necessary libraries
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.sparse import bmat, diags, csr_matrix, csc_matrix, issparse, eye
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors, animation
import seaborn as sns
import pyomo.environ as pyo
import pyomo.dae as dae

from numpy import dot, multiply, diag, power
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from matplotlib import animation
from IPython.display import display, HTML


# =============== Two Pool Model =========================
# two pool model in single cell with calcium in cytosol and ER
# Dupont and Goldbeter 1993


# =============== Pouch Model =========================
# multicell model from Soundajarran et al. 2021
# Tissue level model with gap junctions

# Original pouch model from Soundajarran et al. 2021

# Add later
    
# Model coded to be used in scipy.integrate.solve_ivp

    

# =============== Additional Models =========================

class PouchModelScipy_dense:
    def __init__(self, 
                 laplacian_matrix, 
                 dt=0.2, 
                 sim_time=3600, 
                 param_dict=None,
                 t_eval_boolean=False,
                 random_seed=12345):
        self.dt = dt
        self.sim_time = sim_time
        self.sim_number = random_seed
        if t_eval_boolean:
            self.t_eval = np.arange(0, sim_time, dt)

        # Initialize laplacian matrix
        self.laplacian = laplacian_matrix

        # Number of cells
        self.n_cells = laplacian_matrix.shape[0]

        # Default parameters
        self.default_params = {
            'D_p': 0.005,
            'D_c_ratio': 0.1,
            "K_PLC": 0.2,
            "K_5": 0.66,
            "k_1": 1.11,
            "k_a": 0.08,
            "k_p": 0.13,
            "k_2": 0.0203,
            "V_SERCA": 0.9,
            "K_SERCA": 0.1,
            "c_tot": 2,
            "beta": 0.185,
            "k_i": 0.4,
            "tau_max": 800,
            "k_tau": 1.5,
            "lower": 0.2,
            "upper": 0.4,
            "frac": 0.01538462, # 3 cells in 195 cell pouch
            'alpha': 0.025,  # Optogenetic stimulation factor
            'Imax': 1.0,  # Maximum intensity of optogenetic stimulation
            # 'VPLC': 1.0, # VPLC must be defined here as a default value
        }

        # Update default parameters with user-provided dictionary
        if param_dict:
            self.params = self.default_params.copy()
            for key, value in param_dict.items():
                if isinstance(value, (list, np.ndarray)):
                    # Ensure the array size matches the number of cells
                    if len(value) != self.n_cells:
                        raise ValueError(f"Parameter '{key}' has incorrect size. Expected {self.n_cells}, got {len(value)}.")
                    self.params[key] = np.array(value)
                else:
                    self.params[key] = value
        else:
            self.params = self.default_params

        # Convert single-value parameters to arrays for vectorized operations
        for key, value in self.params.items():
            if not isinstance(value, np.ndarray):
                self.params[key] = np.full(self.n_cells, value)

    def pouch_odes(self, t, y, theta, optogenetic=False):
        n = self.n_cells
        # y is (3*n, k)
        Ca, IP3, R = y[:n, :], y[n:2*n, :], y[2*n:, :]
        
        # Helper for broadcasting: converts (n,) to (n, 1)
        def b(val): return val.reshape(-1, 1) if isinstance(val, np.ndarray) else val


        # 1. Diffusion (Matrix @ 2D Dense is very fast)
        ca_lap = (self.laplacian @ Ca) * b(theta['D_p'] * theta['D_c_ratio'])
        ip3_lap = (self.laplacian @ IP3) * b(theta['D_p'])

        # 2. Reaction Terms (All element-wise)
        S_alg = (b(theta['c_tot']) - Ca) / b(theta['beta'])
        
        # J_IP3R: Using powers/division is fine on (n, k) matrices
        denom = (b(theta['k_a']) + Ca) * (b(theta['k_p']) + IP3)
        J_IP3R = b(theta['k_1']) * ((R * Ca * IP3) / denom)**3
        
        J_SERCA = b(theta['V_SERCA']) * Ca**2 / (Ca**2 + b(theta['K_SERCA'])**2)
        
        # ODEs
        dCa_dt = ca_lap + (J_IP3R + b(theta['k_2'])) * (S_alg - Ca) - J_SERCA
        if optogenetic: dCa_dt += b(theta['alpha'] * theta['Imax'])
        
        V_PLC = b(theta['VPLC']) * (Ca**2) / (Ca**2 + b(theta['K_PLC'])**2)
        dIP3_dt = ip3_lap + V_PLC - b(theta['K_5']) * IP3
        
        dR_dt = ((b(theta['k_tau'])**4 + Ca**4) / (b(theta['tau_max']) * b(theta['k_tau'])**4)) * \
                (1 - R * (b(theta['k_i']) + Ca) / b(theta['k_i']))

        return np.vstack([dCa_dt, dIP3_dt, dR_dt])

    def simulate(self, theta_dict=None, y0=None, int_method = 'Radau', opto_bool=False, get_initiator_cells=False):
        np.random.seed(self.sim_number) # Set the seed for reproducibility (keep initiator cells consistent each run)
           
        if y0 is None:
            # Initial values: Ca, IP3, S, R
            Ca0 = np.zeros(self.n_cells)
            IP30 = np.zeros(self.n_cells)
            R0 = np.random.uniform(0.5, 0.7, self.n_cells)
            y0 = np.concatenate([Ca0, IP30, R0])

        if theta_dict is None:
            # If no theta_dict is provided, use the instance's pre-processed params
            theta_dict = self.params

        # Initialize VPLC values based on the user's request
        self.params['VPLC'] = np.random.uniform(self.params['lower'], self.params['upper'], self.n_cells)
        # Choose which cells are initiator cells based on 'frac'
        stimulated_cell_idxs = np.random.choice(self.n_cells, int(self.params['frac'][0] * self.n_cells))
        self.stimulated_cell_idxs = stimulated_cell_idxs
        # Set the VPLC of initiator cells to be random uniformly distributed between 1.3 and 1.5
        self.params['VPLC'][stimulated_cell_idxs] = np.random.uniform(1.3, 1.5, len(stimulated_cell_idxs))

        if get_initiator_cells:
            return stimulated_cell_idxs
        
        # NOTE: If a new theta_dict is passed here, it should be pre-processed
        # by the user to be a dictionary of arrays, if per-cell variation is desired.

        sol = solve_ivp(
            fun=lambda t, y: self.pouch_odes(t, y, self.params, optogenetic=opto_bool), 
            t_span=(0, self.sim_time),
            y0=y0,
            method=int_method,
            vectorized=True,
            rtol=1e-4, # Often enough for calcium waves
            atol=1e-7
            )
        return sol

## =============== End of Model =========================

class PouchModelScipy_sparse_v2:
    def __init__(self,
                 laplacian_matrix,
                 dt=0.2,
                 sim_time=3600,
                 param_dict=None,
                 t_eval_boolean=False,
                 random_seed=12345,
                 vertices=None,
                 size='small',
                 save=False,
                 saveName='default'):
        self.dt = dt
        self.sim_time = sim_time
        self.sim_number = random_seed
        self.sol = None  # Stores the most recent solve_ivp result from simulate()

        # Plotting / saving attributes (used by make_animation, draw_profile, etc.)
        self.new_vertices = vertices  # Cell polygon vertices for the selected disc size
        self.size = size
        self.save = save
        self.saveName = saveName
        if t_eval_boolean:
            self.t_eval = np.arange(0, sim_time, dt)

        # Initialize laplacian matrix
        laplacian_dense = laplacian_matrix

        # Convert Laplacian to sparse CSR format once.
        # If you already have a dense Laplacian, pass it in as laplacian_dense
        # or build it directly as sparse outside this class.
        if laplacian_dense is not None:
            self.laplacian = csr_matrix(laplacian_dense)
        else:
            raise ValueError("Provide a Laplacian (dense or sparse).")
        
        self.laplacian = laplacian_matrix

        # Number of cells
        self.n_cells = laplacian_matrix.shape[0]

        # Default parameters
        self.default_params = {
            'D_p': 0.005,
            'D_c_ratio': 0.1,
            "K_PLC": 0.2,
            "K_5": 0.66,
            "k_1": 1.11,
            "k_a": 0.08,
            "k_p": 0.13,
            "k_2": 0.0203,
            "V_SERCA": 0.9,
            "K_SERCA": 0.1,
            "c_tot": 2,
            "beta": 0.185,
            "k_i": 0.4,
            "tau_max": 800,
            "k_tau": 1.5,
            "lower": 0.2,
            "upper": 0.4,
            "frac": 0.01538462, # 3 cells in 195 cell pouch
            'alpha': 0.025,  # Optogenetic stimulation factor
            'Imax': 1.0,  # Maximum intensity of optogenetic stimulation
            # 'VPLC': 1.0, # VPLC must be defined here as a default value
        }

        # Update default parameters with user-provided dictionary
        if param_dict:
            self.params = self.default_params.copy()
            for key, value in param_dict.items():
                if isinstance(value, (list, np.ndarray)):
                    # Ensure the array size matches the number of cells
                    if len(value) != self.n_cells:
                        raise ValueError(f"Parameter '{key}' has incorrect size. Expected {self.n_cells}, got {len(value)}.")
                    self.params[key] = np.array(value)
                else:
                    self.params[key] = value
        else:
            self.params = self.default_params

        # Convert single-value parameters to arrays for vectorized operations
        for key, value in self.params.items():
            if not isinstance(value, np.ndarray):
                self.params[key] = np.full(self.n_cells, value)

        # Pre-calculate sparsity mask once
        self._jac_sparsity = self._get_sparsity_mask()

    def _get_sparsity_mask(self):
        # connectivity = Adjacency + Identity
        # Since Laplacian has the same non-zero structure as (Adj + I)
        conn = (self.laplacian != 0).astype(int)
        I = eye(self.n_cells, format='csr')
        
        # 3x3 Block matrix: Ca, IP3, R
        # R only depends on local Ca (the I blocks), others diffuse
        mask = bmat([
            [conn, conn, conn], # dCa/dt
            [conn, conn, conn], # dIP3/dt
            [I,    None, I   ]  # dR/dt (R doesn't depend on IP3)
        ], format='csr')
        return mask

    def pouch_odes(self, t, y, theta, optogenetic=False):
        n = self.n_cells
        # y is (3*n, k)
        Ca, IP3, R = y[:n, :], y[n:2*n, :], y[2*n:, :]
        
        # Helper for broadcasting: converts (n,) to (n, 1)
        def b(val): return val.reshape(-1, 1) if isinstance(val, np.ndarray) else val


        # 1. Diffusion (Matrix @ 2D Dense is very fast)
        ca_lap = (self.laplacian @ Ca) * b(theta['D_p'] * theta['D_c_ratio'])
        ip3_lap = (self.laplacian @ IP3) * b(theta['D_p'])

        # 2. Reaction Terms (All element-wise)
        S_alg = (b(theta['c_tot']) - Ca) / b(theta['beta'])
        
        # J_IP3R: Using powers/division is fine on (n, k) matrices
        denom = (b(theta['k_a']) + Ca) * (b(theta['k_p']) + IP3)
        J_IP3R = b(theta['k_1']) * ((R * Ca * IP3) / denom)**3
        
        J_SERCA = b(theta['V_SERCA']) * Ca**2 / (Ca**2 + b(theta['K_SERCA'])**2)
        
        # ODEs
        dCa_dt = ca_lap + (J_IP3R + b(theta['k_2'])) * (S_alg - Ca) - J_SERCA
        if optogenetic: dCa_dt += b(theta['alpha'] * theta['Imax'])
        
        V_PLC = b(theta['VPLC']) * (Ca**2) / (Ca**2 + b(theta['K_PLC'])**2)
        dIP3_dt = ip3_lap + V_PLC - b(theta['K_5']) * IP3
        
        dR_dt = ((b(theta['k_tau'])**4 + Ca**4) / (b(theta['tau_max']) * b(theta['k_tau'])**4)) * \
                (1 - R * (b(theta['k_i']) + Ca) / b(theta['k_i']))

        return np.vstack([dCa_dt, dIP3_dt, dR_dt])

    def simulate(self, theta_dict=None, y0=None, int_method = 'Radau', atol = 1e-7, rtol=1e-7,  opto_bool=False, get_initiator_cells=False):
        np.random.seed(self.sim_number) # Set the seed for reproducibility (keep initiator cells consistent each run)
           
        if y0 is None:
            # Initial values: Ca, IP3, S, R
            Ca0 = np.zeros(self.n_cells)
            IP30 = np.zeros(self.n_cells)
            R0 = np.random.uniform(0.5, 0.7, self.n_cells)
            y0 = np.concatenate([Ca0, IP30, R0])

        if theta_dict is None:
            # If no theta_dict is provided, use the instance's pre-processed params
            theta_dict = self.params

        # Initialize VPLC values based on the user's request
        self.params['VPLC'] = np.random.uniform(self.params['lower'], self.params['upper'], self.n_cells)
        # Choose which cells are initiator cells based on 'frac'
        stimulated_cell_idxs = np.random.choice(self.n_cells, int(self.params['frac'][0] * self.n_cells))
        self.stimulated_cell_idxs = stimulated_cell_idxs
        # Set the VPLC of initiator cells to be random uniformly distributed between 1.3 and 1.5
        self.params['VPLC'][stimulated_cell_idxs] = np.random.uniform(1.3, 1.5, len(stimulated_cell_idxs))

        if get_initiator_cells:
            return stimulated_cell_idxs
        
        # NOTE: If a new theta_dict is passed here, it should be pre-processed
        # by the user to be a dictionary of arrays, if per-cell variation is desired.

        sol = solve_ivp(
            fun=lambda t, y: self.pouch_odes(t, y, self.params, optogenetic=opto_bool), 
            t_span=(0, self.sim_time),
            y0=y0,
            method=int_method,
            jac_sparsity=self._jac_sparsity,
            vectorized=True,
            rtol=rtol, # Often enough for calcium waves
            atol=atol
            )
        self.sol = sol
        return sol

    # =============== Plotting =========================
    # Ported from Pouch_original. solve_ivp returns adaptive time points, so the
    # calcium trajectories are interpolated onto the original fixed 0.2 s grid
    # (T = sim_time/0.2 samples) so frame spacing, kymograph scaling, and tick
    # positions match the original figures.
    _plot_dt = 0.2

    def _check_plot_inputs(self, sol, need_vertices=True):
        if sol is None:
            sol = self.sol
        if sol is None:
            raise ValueError("No solution available. Run simulate() first or pass sol=.")
        if need_vertices and self.new_vertices is None:
            raise ValueError("Cell vertices are required for plotting. Pass vertices= to the constructor "
                             "or set model.new_vertices = disc_vertices[size].")
        return sol

    def _uniform_calcium(self, sol):
        """Interpolate cytosolic calcium onto the fixed grid used by Pouch_original.
        Returns (ca, T) where ca has shape (n_cells, T)."""
        T = int(self.sim_time / self._plot_dt)
        t_grid = np.arange(T) * self._plot_dt
        t_grid = t_grid[t_grid <= sol.t[-1]]  # Guard against a solve that terminated early
        ca = interp1d(sol.t, sol.y[:self.n_cells, :], axis=1, assume_sorted=True)(t_grid)
        return ca, len(t_grid)

    def make_animation(self, sol=None, path=None): # Creation of calcium video
        sol = self._check_plot_inputs(sol)
        ca, T = self._uniform_calcium(sol)
        colormap = plt.cm.Greens
        normalize = matplotlib.colors.Normalize(vmin=np.min(ca), vmax=max(np.max(ca),1))
        with sns.axes_style("white"):
                fig=plt.figure(figsize=(25,15))
                fig.patch.set_alpha(0.)
                ax = fig.add_subplot(1,1,1)
                ax.axis('off')
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
                sm._A = []
                cbar=fig.colorbar(sm, ax=ax)
                cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=15,fontweight="bold")
                for cell in self.new_vertices:
                    ax.plot(cell[:,0],cell[:,1], linewidth=0.0, color='w', alpha = 0.0)
                patches = [matplotlib.patches.Polygon(verts) for verts in self.new_vertices ]
                def time_stamp_gen(n):
                    j=0
                    while j < n: # 0.2 sec interval to 1 hour time lapse
                        yield "Elapsed time: "+'{0:02.0f}:{1:02.0f}'.format(*divmod(j*self._plot_dt , 60))
                        j+= 50
                time_stamps=time_stamp_gen(T)
                def init():
                    return [ax.add_patch(p) for p in patches]

                def animate(frame,time_stamps):
                    for j in range(len(patches)):
                        c=colors.to_hex(colormap(normalize(frame[j])), keep_alpha=False)
                        patches[j].set_facecolor(c)
                    ax.set_title( next(time_stamps) ,fontsize=50, fontweight="bold")
                    return patches

                anim = animation.FuncAnimation(fig, animate,
                                               init_func=init,
                                               frames=ca[:,::50].T, # Calcium: Array of [n cells x time-samples]
                                               fargs=(time_stamps,),
                                               interval=70,
                                               blit=True)
        if self.save:
            if path!=None:
                if not os.path.exists(path):
                    os.makedirs(path)
                fname = path+"/"+self.size+'Disc_'+str(self.sim_number)+'_'+self.saveName
                if animation.writers.is_available('ffmpeg'):
                    anim.save(fname+'.mp4', writer='ffmpeg')
                else:
                    # Pillow (bundled with matplotlib) cannot write .mp4, so fall back to .gif
                    print("ffmpeg not found; saving animation as .gif instead "
                          "(install with: conda install -c conda-forge ffmpeg)")
                    anim.save(fname+'.gif', writer='pillow')
            else:
                print("Provide a path for saving videos")
        return anim

    def draw_profile(self, path=None): # Draw the VPLC Profile for the simulation
        if self.new_vertices is None:
            raise ValueError("Cell vertices are required for plotting. Pass vertices= to the constructor "
                             "or set model.new_vertices = disc_vertices[size].")
        if 'VPLC' not in self.params:
            raise ValueError("VPLC values not set. Run simulate() first.")
        VPLC_state = self.params['VPLC']
        colormap = plt.cm.Greens
        normalize = matplotlib.colors.Normalize(vmin=.0, vmax=1.5)
        with sns.axes_style("white"):
                fig=plt.figure(figsize=(45,25))
                ax = fig.add_subplot(1,1,1)
                ax.axis('off')
                fig.patch.set_alpha(0.)
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
                sm._A = []
                cbar=fig.colorbar(sm, ax=ax)
                cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=80,fontweight="bold" )
                for cell in self.new_vertices:
                    ax.plot(cell[:,0],cell[:,1], linewidth=1.0, color='black')
                for k in range(len(self.new_vertices)):
                        cell=self.new_vertices[k]
                        c=colors.to_hex(colormap(normalize(VPLC_state[k])), keep_alpha=False)
                        ax.fill(cell[:,0],cell[:,1], c)
        if self.save:
            if path!=None:
                if not os.path.exists(path):
                    os.makedirs(path)
                fig.savefig(path+"/"+self.size+'Disc_VPLCProfile_'+str(self.sim_number)+'_'+self.saveName+".svg",transparent=True, bbox_inches="tight")
                fig.savefig(path+"/"+self.size+'Disc_VPLCProfile_'+str(self.sim_number)+'_'+self.saveName+".png",transparent=True, bbox_inches="tight")
            else:
                print("Provide a path for saving images")

    def draw_profile_ca2(self, sol=None, path=None): # Draw the max calcium concentration profile for the simulation
        sol = self._check_plot_inputs(sol)
        ca, T = self._uniform_calcium(sol)
        # Making the color map inverted green
        colormap = plt.cm.Greens_r
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
        with sns.axes_style("white"):
                fig = plt.figure(figsize=(45, 25))
                ax = fig.add_subplot(1, 1, 1)
                ax.axis('off')
                fig.patch.set_alpha(0.)
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
                sm._A = []
                cbar = fig.colorbar(sm, ax=ax)
                cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=80, fontweight="bold")
                # Set colorbar label
                cbar.set_label('Max Calcium Concentration', fontsize=80, fontweight="bold")
                for cell in self.new_vertices:
                    ax.plot(cell[:, 0], cell[:, 1], linewidth=1.0, color='black')
                max_calcium = np.max(ca, axis=1)  # Calculate max calcium concentration for each cell
                for k in range(len(self.new_vertices)):
                        cell = self.new_vertices[k]
                        c = colors.to_hex(colormap(normalize(max_calcium[k])), keep_alpha=False)
                        ax.fill(cell[:, 0], cell[:, 1], c)
        if self.save:
            if path is not None:
                if not os.path.exists(path):
                    os.makedirs(path)
                fig.savefig(path + "/" + self.size + 'Disc_MaxCalciumProfile_' + str(self.sim_number) + '_' + self.saveName + ".svg", transparent=True, bbox_inches="tight")
                fig.savefig(path + "/" + self.size + 'Disc_MaxCalciumProfile_' + str(self.sim_number) + '_' + self.saveName + ".png", transparent=True, bbox_inches="tight")
            else:
                print("Provide a path for saving images")

    def draw_kymograph(self, sol=None, path=None): # Draw the calcium Kymograph for the simulation
        sol = self._check_plot_inputs(sol)
        ca, T = self._uniform_calcium(sol)
        with sns.axes_style("white"):
            centeriods= np.zeros((self.n_cells,2))
            for j in range(self.n_cells):
                x_center, y_center=self.new_vertices[j].mean(axis=0)
                centeriods[j,0],centeriods[j,1]=x_center, y_center
            y_axis=centeriods[:,1]
            kymograp_index=np.where((y_axis<(-490)) & (y_axis>(-510))) # Location of where to draw the kymograph line

            colormap = plt.cm.Greens
            normalize = matplotlib.colors.Normalize(vmin=np.min(ca), vmax=max(1,np.max(ca)))
            fig=plt.figure(figsize=(30,10))
            kymograph=ca[kymograp_index][:,::2]
            kymograph=np.repeat(kymograph,60,axis=0)

            plt.imshow(kymograph.T,cmap=colormap,norm=normalize)
            ax = plt.gca()
            # Ticks every ~10 min (1498 rows of 0.4 s); identical to original for a 1 hour simulation
            yticks = np.arange(0,T/2,1498)
            plt.yticks(yticks , [10*i for i in range(len(yticks))],fontsize=30, fontweight="bold")
            plt.xticks([])
            plt.ylabel('Time (min)',fontsize=30,fontweight='bold')
            if self.size=='xsmall':
                plt.xlabel('Position',fontsize=20,fontweight='bold')
            else:
                plt.xlabel('Position',fontsize=30,fontweight='bold')

            if self.save:
                if path!=None:
                    if not os.path.exists(path):
                        os.makedirs(path)
                    fig.savefig(path+"/"+self.size+'Disc_Kymograph_'+str(self.sim_number)+'_'+self.saveName+".png",transparent=True, bbox_inches="tight")
                else:
                    print("Provide a path for saving images")

        del kymograph

## =============== End of Model =========================
class PouchModelPyomoExperiment:
    """
    Pyomo-DAE experiment model of the multicell pouch dynamics.
    Equations and parameter logic are aligned with PouchModelScipy_sparse_v2.
    """

    def __init__(
        self,
        data,
        laplacian_matrix,
        dt=0.2,
        sim_time=3600,
        param_dict=None,
        random_seed=12345,
        nfe=50,
        ncp=3,
        collocation_scheme="LAGRANGE-RADAU",
        sim_initialize=False,
        opto_bool=False,
    ):
        self.data = data
        self.dt = dt
        self.sim_time = sim_time
        self.sim_number = random_seed
        self.nfe = nfe
        self.ncp = ncp
        self.collocation_scheme = collocation_scheme
        self.opto_bool = opto_bool
        self.model = None
        self.stimulated_cell_idxs = None
        self.sim_initialize = sim_initialize

        if laplacian_matrix is None:
            raise ValueError("Provide a Laplacian (dense or sparse).")
        self.laplacian = csr_matrix(laplacian_matrix)
        self.n_cells = self.laplacian.shape[0]

        self.default_params = {
            'D_p': 0.005,
            'D_c_ratio': 0.1,
            "K_PLC": 0.2,
            "K_5": 0.66,
            "k_1": 1.11,
            "k_a": 0.08,
            "k_p": 0.13,
            "k_2": 0.0203,
            "V_SERCA": 0.9,
            "K_SERCA": 0.1,
            "c_tot": 2,
            "beta": 0.185,
            "k_i": 0.4,
            "tau_max": 800,
            "k_tau": 1.5,
            "lower": 0.2,
            "upper": 0.4,
            "frac": 0.01538462,
            'alpha': 0.025,
            'Imax': 1.0,
        }

        if param_dict:
            self.params = self.default_params.copy()
            for key, value in param_dict.items():
                if isinstance(value, (list, np.ndarray)):
                    if len(value) != self.n_cells:
                        raise ValueError(
                            f"Parameter '{key}' has incorrect size. "
                            f"Expected {self.n_cells}, got {len(value)}."
                        )
                    self.params[key] = np.array(value, dtype=float)
                else:
                    self.params[key] = float(value)
        else:
            self.params = self.default_params.copy()

        for key, value in self.params.items():
            if not isinstance(value, np.ndarray):
                self.params[key] = np.full(self.n_cells, value, dtype=float)

        self._lap_rows = []
        indptr = self.laplacian.indptr
        indices = self.laplacian.indices
        data_vals = self.laplacian.data
        for i in range(self.n_cells):
            start = indptr[i]
            end = indptr[i + 1]
            row_j = indices[start:end]
            row_v = data_vals[start:end]
            self._lap_rows.append(list(zip(row_j.tolist(), row_v.tolist())))

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_model()
        return self.model

    def _build_vplc_profile(self):
        np.random.seed(self.sim_number)
        lower = self.params["lower"]
        upper = self.params["upper"]
        frac = self.params["frac"]

        vplc = np.random.uniform(lower, upper, self.n_cells)
        n_stim = int(frac[0] * self.n_cells)
        if n_stim > 0:
            stimulated = np.random.choice(self.n_cells, n_stim, replace=False)
            vplc[stimulated] = np.random.uniform(1.3, 1.5, len(stimulated))
            self.stimulated_cell_idxs = stimulated
        else:
            self.stimulated_cell_idxs = np.array([], dtype=int)
        return vplc

    def create_model(self):
        m = pyo.ConcreteModel()
        m.t = dae.ContinuousSet(bounds=(0, self.sim_time))
        m.cells = pyo.RangeSet(0, self.n_cells - 1)
        m.eps_guard = pyo.Param(initialize=1e-12, mutable=True)

        m.D_p = pyo.Var(initialize=float(self.params["D_p"][0]), bounds=(1e-8, 1))
        m.D_c_ratio = pyo.Var(initialize=float(self.params["D_c_ratio"][0]), bounds=(1e-8, 10))
        m.K_PLC = pyo.Var(initialize=float(self.params["K_PLC"][0]), bounds=(1e-8, 10))
        m.K_5 = pyo.Var(initialize=float(self.params["K_5"][0]), bounds=(1e-8, 10))
        m.k_1 = pyo.Var(initialize=float(self.params["k_1"][0]), bounds=(1e-8, 10))
        m.k_a = pyo.Var(initialize=float(self.params["k_a"][0]), bounds=(1e-8, 10))
        m.k_p = pyo.Var(initialize=float(self.params["k_p"][0]), bounds=(1e-8, 10))
        m.k_2 = pyo.Var(initialize=float(self.params["k_2"][0]), bounds=(1e-8, 10))
        m.V_SERCA = pyo.Var(initialize=float(self.params["V_SERCA"][0]), bounds=(1e-8, 10))
        m.K_SERCA = pyo.Var(initialize=float(self.params["K_SERCA"][0]), bounds=(1e-8, 10))
        m.c_tot = pyo.Var(initialize=float(self.params["c_tot"][0]), bounds=(1e-8, 10))
        m.beta = pyo.Var(initialize=float(self.params["beta"][0]), bounds=(1e-8, 10))
        m.k_i = pyo.Var(initialize=float(self.params["k_i"][0]), bounds=(1e-8, 10))
        m.tau_max = pyo.Var(initialize=float(self.params["tau_max"][0]), bounds=(1e-8, 1e5))
        m.k_tau = pyo.Var(initialize=float(self.params["k_tau"][0]), bounds=(1e-8, 10))
        m.lower = pyo.Var(initialize=float(self.params["lower"][0]), bounds=(0.0, 10))
        m.upper = pyo.Var(initialize=float(self.params["upper"][0]), bounds=(0.0, 10))
        m.frac = pyo.Var(initialize=float(self.params["frac"][0]), bounds=(0.0, 1.0))
        m.alpha = pyo.Var(initialize=float(self.params["alpha"][0]), bounds=(0.0, 10))
        m.Imax = pyo.Var(initialize=float(self.params["Imax"][0]), bounds=(0.0, 10))

        vplc_profile = self._build_vplc_profile()
        vplc_init = {i: float(vplc_profile[i]) for i in range(self.n_cells)}
        m.VPLC = pyo.Var(m.cells, initialize=vplc_init, bounds=(0.0, 5.0))

        m.Ca = pyo.Var(m.cells, m.t, initialize=0.0, within=pyo.NonNegativeReals)
        m.IP3 = pyo.Var(m.cells, m.t, initialize=0.0, within=pyo.NonNegativeReals)
        m.R = pyo.Var(m.cells, m.t, initialize=0.6, within=pyo.NonNegativeReals)

        m.dCadt = dae.DerivativeVar(m.Ca, wrt=m.t)
        m.dIP3dt = dae.DerivativeVar(m.IP3, wrt=m.t)
        m.dRdt = dae.DerivativeVar(m.R, wrt=m.t)

        def ode_ca_rule(m, i, t):
            ca_lap = sum(v * m.Ca[j, t] for j, v in self._lap_rows[i])
            ip3 = m.IP3[i, t]
            ca = m.Ca[i, t]
            r = m.R[i, t]
            denom = (m.k_a + ca) * (m.k_p + ip3) + m.eps_guard
            j_ip3r = m.k_1 * ((r * ca * ip3) / denom) ** 3
            j_serca = m.V_SERCA * ca**2 / (ca**2 + m.K_SERCA**2 + m.eps_guard)
            s_alg = (m.c_tot - ca) / (m.beta + m.eps_guard)
            rhs = (
                m.D_p * m.D_c_ratio * ca_lap
                + (j_ip3r + m.k_2) * (s_alg - ca)
                - j_serca
            )
            if self.opto_bool:
                rhs += m.alpha * m.Imax
            return m.dCadt[i, t] == rhs

        m.ode_Ca = pyo.Constraint(m.cells, m.t, rule=ode_ca_rule)

        def ode_ip3_rule(m, i, t):
            ip3_lap = sum(v * m.IP3[j, t] for j, v in self._lap_rows[i])
            ca = m.Ca[i, t]
            j_plc = m.VPLC[i] * (ca**2 / (ca**2 + m.K_PLC**2 + m.eps_guard))
            return m.dIP3dt[i, t] == m.D_p * ip3_lap + j_plc - m.K_5 * m.IP3[i, t]

        m.ode_IP3 = pyo.Constraint(m.cells, m.t, rule=ode_ip3_rule)

        def ode_r_rule(m, i, t):
            ca = m.Ca[i, t]
            num = m.k_tau**4 + ca**4
            den = m.tau_max * m.k_tau**4 + m.eps_guard
            return m.dRdt[i, t] == (num / den) * (1 - m.R[i, t] * (m.k_i + ca) / (m.k_i + m.eps_guard))

        m.ode_R = pyo.Constraint(m.cells, m.t, rule=ode_r_rule)

        self.model = m
        return m

    def finalize_model(self):
        m = self.model

        fixed_scalar_params = [
            m.D_p, m.D_c_ratio, m.K_PLC, m.K_5, m.k_1, m.k_a, m.k_p, m.k_2,
            m.V_SERCA, m.K_SERCA, m.c_tot, m.beta, m.k_i, m.tau_max, m.k_tau,
            m.lower, m.upper, m.frac, m.alpha, m.Imax,
        ]
        for var in fixed_scalar_params:
            var.fix(pyo.value(var))

        for i in m.cells:
            m.VPLC[i].fix(pyo.value(m.VPLC[i]))

        np.random.seed(self.sim_number)
        r0 = np.random.uniform(0.5, 0.7, self.n_cells)
        for i in m.cells:
            m.Ca[i, 0].fix(0.0)
            m.IP3[i, 0].fix(0.0)
            m.R[i, 0].fix(float(r0[i]))

        discretizer = pyo.TransformationFactory("dae.collocation")
        discretizer.apply_to(
            m,
            nfe=self.nfe,
            ncp=self.ncp,
            wrt=m.t,
            scheme=self.collocation_scheme,
        )
        if self.sim_initialize:
            try:
                sim = dae.Simulator(m, package="scipy")
                tsim, profiles = sim.simulate(numpoints=self.nfe + 1, integrator="lsoda")
                sim.initialize_model()
            except Exception:
                pass

        self.model = m
        return m

    def _nearest_time(self, t_val, tol=1e-8):
        t_float = float(t_val)
        for t in self.model.t:
            if abs(float(t) - t_float) <= tol:
                return t
        t_list = list(self.model.t)
        if not t_list:
            raise ValueError("Model time set is empty.")
        nearest = min(t_list, key=lambda t: abs(float(t) - t_float))
        return nearest

    def label_model(self):
        m = self.model

        required_keys = ("cell", "t", "ca_obs")
        if not all(k in self.data for k in required_keys):
            raise KeyError(
                "data must include keys 'cell', 't', and 'ca_obs' "
                "for calcium output labeling."
            )
        n_obs = len(self.data["cell"])
        if len(self.data["t"]) != n_obs or len(self.data["ca_obs"]) != n_obs:
            raise ValueError("data['cell'], data['t'], and data['ca_obs'] must have the same length.")

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        for cell_i, t_i, y_i in zip(self.data["cell"], self.data["t"], self.data["ca_obs"]):
            i = int(cell_i)
            if i < 0 or i >= self.n_cells:
                raise IndexError(f"Cell index {i} is out of bounds for n_cells={self.n_cells}.")
            t_match = self._nearest_time(t_i)
            m.experiment_outputs[m.Ca[i, t_match]] = float(y_i)

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        for i in m.cells:
            m.unknown_parameters[m.VPLC[i]] = pyo.value(m.VPLC[i])

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        for cell_i, t_i in zip(self.data["cell"], self.data["t"]):
            i = int(cell_i)
            t_match = self._nearest_time(t_i)
            m.measurement_error[m.Ca[i, t_match]] = None

        return m

## =============== End of Model =========================

# Original implementation of the Pouch class for reference.
class Pouch_original(object): 
    def __init__(self, params=None, size = 'small', sim_number=0, save=False, saveName='default',
                 folder_path=None, sim_folder_path=None):
        
        """Class implementing pouch structure and simulating Calcium signaling.
        Inputs:
        
        params (dict)
            A Python dictionary of parameters to simulate with the keys:
            ['K_PLC', 'K_5', 'k_1' , 'k_a', 'k_p', 'k_2', 'V_SERCA', 'K_SERCA', 'c_tot', 'beta', 'k_i', 'D_p', 'tau_max', 'k_tau', 'lower', 'upper','frac', 'D_c_ratio']
        
        size (string)
            Size of the pouch to simulate:
            [xsmall, small, medium, or large]
        
        sim_number (integer)
            Represents ID of a simulation to save the figures with unique names and set the random number generator seed
        
        save (boolean)
            If True, the simulation outputs will be saved
        
        saveName (string)
            Additional distinct name to save the output files as
        
        """
        # Create characteristics of the pouch object
        self.size=size
        self.saveName=saveName
        self.sim_number=sim_number
        self.save=save
        self.param_dict=params
        
        # If parameters are not set, then use baseline values
        if self.param_dict==None:
            self.param_dict={'K_PLC': 0.2, 'K_5': 0.66, 'k_1': 1.11, 'k_a': 0.08, 'k_p': 0.13, 'k_2': 0.0203, 'V_SERCA': 0.9, 'K_SERCA': 0.1,
            'c_tot': 2, 'beta': .185, 'k_i': 0.4, 'D_p': 0.005, 'tau_max': 800, 'k_tau': 1.5, 'lower': 0.2, 'upper': 0.4, 'frac': 0.01538462, 'D_c_ratio': 0.1,
            'alpha': 0.025, 'Imax': 1, 'sim_time': 3600, } 
        # If a dictionary is given, assure all parameters are provided
        required_params = ['D_c_ratio', 'D_p', 'K_5', 'K_PLC', 'K_SERCA', 'V_SERCA', 'beta', 'c_tot', 'frac', 
                   'k_1', 'k_2', 'k_a', 'k_i', 'k_p', 'k_tau', 'lower', 'tau_max', 'upper', 'alpha', 'Imax',
                   'sim_time']
        missing_params = [param for param in required_params if param not in self.param_dict]
        if missing_params:
            print(f"Improper parameter input, missing parameters: {missing_params}")
            return
            
        # Load statics for wing disc geometries    
        disc_vertices = np.load(os.path.join(folder_path, "geometry", "disc_vertices.npy"), allow_pickle=True).item()  # Vertices
        disc_laplacians = np.load(os.path.join(folder_path, "geometry", "disc_sizes_laplacian.npy"), allow_pickle=True).item()  # Laplacian Matrix
        disc_adjs = np.load(os.path.join(folder_path, "geometry", "disc_sizes_adj.npy"), allow_pickle=True).item()  # Adjacency matrix
        
        self.adj_matrix=disc_adjs[self.size] # Adjacency Matrix
        self.laplacian_matrix=disc_laplacians[self.size] # Laplacian Matrix
        self.new_vertices=disc_vertices[self.size] # Vertices
        
        # Establish baseline parameter values for the simulation
        self.K_PLC=self.param_dict['K_PLC']  # .2
        self.K_5=self.param_dict['K_5'] # 0.66
        self.k_1=self.param_dict['k_1'] # 1.11
        self.k_a=self.param_dict['k_a'] # 0.08
        self.k_p=self.param_dict['k_p'] # 0.13
        self.k_2=self.param_dict['k_2'] # 0.0203
        self.V_SERCA=self.param_dict['V_SERCA'] # .9
        self.K_SERCA=self.param_dict['K_SERCA'] # .1
        self.c_tot=self.param_dict['c_tot'] # 2
        self.beta=self.param_dict['beta'] # .185
        self.k_i=self.param_dict['k_i'] # 0.4
        self.D_p =self.param_dict['D_p'] # 0.005
        self.D_c =self.param_dict['D_c_ratio']*self.D_p
        self.tau_max=self.param_dict['tau_max'] # 800
        self.k_tau=self.param_dict['k_tau'] # 1.5
        self.lower=self.param_dict['lower'] # Lower bound of standby cell VPLCs
        self.upper=self.param_dict['upper'] # Upper bound of standy cell VPLCs
        self.frac=self.param_dict['frac']   # Fraction of initiator cells
        self.alpha = self.param_dict['alpha'] # 0.005
        self.Imax = self.param_dict['Imax'] # 1
        self.sim_time = self.param_dict['sim_time'] # 3600 seconds (1 hour)


        # Establish characteristics of the pouch for simulations
        self.n_cells=self.adj_matrix.shape[0] # Number of cells in the pouch
        self.dt=.2 # Time step for ODE approximations
        self.T=int(self.sim_time/self.dt) # Simulation to run for 3600 seconds (1 hour) by default



        self.disc_dynamics=np.zeros((self.n_cells,4,self.T)) # Initialize disc_dynamics to save simulation calcium, IP3, calcium_ER, ratio
        self.VPLC_state=np.zeros((self.n_cells,1)) # Initialize VPLC array for cells
    
    def simulate(self): # Simulate dynamics of system
            np.random.seed(self.sim_number) # Set the seed for reproducibility (keep initiator cells consistent each run)
            
            self.disc_dynamics[:,2,0] = (self.c_tot-self.disc_dynamics[:,0,0])/self.beta # Initialize simulation ER Calcium
            self.disc_dynamics[:,3,0]=np.random.uniform(.5,.7,size=(self.n_cells,1)).T # Initialize simulation fraction of inactivated IP3R receptors
            self.VPLC_state=np.random.uniform(self.lower,self.upper,(self.n_cells,1)) # Initialize the values for VPLCs of standby cells to be random uniformly distributed from lower to upper
            stimulated_cell_idxs=np.random.choice(self.n_cells, int(self.frac*self.n_cells)) # Choose which cells are initiator cells
            self.stimulated_cell_idxs=stimulated_cell_idxs
            self.VPLC_state[stimulated_cell_idxs,0]=np.random.uniform(1.3,1.5,len(stimulated_cell_idxs)) # Set the VPLC of initiator cells to be random uniformly distributed between 1.3 and 1.5
            
            V_PLC=self.VPLC_state.reshape((self.n_cells,1)) # Establish the VPLCs to be passed into the ODE approximations
            
            # ODE approximation solving
            for step in range(1,self.T):
                # ARRAY REFORMATTING
                ca=self.disc_dynamics[:,0,step-1].reshape(-1,1)
                ipt=self.disc_dynamics[:,1,step-1].reshape(-1,1)
                s=self.disc_dynamics[:,2,step-1].reshape(-1,1)
                r=self.disc_dynamics[:,3,step-1].reshape(-1,1)
                ca_laplacian=self.D_c*np.dot(self.laplacian_matrix,ca)
                ipt_laplacian=self.D_p*np.dot(self.laplacian_matrix,ipt)
                
                # ODE EQUATIONS
                self.disc_dynamics[:,0,step]=(ca+self.dt*(ca_laplacian+(self.k_1*(np.divide(np.divide(r*np.multiply(ca,ipt),(self.k_a+ca)),(self.k_p+ipt)))**3 +self.k_2)*(s-ca)-self.V_SERCA*(ca**2)/(ca**2+self.K_SERCA**2))).T
                self.disc_dynamics[:,1,step]=(ipt+self.dt*(ipt_laplacian+np.multiply(V_PLC,np.divide(ca**2,(ca**2+self.K_PLC**2)))-self.K_5*ipt)).T
                self.disc_dynamics[:,2,step]=((self.c_tot-ca)/self.beta).T
                self.disc_dynamics[:,3,step]=(r+self.dt*((self.k_tau**4+ca**4)/(self.tau_max*self.k_tau**4))*((1-r*(self.k_i+ca)/self.k_i))).T

            
    def simulate_optogenetic_dt(self): # Simulate dynamics of system
        np.random.seed(self.sim_number) # Set the seed for reproducibility (keep initiator cells consistent each run)

        # Optogenetic stimulation parameters
         # Time setup
        
        time_vector = np.arange(0, self.sim_time, self.dt)
        self.time_vector = time_vector
        self.T = len(time_vector)  # Ensure simulation steps match

        # Optogenetic stimulus setup, always on by default
        cycle_duration = 120     # Full cycle (on + off)
        on_duration = 120        # Stimulation ON time
        off_duration = cycle_duration - on_duration

        opto_stimulation = np.zeros_like(time_vector)
        for i, t in enumerate(time_vector):
            if (t % cycle_duration) < on_duration:
                opto_stimulation[i] = self.alpha*self.Imax*1.0  # Stimulus is ON

        # Expand opto signal to all cells (shape: n_cells x T)
        opto_stimulation_matrix = np.tile(opto_stimulation, (self.n_cells, 1))


        
        self.disc_dynamics[:,2,0] = (self.c_tot-self.disc_dynamics[:,0,0])/self.beta # Initialize simulation ER Calcium
        self.disc_dynamics[:,3,0]=np.random.uniform(.5,.7,size=(self.n_cells,1)).T # Initialize simulation fraction of inactivated IP3R receptors
        self.VPLC_state=np.random.uniform(self.lower,self.upper,(self.n_cells,1)) # Initialize the values for VPLCs of standby cells to be random uniformly distributed from lower to upper
        stimulated_cell_idxs=np.random.choice(self.n_cells, int(self.frac*self.n_cells)) # Choose which cells are initiator cells
        self.VPLC_state[stimulated_cell_idxs,0]=np.random.uniform(1.3,1.5,len(stimulated_cell_idxs)) # Set the VPLC of initiator cells to be random uniformly distributed between 1.3 and 1.5
        
        V_PLC=self.VPLC_state.reshape((self.n_cells,1)) # Establish the VPLCs to be passed into the ODE approximations
        # ODE approximation solving
        for step in range(1,self.T):
            # ARRAY REFORMATTING
            ca=self.disc_dynamics[:,0,step-1].reshape(-1,1)
            ipt=self.disc_dynamics[:,1,step-1].reshape(-1,1)
            s=self.disc_dynamics[:,2,step-1].reshape(-1,1)
            r=self.disc_dynamics[:,3,step-1].reshape(-1,1)
            ca_laplacian=self.D_c*np.dot(self.laplacian_matrix,ca)
            ipt_laplacian=self.D_p*np.dot(self.laplacian_matrix,ipt)

            # Add optogenetic stimulus to calcium term
            opto_term = opto_stimulation_matrix[:, step].reshape(-1, 1)

            opto_term_dt = opto_term * self.dt
            # opto_term_dt = np.minimum(opto_term_dt, 5)  # Apply maximum limit for calcium
            # ODE EQUATIONS
            self.disc_dynamics[:,0,step]=(ca+self.dt*(ca_laplacian+(self.k_1*(np.divide(np.divide(r*np.multiply(ca,ipt),(self.k_a+ca)),(self.k_p+ipt)))**3 +self.k_2)*(s-ca)-self.V_SERCA*(ca**2)/(ca**2+self.K_SERCA**2)) + opto_term_dt).T
            self.disc_dynamics[:,1,step]=(ipt+self.dt*(ipt_laplacian+np.multiply(V_PLC,np.divide(ca**2,(ca**2+self.K_PLC**2)))-self.K_5*ipt)).T
            self.disc_dynamics[:,2,step]=((self.c_tot-ca)/self.beta).T
            self.disc_dynamics[:,3,step]=(r+self.dt*((self.k_tau**4+ca**4)/(self.tau_max*self.k_tau**4))*((1-r*(self.k_i+ca)/self.k_i))).T

    # def make_animation(self, path=sim_folder_path): # Creation of calcium video
    #     colormap = plt.cm.Greens
    #     normalize = matplotlib.colors.Normalize(vmin=np.min(self.disc_dynamics[:,0,:]), vmax=max(np.max(self.disc_dynamics[:,0,:]),1))
    #     with sns.axes_style("white"):
    #             fig=plt.figure(figsize=(25,15))
    #             fig.patch.set_alpha(0.)
    #             ax = fig.add_subplot(1,1,1)
    #             ax.axis('off')
    #             sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
    #             sm._A = []
    #             cbar=fig.colorbar(sm, ax=ax)
    #             cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=15,fontweight="bold")
    #             for cell in self.new_vertices:
    #                 ax.plot(cell[:,0],cell[:,1], linewidth=0.0, color='w', alpha = 0.0)
    #             patches = [matplotlib.patches.Polygon(verts) for verts in self.new_vertices ]
    #             def time_stamp_gen(n):
    #                 j=0
    #                 while j < n: # 0.2 sec interval to 1 hour time lapse
    #                     yield "Elapsed time: "+'{0:02.0f}:{1:02.0f}'.format(*divmod(j*self.dt , 60))
    #                     j+= 50
    #             time_stamps=time_stamp_gen(self.T)
    #             def init():
    #                 return [ax.add_patch(p) for p in patches]

    #             def animate(frame,time_stamps):
    #                 for j in range(len(patches)):
    #                     c=colors.to_hex(colormap(normalize(frame[j])), keep_alpha=False)
    #                     patches[j].set_facecolor(c)
    #                 ax.set_title( next(time_stamps) ,fontsize=50, fontweight="bold")
    #                 return patches

    #             anim = animation.FuncAnimation(fig, animate, 
    #                                            init_func=init, 
    #                                            frames=self.disc_dynamics[:,0,::50].T, # Disc dynamics: Array of: [n cells x 4 x time-samples]
    #                                            fargs=(time_stamps,),
    #                                            interval=70,
    #                                            blit=True)
    #     if self.save:
    #         if path!=None:
    #             if not os.path.exists(path):
    #                 os.makedirs(path)
    #             anim.save(path+"/"+self.size+'Disc_'+str(self.sim_number)+'_'+self.saveName+'.mp4')
    #         else:
    #             print("Provide a path for saving videos")
        
    # def draw_profile(self, path=sim_folder_path): # Draw the VPLC Profile for the simulation
    #     colormap = plt.cm.Greens
    #     normalize = matplotlib.colors.Normalize(vmin=.0, vmax=1.5)
    #     with sns.axes_style("white"):
    #             fig=plt.figure(figsize=(45,25))
    #             ax = fig.add_subplot(1,1,1)
    #             ax.axis('off')
    #             fig.patch.set_alpha(0.)
    #             sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
    #             sm._A = []
    #             cbar=fig.colorbar(sm, ax=ax)
    #             cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=80,fontweight="bold" )
    #             for cell in self.new_vertices:
    #                 ax.plot(cell[:,0],cell[:,1], linewidth=1.0, color='black')
    #             for k in range(len(self.new_vertices)):
    #                     cell=self.new_vertices[k]
    #                     c=colors.to_hex(colormap(normalize(self.VPLC_state[k]))[0], keep_alpha=False)
    #                     ax.fill(cell[:,0],cell[:,1], c)
    #     if self.save:
    #         if path!=None:
    #             if not os.path.exists(path):
    #                 os.makedirs(path)
    #             fig.savefig(path+"/"+self.size+'Disc_VPLCProfile_'+str(self.sim_number)+'_'+self.saveName+".svg",transparent=True, bbox_inches="tight")
    #             fig.savefig(path+"/"+self.size+'Disc_VPLCProfile_'+str(self.sim_number)+'_'+self.saveName+".png",transparent=True, bbox_inches="tight")
    #         else:
    #             print("Provide a path for saving images")

    # def draw_profile_ca2(self, path=sim_folder_path): # Draw the max calcium concentration profile for the simulation
    #     # Making the color map inverted green
    #     colormap = plt.cm.Greens_r
    #     normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
    #     with sns.axes_style("white"):
    #             fig = plt.figure(figsize=(45, 25))
    #             ax = fig.add_subplot(1, 1, 1)
    #             ax.axis('off')
    #             fig.patch.set_alpha(0.)
    #             sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
    #             sm._A = []
    #             cbar = fig.colorbar(sm, ax=ax)
    #             cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=80, fontweight="bold")
    #             # Set colorbar label
    #             cbar.set_label('Max Calcium Concentration', fontsize=80, fontweight="bold")
    #             for cell in self.new_vertices:
    #                 ax.plot(cell[:, 0], cell[:, 1], linewidth=1.0, color='black')
    #             max_calcium = np.max(self.disc_dynamics[:, 0, :], axis=1)  # Calculate max calcium concentration for each cell
    #             for k in range(len(self.new_vertices)):
    #                     cell = self.new_vertices[k]
    #                     c = colors.to_hex(colormap(normalize(max_calcium[k])), keep_alpha=False)
    #                     ax.fill(cell[:, 0], cell[:, 1], c)
    #     if self.save:
    #         if path is not None:
    #             if not os.path.exists(path):
    #                 os.makedirs(path)
    #             fig.savefig(path + "/" + self.size + 'Disc_MaxCalciumProfile_' + str(self.sim_number) + '_' + self.saveName + ".svg", transparent=True, bbox_inches="tight")
    #             fig.savefig(path + "/" + self.size + 'Disc_MaxCalciumProfile_' + str(self.sim_number) + '_' + self.saveName + ".png", transparent=True, bbox_inches="tight")
    #         else:
    #             print("Provide a path for saving images")

    # def draw_kymograph(self, path=sim_folder_path): # Draw the calcium Kymograph for the simulation
    #     with sns.axes_style("white"):
    #         centeriods= np.zeros((self.adj_matrix.shape[0],2))
    #         for j in range(self.adj_matrix.shape[0]):
    #             x_center, y_center=self.new_vertices[j].mean(axis=0)
    #             centeriods[j,0],centeriods[j,1]=x_center, y_center
    #         y_axis=centeriods[:,1]
    #         kymograp_index=np.where((y_axis<(-490)) & (y_axis>(-510))) # Location of where to draw the kymograph line

    #         colormap = plt.cm.Greens
    #         normalize = matplotlib.colors.Normalize(vmin=np.min(self.disc_dynamics[:,0,:]), vmax=max(1,np.max(self.disc_dynamics[:,0,:])))
    #         fig=plt.figure(figsize=(30,10))
    #         kymograph=self.disc_dynamics[kymograp_index,0,::][0][:,::2]
    #         kymograph=np.repeat(kymograph,60,axis=0)

    #         plt.imshow(kymograph.T,cmap=colormap,norm=normalize)
    #         ax = plt.gca()
    #         plt.yticks(np.arange(0,self.T/2,1498) , [0,10,20,30,40,50,60],fontsize=30, fontweight="bold")
    #         plt.xticks([])
    #         plt.ylabel('Time (min)',fontsize=30,fontweight='bold')
    #         if self.size=='xsmall':
    #             plt.xlabel('Position',fontsize=20,fontweight='bold')
    #         else:
    #             plt.xlabel('Position',fontsize=30,fontweight='bold')
            
    #         if self.save:
    #             if path!=None:
    #                 if not os.path.exists(path):
    #                     os.makedirs(path)
    #                 fig.savefig(path+"/"+self.size+'Disc_Kymograph_'+str(self.sim_number)+'_'+self.saveName+".png",transparent=True, bbox_inches="tight")
    #             else:
    #                 print("Provide a path for saving images")

    #     del kymograph

    # def count_spiking_cells(self, threshold=0.5, path=sim_folder_path):
    #     """
    #     Count the number of cells that spike above a calcium threshold in this simulation.

    #     Parameters:
    #         threshold: float, the calcium concentration threshold for spiking.

    #     Returns:
    #         int: The number of cells that spike above the threshold.
    #     """
    #     return int(np.sum(np.max(self.disc_dynamics[:, 0, :], axis=1) > threshold))
    # def count_oscillating_cells(self, threshold=0.5):
    #     """
    #     Count the number of cells that exhibit oscillatory behavior above a calcium threshold.

    #     Parameters:
    #         threshold: float, the calcium concentration threshold for oscillation detection.

    #     Returns:
    #         int: The number of cells that exhibit oscillations.
    #     """
    #     from scipy.signal import find_peaks
    #     osc_count = 0
    #     for cell_idx in range(self.n_cells):
    #         ca_signal = self.disc_dynamics[cell_idx, 0, :]
    #         peaks, _ = find_peaks(ca_signal, height=threshold)
    #         peak_times = self.time_vector[peaks]
    #         if len(peaks) >= 2:
    #             osc_count += 1 
    #             periods = np.diff(peak_times)
    #             avg_period = np.mean(periods)
    #             frequency = 1 / avg_period if avg_period > 0 else 0
    #             # frequencies.append(frequency)
    #         # else:
    #             # frequencies.append(0)
    #     return osc_count
    

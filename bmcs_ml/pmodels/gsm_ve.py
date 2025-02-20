import sympy as sp
import numpy as np
from scipy.optimize import minimize
from cymbol import Cymbol, cymbols
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path
import traits.api as tr
sp.init_printing()

class GSM_VE(tr.HasTraits):
    # Define symbols
    training_data_dir = tr.Directory(Path.home() / 'bmcs_training_data' / 've')
    problem_name = tr.Str('default_problem')
    
    @property
    def data_dir(self):
        data_dir = Path.home() / 'bmcs_training_data' / self.problem_name
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    eps, eps_v = cymbols(r'\varepsilon \varepsilon_\mathrm{v}', 
                         codenames='epsilon epsilon_v', real=True)
    E = Cymbol(r'E', codename='E', positive=True, real=True)
    eta = Cymbol(r'\eta', codename='eta', positive=True, real=True)
    dot_eps = Cymbol(r'\dot{\varepsilon}', codename='dot_eps', real=True)
    dot_eps_v = Cymbol(r'\dot{\varepsilon}_\mathrm{v}', codename='dot_eps_v', real=True)

    # Define the total strain as the sum of elastic and viscous parts
    eps_e_ = eps - eps_v

    # Define Helmholtz free energy Psi
    psi_ = sp.Rational(1, 2) * E * eps_e_**2

    # Define dissipation potential Phi
    phi_ = sp.Rational(1, 2) * eta * dot_eps_v**2

    # Define the stress-strain relationship for the elastic part
    sig_ = psi_.diff(eps)

    gamma_mech_ = sp.simplify(-psi_.diff(eps_v) * dot_eps_v)

    d_t = Cymbol(r'\Delta t', codename='d_t')
    eps_n = Cymbol(r'\varepsilon^{(n)}', codename='eps_n')
    eps_v_n = Cymbol(r'\varepsilon_\mathrm{v}^{(n)}', codename='eps_v_n')
    dot_eps_n = Cymbol(r'\dot{\varepsilon}^{(n)}', codename='dot_eps_n')
    dot_eps_v_n = Cymbol(r'\dot{\varepsilon}_\mathrm{v}^{(n)}', codename='dot_eps_v_n')
    delta_eps_n = Cymbol(r'\Delta{\varepsilon}^{(n)}', codename='delta_eps_n')
    delta_eps_v_n = Cymbol(r'\Delta{\varepsilon}_\mathrm{v}^{(n)}', codename='delta_eps_v_n')

    def __init__(self):
        self.psi_n1 = self.psi_.subs({self.eps_v: self.eps_v_n + self.delta_eps_v_n, 
                                      self.eps: self.eps_n + self.delta_eps_n})
        self.phi_n1 = self.phi_.subs({self.dot_eps_v: self.delta_eps_v_n / self.d_t})
        self.Pi_n1 = (-self.gamma_mech_ + self.phi_ * self.d_t).subs({self.eps_v: self.eps_v_n + self.delta_eps_v_n, 
                                                 self.eps: self.eps_n + self.delta_eps_n, 
                                                 self.dot_eps: self.delta_eps_n / self.d_t,
                                                 self.dot_eps_v: self.delta_eps_v_n / self.d_t})
        self.jac_Pi_n1 = sp.diff(self.Pi_n1, self.delta_eps_v_n) * self.d_t
        self.hes_Pi_n1 = sp.diff(self.jac_Pi_n1, self.delta_eps_v_n)
        self.sig_n1 = self.sig_.subs({self.eps: self.eps_n + self.delta_eps_n, 
                                      self.eps_v: self.eps_v_n + self.delta_eps_v_n})
        self.get_Pi = self.lambdify_functions(self.Pi_n1, (self.eps_n, self.delta_eps_n, self.eps_v_n, 
                                                        self.delta_eps_v_n, self.d_t, self.E, self.eta))
        self.get_jac_Pi = self.lambdify_functions(self.jac_Pi_n1, (self.eps_n, self.delta_eps_n, self.eps_v_n, 
                                                                self.delta_eps_v_n, self.d_t, self.E, self.eta))
        self.get_hes_Pi = self.lambdify_functions(self.hes_Pi_n1, (self.eps_n, self.delta_eps_n, self.eps_v_n, 
                                                                self.delta_eps_v_n, self.d_t, self.E, self.eta))
        self.get_sig_n1 = self.lambdify_functions(self.sig_n1, (self.eps_n, self.delta_eps_n, 
                                                                self.eps_v_n, self.delta_eps_v_n, 
                                                                self.d_t, self.E, self.eta))
        self.get_psi_n1 = self.lambdify_functions(self.psi_n1, (self.eps_n, 
                                                                self.eps_v_n,
                                                                self.delta_eps_n,
                                                                self.delta_eps_v_n, 
                                                                self.E, self.eta))
        self.get_phi_n1 = self.lambdify_functions(self.phi_n1, (self.delta_eps_v_n, 
                                                                self.d_t, 
                                                                self.E, self.eta))

    def lambdify_functions(self, expr, variables):
        return sp.lambdify(variables, expr, 'numpy', cse=True)

    # Newton-Raphson iteration function
    def newton_iteration(self, eps_t, dot_eps_t, eps_v_t, d_eps_v_next, d_t, *args, max_iter=10):
        for j in range(max_iter):
            R_ = self.get_jac_Pi(eps_t, dot_eps_t, eps_v_t, d_eps_v_next, d_t, *args)
            norm_R_ = np.sqrt(R_**2)
            if norm_R_ < 1e-6:
                break
            dR_dEps_ = self.get_hes_Pi(eps_t, dot_eps_t, eps_v_t, d_eps_v_next, d_t, *args)
            d_eps_v_next -= R_ / dR_dEps_
        if j == max_iter - 1:
            raise ValueError(f'Newton-Raphson did not converge in max_iter={max_iter}')
        return d_eps_v_next

    # Residual based time integrator for visco-elasticity
    def ti_nr(self, eps_t, time_t, *args):
        d_eps_t = np.diff(eps_t, axis=0)
        d_t_t = np.diff(time_t, axis=0)
        dd_eps_t = d_eps_t

        n_steps = len(eps_t)
        eps_v_t = np.zeros(n_steps)
        sig_t = np.zeros(n_steps)
        dd_eps_v_next = 0

        # Initialize arrays to store training data
        Pi_data = []

        for i, d_t in enumerate(d_t_t):
            dd_eps_v_next = self.newton_iteration(eps_t[i], dd_eps_t[i], eps_v_t[i], dd_eps_v_next, d_t, *args)
            sig_t[i+1] = self.get_sig_n1(eps_t[i], dd_eps_t[i], eps_v_t[i], dd_eps_v_next, d_t, *args)
            eps_v_t[i+1] = eps_v_t[i] + dd_eps_v_next

            # Collect data for training
            Pi_n1 = self.get_Pi(eps_t[i], dd_eps_t[i], eps_v_t[i], dd_eps_v_next, d_t, *args)
            Pi_data.append([eps_t[i], dd_eps_t[i], eps_v_t[i], d_t, Pi_n1])

        # Convert list to numpy array
        Pi_data = np.array(Pi_data)

        # Generate unique filename
        Pi_filename = self.data_dir / f'Pi_data_{self.problem_name}.npy'

        # Save the data
        np.save(Pi_filename, Pi_data)
        return eps_t, eps_v_t, sig_t

    # Create a simple container class to hold the data arrays
    class DataContainer:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    # Function to perform integration and store results
    def perform_integration(self, eps_t, time_t, *args):
        eps_t, eps_v_t, sig_t = self.ti_nr(eps_t, time_t, *args)
        eps_e_t = eps_t - eps_v_t
        return GSM_VE.DataContainer(
            time=time_t,
            eps_t_cycles=eps_t,
            eps_e_t=eps_e_t,
            eps_v_t=eps_v_t,
            sig_t=sig_t
        )

    @staticmethod    
    def plot_results(ax1, ax2, ax3, data, label_suffix, color='blue'):
        # Plot results
        ax1.plot(data.time, data.eps_t_cycles, color=color, label=f'Total Strain {label_suffix}', lw=1.2, linestyle='-')
        ax1.plot(data.time, data.eps_v_t, color=color, label=f'Viscous Strain {label_suffix}', lw=0.6, linestyle='--')
        ax1.fill_between(data.time, data.eps_t_cycles, data.eps_v_t, color=color, alpha=0.1, label=f'Elastic Strain {label_suffix}')
        #ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Strain')
        ax1.legend()
        ax1.set_title('Strain Components')

        # Remove the top and right spines
        ax1.spines['top'].set_color('none')
        ax1.spines['right'].set_color('none')

        # Move the bottom spine to the zero of the y-axis
        ax1.spines['bottom'].set_position('zero')

        # Add an arrow at the end of the x-axis
        ax1.annotate('', xy=(1, 0), xytext=(1, 0),
                    arrowprops=dict(arrowstyle="->", color='black', lw=0.5),
                    xycoords=('axes fraction', 'data'), textcoords='data')

        # Add a label near the arrow
        ax1.text(1, 0, 'Time [s]', va='center', ha='left', color='black', fontsize=ax1.xaxis.get_label().get_size(),
                transform=ax1.get_yaxis_transform())
        ax2.plot(data.time, data.sig_t, color=color, label=f'Stress {label_suffix}')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Stress [Pa]')
        ax2.legend()
        ax2.set_title('Stress Response')

        ax3.plot(data.eps_t_cycles, data.sig_t, color=color, label=f'Stress-strain {label_suffix}')
        ax3.set_xlabel('Strain [-]')
        ax3.set_ylabel('Stress [Pa]')
        ax3.legend()
        ax3.set_title('Stress-Strain Response')
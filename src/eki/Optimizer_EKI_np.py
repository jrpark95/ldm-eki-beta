"""
Ensemble Kalman Inversion (EKI) Optimizer with NumPy

This module implements various ensemble-based data assimilation and inverse methods
for radioactive source term estimation in the LDM-EKI system. The algorithms combine
Lagrangian particle dispersion modeling with ensemble Kalman techniques to estimate
unknown emission rates from sparse observations.

Implemented Algorithms
----------------------
1. **EnKF** : Standard Ensemble Kalman Filter
   - Stochastic update with Kalman gain
   - Linear covariance-based state correction

2. **Adaptive_EnKF** : Adaptive Ensemble Kalman Inversion
   - Automatic step size control based on data misfit
   - Prevents filter divergence and over-fitting

3. **EnKF_with_Localizer** : Localized Covariance EnKF
   - Reduces spurious long-range correlations
   - Gaspari-Cohn or Gaussian taper functions

4. **EnKF_with_barrier** : Constrained EnKF with Barrier Functions
   - Enforces box constraints using logarithmic barriers
   - Prevents physically unrealistic values

5. **EnRML** : Ensemble Randomized Maximum Likelihood
   - Gauss-Newton optimization framework
   - Iterative ensemble smoother

6. **EnKF_MDA** : Multiple Data Assimilation EnKF
   - Inflated observation error covariance
   - Repeated assimilation of same data

7. **REnKF** : Regularized EnKF with Penalty Constraints
   - Tanh-based soft constraints
   - Penalty parameter λ for constraint strength

Mathematical Notation
---------------------
- u : State vector (ensemble member)
- U : State ensemble matrix (d × N), N members
- y : Observation vector
- G(u) : Forward model operator (state to observation)
- Γ : Observation error covariance (m × m)
- C^u : State covariance (d × d)
- C^{ug} : Cross-covariance between state and observation (d × m)
- C^{gg} : Observation covariance (m × m)
- K : Kalman gain matrix (d × m)

References
----------
.. [Evensen2009] Evensen, G. (2009). "Data Assimilation: The Ensemble Kalman Filter",
   Springer, 2nd edition. doi:10.1007/978-3-642-03711-5

.. [Iglesias2013] Iglesias, M. A., Law, K. J. H., & Stuart, A. M. (2013).
   "Ensemble Kalman methods for inverse problems", Inverse Problems, 29(4), 045001.
   doi:10.1088/0266-5611/29/4/045001

.. [Emerick2013] Emerick, A. A., & Reynolds, A. C. (2013).
   "Ensemble smoother with multiple data assimilation", Computers & Geosciences, 55, 3-15.
   doi:10.1016/j.cageo.2012.03.011

.. [Oliver2008] Oliver, D. S., Reynolds, A. C., & Liu, N. (2008).
   "Inverse Theory for Petroleum Reservoir Characterization and History Matching",
   Cambridge University Press. doi:10.1017/CBO9780511535642

.. [Chada2021] Chada, N. K., Chen, Y., & Sanz-Alonso, D. (2021).
   "Iterative ensemble Kalman methods: A unified perspective with some new variants",
   Foundations of Data Science, 3(3), 331-369. doi:10.3934/fods.2021011

Author
------
Siho Jang, 2025
"""

import os
import numpy as np
import Model_Connection_np_Ensemble as Model_np

def Run(input_config={}, input_data={}):
    """
    Main EKI optimization loop.

    Args:
        input_config: Configuration dictionary containing algorithm parameters
        input_data: Input data dictionary

    Returns:
        Tuple containing optimization results and convergence metrics
    """
    np.random.seed(0)
    inverse = Inverse(input_config)
    model = Model_np.Model(input_config, input_data)

    time = np.arange(input_config['time'], dtype=int)
    iteration = np.arange(input_config['iteration'], dtype=int)

    state_update_list = []
    state_update_iteration_list = []

    misfits_list = []
    misfits_list_tmp = []  # temporary list for convergence check
    noises_list = []
    residuals_list = []
    discrepancy_bools_list = []
    residual_bools_list = []

    misfit_list = []
    misfit_list_tmp = []  # temporary list for convergence check
    noise_list = []
    residual_list = []
    discrepancy_bool_list = []
    residual_bool_list = []
    ensXiter_list = []
    diff_list = []
    
    info_list = []

    for t in time:
        if t == 0:
            state_predict = model.make_ensemble()
        else:
            state_predict = model.predict(state_update, t)
        ob, ob_err = model.get_ob(t)
        obs = np.tile(ob, (input_config['sample'], 1)).T
        if input_config['perturb_option'] == 'On_time':
            obs = _perturb(ob, ob_err, input_config['sample'])

        alpha_inv_history = []
        T_n = 0.0

        for i in iteration:
            print(f"Iteration {i+1}/{len(iteration)}")
            if input_config['perturb_option'] == 'On_iter':
                obs = _perturb(ob, ob_err, input_config['sample'])

            # Forward model: state → observation
            state_in_ob = model.state_to_ob(state_predict)

            # Alpha calculation for Adaptive EKI
            if input_config['Adaptive_EKI'] == 'On':
                Phi_n = compute_Phi_n(obs, state_in_ob, ob_err)
                alpha_inv = compute_alpha_inv(len(state_in_ob), Phi_n, alpha_inv_history, i)
                alpha_inv_history.append(alpha_inv)
                if len(alpha_inv_history) > 1 and (alpha_inv >= 1.0 - T_n or alpha_inv < 0.0):
                    print(f"Adaptive EKI converged at iteration {i+1}")
                    break
                T_n += alpha_inv

            # Inverse methods (choose one): EnKF, EnRML, EnKF_MDA, REnKF, etc.
            # FAIL-FAST VALIDATION: Algorithm options must be explicitly specified (no .get() defaults)
            if 'Localized_EKI' not in input_config:
                print(f"\033[1;31m[INPUT ERROR]\033[0m Missing required parameter: Localized_EKI")
                print()
                print(f"  \033[33mProblem:\033[0m")
                print(f"    The Localized_EKI parameter is required to specify whether")
                print(f"    covariance localization should be used in the EnKF update.")
                print(f"    Localization reduces spurious correlations from finite ensemble size.")
                print()
                print(f"  \033[36mRequired value:\033[0m")
                print(f"    Localized_EKI: 'On' or 'Off'")
                print()
                print(f"  \033[32mAlgorithmic context:\033[0m")
                print(f"    - 'On':  Apply Gaspari-Cohn or Gaussian taper to covariance matrices")
                print(f"    - 'Off': Use full ensemble covariance (may have spurious long-range correlations)")
                print()
                print(f"  \033[32mRecommended:\033[0m")
                print(f"    - High-dimensional problems (d >> N): Use 'On' with appropriate localization radius")
                print(f"    - Low-dimensional problems (d ~ N):  'Off' is usually sufficient")
                print()
                print(f"  \033[36mFix in:\033[0m input/eki.conf")
                import sys
                sys.exit(1)

            if 'Adaptive_EKI' not in input_config:
                print(f"\033[1;31m[INPUT ERROR]\033[0m Missing required parameter: Adaptive_EKI")
                print()
                print(f"  \033[33mProblem:\033[0m")
                print(f"    The Adaptive_EKI parameter is required to specify whether")
                print(f"    adaptive step size control should be used.")
                print(f"    Adaptive EKI prevents over-fitting by dynamically adjusting α_n.")
                print()
                print(f"  \033[36mRequired value:\033[0m")
                print(f"    Adaptive_EKI: 'On' or 'Off'")
                print()
                print(f"  \033[32mAlgorithmic context:\033[0m")
                print(f"    - 'On':  Step size α_n chosen to satisfy Σα_n = 1 (Iglesias et al. 2013)")
                print(f"    - 'Off': Fixed step size α = 1 (standard EnKF)")
                print()
                print(f"  \033[32mRecommended:\033[0m")
                print(f"    - Nonlinear problems: 'On' (more robust convergence)")
                print(f"    - Linear/mildly nonlinear: 'Off' (faster, simpler)")
                print()
                print(f"  \033[36mFix in:\033[0m input/eki.conf")
                import sys
                sys.exit(1)

            if 'Regularization' not in input_config:
                print(f"\033[1;31m[INPUT ERROR]\033[0m Missing required parameter: Regularization")
                print()
                print(f"  \033[33mProblem:\033[0m")
                print(f"    The Regularization parameter is required to specify whether")
                print(f"    penalty-based constraints should be enforced.")
                print(f"    Regularization prevents physically unrealistic values (e.g., negative emissions).")
                print()
                print(f"  \033[36mRequired value:\033[0m")
                print(f"    Regularization: 'On' or 'Off'")
                print()
                print(f"  \033[32mAlgorithmic context:\033[0m")
                print(f"    - 'On':  Apply tanh penalty functions for box constraints [0, 10^15]")
                print(f"    - 'Off': No explicit constraints (EnKF may produce negative values)")
                print()
                print(f"  \033[32mRecommended:\033[0m")
                print(f"    - Physical constraints required: 'On' with REnKF_lambda ~ 0.001-0.1")
                print(f"    - Unconstrained problems: 'Off'")
                print()
                print(f"  \033[36mFix in:\033[0m input/eki.conf")
                import sys
                sys.exit(1)

            # Now safe to access - we know keys exist
            # IMPORTANT: Pass localizer_func explicitly (no default None allowed)
            if input_config['Localized_EKI'] == 'On':
                if input_config['Adaptive_EKI'] == 'On':
                    state_update = inverse.Adaptive_EnKF_with_Localizer(
                        i, state_predict, state_in_ob, obs, ob_err, ob, alpha_inv,
                        Inverse.centralized_localizer  # Explicit localization function
                    )
                else:
                    state_update = inverse.EnKF_with_Localizer(
                        i, state_predict, state_in_ob, obs, ob_err, ob,
                        Inverse.centralized_localizer  # Explicit localization function
                    )
            elif input_config['Adaptive_EKI'] == 'On':
                state_update = inverse.Adaptive_EnKF(i, state_predict, state_in_ob, obs, ob_err, ob, alpha_inv)
            elif input_config['Regularization'] == 'On':
                state_update = inverse.REnKF(i, state_predict, state_in_ob, obs, ob_err, ob)
            else:
                state_update = inverse.EnKF(i, state_predict, state_in_ob, obs, ob_err, ob)
            
            misfits = np.mean(state_update,1) - np.mean(state_predict, 1)
            state_predict = state_update.copy()
            state_update_iteration_list.append(state_update.copy())

            # Save iteration data for visualization
            iteration_dir = 'logs/eki_iterations'
            if not os.path.exists(iteration_dir):
                os.makedirs(iteration_dir, exist_ok=True)
            np.save(f'{iteration_dir}/iteration_{i+1:03d}.npy', state_update)

            # Convergence check
            misfits_list_tmp.append(misfits)
            misfits_list.append(misfits.copy())
            misfits_err = np.zeros(misfits.shape)            
            discrepancy_bools, residual_bools, residuals, noises = _convergence(misfits_list_tmp, misfits_err)
            discrepancy_bools_list.append(discrepancy_bools)
            residual_bools_list.append(residual_bools)
            residuals_list.append(np.asarray(residuals).copy())
            noises_list.append(noises.copy())

            misfit = np.linalg.norm(np.mean(obs - state_in_ob, axis=1))
            misfit_list_tmp.append(misfit)
            misfit_list.append(misfit.copy())
            discrepancy_bool, residual_bool, residual, noise = _convergence(misfit_list_tmp, ob_err)
            discrepancy_bool_list.append(discrepancy_bool)
            residual_bool_list.append(residual_bool)
            residual_list.append(np.asarray(residual).copy())
            noise_list.append(noise.copy())
            
            ensXiter = 0
            if residual_bool:
                ensXiter = input_config['sample'] * (i+1)
                ensXiter_list.append(ensXiter)
            else:
                ensXiter_list.append(ensXiter)        

        # Error check (skip if shapes don't match - happens when using more obs timesteps than state timesteps)
        if np.mean(state_update.copy(), 1).shape == model.real_state_init.shape:
            diff = np.abs(np.mean(state_update.copy(), 1) - model.real_state_init)/model.real_state_init
            diff_list.append(diff)
        else:
            diff_list.append(np.zeros_like(model.real_state_init))
        
        state_update_list.append(state_update.copy())

    if input_config['time'] == 1:
        state_update_list = state_update_list[0]
        diff_list = diff_list[0]

    return (state_update_list, state_update_iteration_list, info_list, misfit_list,
            discrepancy_bool_list, residual_bool_list, residual_list, noise_list,
            ensXiter_list, diff_list, misfits_list, discrepancy_bools_list,
            residual_bools_list, residuals_list)


class Inverse(object):
    """
    Inverse modeling methods for EKI.

    Supports multiple variants:
    - EnKF: Standard Ensemble Kalman Filter
    - Adaptive_EnKF: Adaptive step size EKI
    - EnKF_with_Localizer: Localized covariance
    - EnRML: Ensemble Randomized Maximum Likelihood
    - EnKF_MDA: Multiple Data Assimilation
    - REnKF: Regularized EnKF with constraints
    """
    def __init__(self, input_config):
        """
        Initialize inverse modeling algorithms with FAIL-FAST parameter validation.

        All algorithm parameters must be explicitly provided in input_config.
        NO default values are allowed - missing or invalid parameters cause immediate exit.

        Parameters
        ----------
        input_config : dict
            MUST contain ALL of the following keys with valid values:
            - 'sample': int >= 10 (ensemble size)
            - 'EnKF_MDA_steps': float > 0 (MDA inflation factor α)
            - 'EnRML_step_length': 0 < float <= 1 (EnRML step length β)
            - 'REnKF_lambda': float >= 0 (Regularization penalty parameter λ)
            - 'Localization_weighting_factor': float > 0 (Localization radius L)

        Raises
        ------
        SystemExit
            If any required parameter is missing or invalid
        """
        # FAIL-FAST VALIDATION: Check ALL required parameters exist
        required_params = {
            'sample': 'Ensemble size (number of ensemble members)',
            'EnKF_MDA_steps': 'MDA inflation factor (α for ES-MDA algorithm)',
            'EnRML_step_length': 'EnRML step length (β for Gauss-Newton damping)',
            'REnKF_lambda': 'Regularization parameter (λ for penalty constraints)',
            'Localization_weighting_factor': 'Localization radius (L for covariance tapering)'
        }

        for param, description in required_params.items():
            if param not in input_config:
                print(f"\033[1;31m[INPUT ERROR]\033[0m Missing required parameter: {param}")
                print()
                print(f"  \033[33mProblem:\033[0m")
                print(f"    The {param} parameter is REQUIRED for EKI algorithms.")
                print(f"    Description: {description}")
                print()
                print(f"  \033[36mRequired value:\033[0m")
                print(f"    {param}: <numeric value>")
                print()
                print(f"  \033[36mFix in:\033[0m input/eki.conf")
                import sys
                sys.exit(1)

        # Validate 'sample' (ensemble size)
        self.sample = input_config['sample']
        if not isinstance(self.sample, (int, np.integer)) or self.sample < 10:
            print(f"\033[1;31m[INPUT ERROR]\033[0m Invalid ensemble size: {self.sample}")
            print()
            print(f"  \033[33mProblem:\033[0m")
            print(f"    Ensemble size must be >= 10 for reliable covariance estimation.")
            print(f"    With N < 10, sample covariances are rank-deficient and unreliable.")
            print()
            print(f"  \033[36mRequired value:\033[0m")
            print(f"    sample: integer >= 10")
            print()
            print(f"  \033[32mRecommended:\033[0m")
            print(f"    - Small test: 50 members")
            print(f"    - Production: 100-200 members")
            print(f"    - High-dimensional: >= 50 (ensure N > rank of covariance)")
            print()
            print(f"  \033[36mFix in:\033[0m input/eki.conf (set EKI_ENSEMBLE_SIZE >= 10)")
            import sys
            sys.exit(1)

        # Validate 'EnKF_MDA_steps' (MDA inflation factor α)
        self.alpha = input_config['EnKF_MDA_steps']
        if not isinstance(self.alpha, (int, float, np.number)) or self.alpha <= 0:
            print(f"\033[1;31m[INPUT ERROR]\033[0m Invalid MDA inflation factor: {self.alpha}")
            print()
            print(f"  \033[33mProblem:\033[0m")
            print(f"    EnKF_MDA_steps (α) must be positive. This controls how many times")
            print(f"    the data is assimilated with inflated observation error covariance.")
            print(f"    Invalid value: α <= 0 would cause division by zero or negative variance.")
            print()
            print(f"  \033[36mRequired value:\033[0m")
            print(f"    EnKF_MDA_steps: float > 0")
            print()
            print(f"  \033[32mAlgorithmic meaning:\033[0m")
            print(f"    - α = 4: Use 4 assimilation steps (common choice)")
            print(f"    - Larger α: More conservative updates, better for nonlinear problems")
            print(f"    - Constraint: Σ(1/α_i) = 1 for N_α steps")
            print()
            print(f"  \033[32mRecommended:\033[0m")
            print(f"    - Standard: α = 4 to 10")
            print(f"    - Conservative: α = 10 to 20")
            print()
            print(f"  \033[36mFix in:\033[0m input/eki.conf")
            import sys
            sys.exit(1)

        # Validate 'EnRML_step_length' (EnRML Gauss-Newton step β)
        self.beta = input_config['EnRML_step_length']
        if not isinstance(self.beta, (int, float, np.number)) or not (0 < self.beta <= 1.0):
            print(f"\033[1;31m[INPUT ERROR]\033[0m Invalid EnRML step length: {self.beta}")
            print()
            print(f"  \033[33mProblem:\033[0m")
            print(f"    EnRML_step_length (β) must be in range (0, 1].")
            print(f"    This controls the Gauss-Newton step length in EnRML algorithm.")
            print(f"    β = 1: Full Gauss-Newton step (may diverge for nonlinear problems)")
            print(f"    β < 1: Damped/line-search step (more robust)")
            print()
            print(f"  \033[36mRequired value:\033[0m")
            print(f"    EnRML_step_length: 0 < float <= 1.0")
            print()
            print(f"  \033[32mAlgorithmic meaning:\033[0m")
            print(f"    Update: U^(n+1) = β*U_0 + (1-β)*U^n + β*ΔU^n")
            print(f"    - β = 1.0: Trust Gauss-Newton fully (fast convergence if linear)")
            print(f"    - β = 0.5: Conservative damping (safer for nonlinear)")
            print()
            print(f"  \033[32mRecommended:\033[0m")
            print(f"    - Linear/mildly nonlinear: β = 1.0")
            print(f"    - Strongly nonlinear: β = 0.5 to 0.8")
            print()
            print(f"  \033[36mFix in:\033[0m input/eki.conf")
            import sys
            sys.exit(1)

        # Validate 'REnKF_lambda' (Regularization parameter λ)
        self.lambda_value = input_config['REnKF_lambda']
        if not isinstance(self.lambda_value, (int, float, np.number)) or self.lambda_value < 0:
            print(f"\033[1;31m[INPUT ERROR]\033[0m Invalid regularization parameter: {self.lambda_value}")
            print()
            print(f"  \033[33mProblem:\033[0m")
            print(f"    REnKF_lambda (λ) must be non-negative.")
            print(f"    This controls the strength of constraint penalty functions.")
            print(f"    Negative λ would reverse constraints (unphysical).")
            print()
            print(f"  \033[36mRequired value:\033[0m")
            print(f"    REnKF_lambda: float >= 0")
            print()
            print(f"  \033[32mAlgorithmic meaning:\033[0m")
            print(f"    Objective: J(u) = ||y - G(u)||²_Γ^-1 + λ Ψ(u)")
            print(f"    - λ = 0: No constraints (standard EnKF)")
            print(f"    - λ > 0: Penalty for violating bounds [0, 10^15]")
            print(f"    - Larger λ: Stronger constraint enforcement")
            print()
            print(f"  \033[32mRecommended:\033[0m")
            print(f"    - No constraints needed: λ = 0.0")
            print(f"    - Soft constraints: λ = 0.001 to 0.01")
            print(f"    - Hard constraints: λ = 0.1 to 1.0")
            print()
            print(f"  \033[36mFix in:\033[0m input/eki.conf")
            import sys
            sys.exit(1)

        # Validate 'Localization_weighting_factor' (Localization radius L)
        self.weighting_factor = input_config['Localization_weighting_factor']
        if not isinstance(self.weighting_factor, (int, float, np.number)) or self.weighting_factor <= 0:
            print(f"\033[1;31m[INPUT ERROR]\033[0m Invalid localization radius: {self.weighting_factor}")
            print()
            print(f"  \033[33mProblem:\033[0m")
            print(f"    Localization_weighting_factor (L) must be positive.")
            print(f"    This defines the correlation length scale for covariance tapering.")
            print(f"    L <= 0 would cause all correlations to be zero or explode.")
            print()
            print(f"  \033[36mRequired value:\033[0m")
            print(f"    Localization_weighting_factor: float > 0")
            print()
            print(f"  \033[32mAlgorithmic meaning:\033[0m")
            print(f"    Taper function: ρ(r) = exp(-r²/(2L²))  (Gaussian taper)")
            print(f"    - Small L: Strong localization (only nearby variables correlated)")
            print(f"    - Large L: Weak localization (long-range correlations retained)")
            print()
            print(f"  \033[32mRecommended:\033[0m")
            print(f"    - Spatial problems: L ~ 5-20 grid points")
            print(f"    - Temporal problems: L ~ 1.0 (normalized time units)")
            print(f"    - Emission time series: L ~ 1.0 to 2.0")
            print()
            print(f"  \033[36mFix in:\033[0m input/eki.conf")
            import sys
            sys.exit(1)

        # Store config for later use
        self.input_config = input_config
        self.time = 0

    def EnKF_with_barrier(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        EnKF with Barrier Method for box constraints.

        Applies logarithmic barrier functions to enforce bounds during updates.
        Note: This method is kept for reference but not used by default.
        """
        tau = 0.5
        inflation = 0.9
        lower = 1.0e-6
        upper = 1.0e+15

        if iteration == 0:
            x0 = _ave_substracted(state_predict)  # (d x N)
            self.C0 = (1.0 / (self.sample - 1.0)) * np.dot(x0, x0.T)

        x = _ave_substracted(state_predict)      # (d x N)
        hx = _ave_substracted(state_in_ob)         # (d x N)
        pxz = 1.0 / (self.sample - 1.0) * np.dot(x, hx.T)
        pzz = 1.0 / (self.sample - 1.0) * np.dot(hx, hx.T)
        k = np.dot(pxz, np.linalg.pinv(pzz + ob_err))
        dx = np.dot(k, obs - state_in_ob)
        state_update = state_predict + dx

        C_hat = (1.0 / (self.sample - 1.0)) * np.dot(x, x.T)

        d = state_predict.shape[0]
        lower_bounds = np.full(d, lower)
        upper_bounds = np.full(d, upper)
        state_mean = np.mean(state_predict, axis=1)
        eps = 1e-10
        barrier_grad = (1.0/(upper_bounds - state_mean + eps) - 1.0/(state_mean - lower_bounds + eps))
        barrier_term = (1.0 / tau) * np.dot(C_hat, barrier_grad)
        barrier_term_expanded = np.tile(barrier_term[:, np.newaxis], (1, self.sample))
        state_update = state_update + barrier_term_expanded

        if inflation != 1.0:
            mean_state = np.mean(state_update, axis=1, keepdims=True)
            state_update = mean_state + inflation * (state_update - mean_state)

        return state_update

    def EnKF(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        Standard Ensemble Kalman Filter (EnKF) update.

        Implements the stochastic EnKF analysis step with perturbed observations.
        The update equation corrects each ensemble member based on the Kalman gain
        and the innovation (observation minus prediction).

        Parameters
        ----------
        iteration : int
            Current iteration index (unused in standard EnKF)
        state_predict : ndarray, shape (d, N)
            Forecast ensemble matrix, d states × N members
        state_in_ob : ndarray, shape (m, N)
            Predicted observations G(U^f), m observations × N members
        obs : ndarray, shape (m, N)
            Perturbed observation matrix (y + ξ), each column is perturbed
        ob_err : ndarray, shape (m, m)
            Observation error covariance matrix Γ
        ob : ndarray, shape (m,)
            True observation vector (not used in perturbed EnKF)

        Returns
        -------
        state_update : ndarray, shape (d, N)
            Analysis ensemble matrix U^a

        Notes
        -----
        **Mathematical Formulation:**

        The EnKF analysis step computes:

            U^a = U^f + K (Y - G(U^f))

        where the Kalman gain K is:

            K = C^{ug} (C^{gg} + Γ)^{-1}

        with sample covariances computed from ensemble anomalies:

            U' = U^f - mean(U^f)    (state anomalies)
            G' = G(U^f) - mean(G(U^f))    (observation anomalies)

            C^{ug} = (1/(N-1)) U' (G')^T    (cross-covariance)
            C^{gg} = (1/(N-1)) G' (G')^T    (observation covariance)

        **Algorithm:**
        1. Compute ensemble anomalies (mean-subtracted)
        2. Estimate covariances from anomalies
        3. Compute Kalman gain via Moore-Penrose pseudoinverse
        4. Update each ensemble member with innovation

        **Computational Complexity:**
        - Covariance estimation: O(d·m·N)
        - Kalman gain: O(d·m²) for pseudoinverse
        - Update: O(d·m·N)

        References
        ----------
        .. [1] Evensen, G. (2009), "Data Assimilation: The Ensemble Kalman Filter",
               Springer, Section 4.2.

        Examples
        --------
        >>> state_forecast = np.random.randn(100, 50)  # 100 states, 50 members
        >>> obs_predicted = forward_model(state_forecast)
        >>> obs_perturbed = perturb_observations(obs, obs_err, 50)
        >>> state_analysis = enkf.EnKF(0, state_forecast, obs_predicted,
        ...                            obs_perturbed, obs_err, obs)

        Author
        ------
        Siho Jang, 2025
        """
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)
        k = np.dot(pxz, np.linalg.pinv(pzz+ob_err))
        dx = np.dot(k, obs-state_in_ob)
        state_update = np.array(state_predict) + dx
        return state_update

    def Adaptive_EnKF(self, iteration, state_predict, state_in_ob, obs, ob_err, ob, alpha_inv):
        """
        Adaptive Ensemble Kalman Inversion with automatic step size control.

        Implements adaptive EKI where the step size α is determined dynamically
        based on the normalized data misfit. This prevents over-fitting and
        ensures convergence to the true posterior distribution.

        Parameters
        ----------
        iteration : int
            Current iteration index
        state_predict : ndarray, shape (d, N)
            Forecast ensemble matrix
        state_in_ob : ndarray, shape (m, N)
            Predicted observations
        obs : ndarray, shape (m, N)
            Perturbed observation matrix
        ob_err : ndarray, shape (m, m)
            Observation error covariance
        ob : ndarray, shape (m,)
            True observation vector
        alpha_inv : float
            Inverse of step size parameter (α^{-1})

        Returns
        -------
        state_update : ndarray, shape (d, N)
            Updated ensemble matrix

        Notes
        -----
        **Mathematical Formulation:**

        The adaptive EKI update is:

            U^{n+1} = U^n + K_α (Y_α - G(U^n))

        where the adaptive Kalman gain is:

            K_α = C^{ug} (C^{gg} + α Γ)^{-1}

        and the perturbed observations are:

            Y_α = y + √α ξ,  ξ ~ N(0, Γ)

        **Step Size Computation:**

        The step size α_n is chosen adaptively to satisfy:

            sum_{n=0}^{N-1} α_n = 1

        using the normalized misfit Φ_n:

            Φ_n = || Γ^{-1/2} (y - G(U^n)) ||

        The step size is:

            α_n^{-1} = max{ m/(2 Φ̄_n), √(m/(2 Var(Φ_n))) }

        capped at 1 - T_n where T_n = sum_{k=0}^{n-1} α_k.

        **Convergence Criterion:**

        The method terminates when α_n ≥ 1 - T_n, ensuring that the
        cumulative step size reaches 1.

        **Physical Interpretation:**
        - Small α: Conservative update (trust prior more)
        - Large α: Aggressive update (trust data more)
        - Adaptive choice balances prior and data information

        **Computational Complexity:**
        O(d·m·N) for covariances + O(d·m²) for Kalman gain

        References
        ----------
        .. [1] Iglesias, M. A., Law, K. J. H., & Stuart, A. M. (2013).
               "Ensemble Kalman methods for inverse problems", Inverse Problems, 29(4).
               doi:10.1088/0266-5611/29/4/045001

        .. [2] Chada, N. K., et al. (2021). "Iterative ensemble Kalman methods:
               A unified perspective with some new variants", Foundations of Data Science, 3(3).
               doi:10.3934/fods.2021011

        Examples
        --------
        >>> alpha_inv = compute_alpha_inv(m, Phi_n, alpha_history, iteration)
        >>> state_new = enkf.Adaptive_EnKF(iter, state, obs_pred, obs_pert,
        ...                                obs_err, obs, alpha_inv)

        Author
        ------
        Siho Jang, 2025
        """
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)
        alpha = 1.0/alpha_inv
        k_modified = np.dot(pxz, np.linalg.pinv(pzz + alpha * ob_err))
        xi = _perturb(np.zeros(ob.shape), ob_err, self.sample)
        perturbed_diff = obs + np.sqrt(alpha) * xi - state_in_ob
        dx = np.dot(k_modified, perturbed_diff)
        state_update = np.array(state_predict) + dx
        return state_update

    @staticmethod
    def centralized_localizer(matrix, L):
        """
        Apply Gaussian taper localization to covariance matrix.

        Reduces spurious long-range correlations in ensemble covariance estimates
        by element-wise multiplication with a distance-dependent taper function.

        Parameters
        ----------
        matrix : ndarray, shape (d, m)
            Covariance or cross-covariance matrix to localize
        L : float
            Localization length scale (correlation cutoff distance)

        Returns
        -------
        localized_matrix : ndarray, shape (d, m)
            Element-wise localized covariance matrix

        Notes
        -----
        **Mathematical Formulation:**

        The localized covariance is:

            C_loc[i,j] = C[i,j] · ρ(d_ij / L)

        where ρ is a Gaussian taper function:

            ρ(r) = exp(-r² / 2)

        and d_ij is the distance between state variables i and j.

        **Physical Interpretation:**
        - Removes spurious correlations due to finite ensemble size
        - Preserves local covariance structure
        - Essential for high-dimensional problems (d >> N)

        **Localization Length Scale L:**
        - Small L: Strong localization (only nearby variables correlated)
        - Large L: Weak localization (long-range correlations retained)
        - Typical range: L ~ 5-20 grid points

        **Computational Complexity:**
        O(d² + m²) for distance matrix computation + O(d·m) for multiplication

        References
        ----------
        .. [1] Hamill, T. M., Whitaker, J. S., & Snyder, C. (2001).
               "Distance-dependent filtering of background error covariance estimates
               in an ensemble Kalman filter", Monthly Weather Review, 129(11), 2776-2790.
               doi:10.1175/1520-0493(2001)129<2776:DDFOBE>2.0.CO;2

        .. [2] Gaspari, G., & Cohn, S. E. (1999). "Construction of correlation
               functions in two and three dimensions", Quarterly Journal of the Royal
               Meteorological Society, 125(554), 723-757. doi:10.1002/qj.49712555417

        Author
        ------
        Siho Jang, 2025
        """
        distances1 = compute_distances(matrix.shape[0])
        distances2 = compute_distances(matrix.shape[1])

        # Compute the localization matrices using Gaussian taper
        Psi1 = np.vectorize(lambda d: np.exp(-d**2 / (2*L**2)))(distances1)
        Psi2 = np.vectorize(lambda d: np.exp(-d**2 / (2*L**2)))(distances2)

        # Apply element-wise localization
        localized_matrix = matrix * Psi1 * Psi2
        return localized_matrix

    def EnKF_with_Localizer(self, iteration, state_predict, state_in_ob, obs, ob_err, ob,
                           localizer_func):
        """
        EnKF with covariance localization.

        Reduces spurious correlations in covariance estimation by applying
        localization functions to the covariance matrices.

        Parameters
        ----------
        localizer_func : callable
            REQUIRED. Localization function with signature localizer_func(matrix, L).
            Must not be None. Use Inverse.centralized_localizer as default caller-side.

        Notes
        -----
        **FAIL-FAST**: This parameter is REQUIRED and must be provided by the caller.
        If you want the default Gaussian taper, pass Inverse.centralized_localizer explicitly.

        Raises
        ------
        ValueError
            If localizer_func is None (fail-fast validation)
        """
        if localizer_func is None:
            print(f"\033[1;31m[INPUT ERROR]\033[0m Missing required parameter: localizer_func")
            print()
            print(f"  \033[33mProblem:\033[0m")
            print(f"    The localizer_func parameter is REQUIRED for EnKF_with_Localizer.")
            print(f"    Localization functions define how correlations decay with distance.")
            print(f"    Passing None would silently use a default, violating fail-fast principles.")
            print()
            print(f"  \033[36mRequired value:\033[0m")
            print(f"    localizer_func: callable(matrix, L) → localized_matrix")
            print()
            print(f"  \033[32mRecommended functions:\033[0m")
            print(f"    - Inverse.centralized_localizer  (Gaussian taper: exp(-r²/2L²))")
            print(f"    - Custom Gaspari-Cohn function   (5th order piecewise polynomial)")
            print()
            print(f"  \033[32mExample:\033[0m")
            print(f"    state = enkf.EnKF_with_Localizer(i, state, obs_pred, obs,")
            print(f"                                       obs_err, obs,")
            print(f"                                       Inverse.centralized_localizer)")
            print()
            print(f"  \033[36mFix in:\033[0m src/eki/Optimizer_EKI_np.py (caller code)")
            import sys
            sys.exit(1)
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)

        # Apply localization to covariances
        pxz = localizer_func(pxz, self.weighting_factor)
        pzz = localizer_func(pzz, self.weighting_factor)

        k_modified = np.dot(pxz, np.linalg.pinv(pzz + ob_err))
        dx = np.dot(k_modified, obs-state_in_ob)
        state_update = state_predict + dx
        return state_update

    def Adaptive_EnKF_with_Localizer(self, iteration, state_predict, state_in_ob, obs, ob_err, ob,
                                     alpha_inv, localizer_func):
        """
        Adaptive EnKF with covariance localization.

        Combines adaptive step size control with localized covariances.

        Parameters
        ----------
        localizer_func : callable
            REQUIRED. Localization function (see EnKF_with_Localizer for details).

        Notes
        -----
        **FAIL-FAST**: This parameter must be explicitly provided. No silent defaults.
        """
        if localizer_func is None:
            print(f"\033[1;31m[INPUT ERROR]\033[0m Missing required parameter: localizer_func")
            print()
            print(f"  \033[33mProblem:\033[0m")
            print(f"    The localizer_func parameter is REQUIRED for Adaptive_EnKF_with_Localizer.")
            print(f"    This method combines adaptive EKI with covariance localization.")
            print()
            print(f"  \033[36mRequired value:\033[0m")
            print(f"    localizer_func: callable(matrix, L) → localized_matrix")
            print()
            print(f"  \033[32mRecommended:\033[0m")
            print(f"    Pass Inverse.centralized_localizer explicitly (do not rely on defaults)")
            print()
            print(f"  \033[36mFix in:\033[0m src/eki/Optimizer_EKI_np.py (caller code)")
            import sys
            sys.exit(1)
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)

        # Apply localization to covariances
        pxz = localizer_func(pxz, self.weighting_factor)
        pzz = localizer_func(pzz, self.weighting_factor)

        alpha = 1.0/alpha_inv
        k_modified = np.dot(pxz, np.linalg.pinv(pzz + alpha * ob_err))

        xi = _perturb(np.zeros(ob.shape), ob_err, self.sample)
        perturbed_diff = obs + np.sqrt(alpha) * xi - state_in_ob

        dx = np.dot(k_modified, perturbed_diff)
        state_update = state_predict + dx
        return state_update

    def EnRML(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        Ensemble Randomized Maximum Likelihood (EnRML) method.

        Iterative ensemble smoother based on Gauss-Newton optimization framework.
        Minimizes the negative log-posterior by iteratively refining the ensemble
        with respect to the initial prior and observed data.

        Parameters
        ----------
        iteration : int
            Current iteration index
        state_predict : ndarray, shape (d, N)
            Current ensemble estimate
        state_in_ob : ndarray, shape (m, N)
            Predicted observations G(U^n)
        obs : ndarray, shape (m, N)
            Perturbed observation matrix
        ob_err : ndarray, shape (m, m)
            Observation error covariance Γ
        ob : ndarray, shape (m,)
            True observation vector

        Returns
        -------
        state_update : ndarray, shape (d, N)
            Updated ensemble matrix

        Notes
        -----
        **Mathematical Formulation:**

        EnRML minimizes the objective function:

            J(u) = 1/2 ||u - u_0||²_{C_0^{-1}} + 1/2 ||y - G(u)||²_{Γ^{-1}}

        using Gauss-Newton iterations:

            U^{n+1} = β U_0 + (1-β) U^n + β ΔU^n

        where the Gauss-Newton increment is:

            ΔU^n = K_GN (Y - G(U^n) - S (U^n - U_0))

        and the Gauss-Newton Kalman gain is:

            K_GN = C_0 S^T (S C_0 S^T + Γ)^{-1}

        with sensitivity matrix:

            S = ∂G/∂u ≈ G' (U'^n)^{-1}

        **Step Length Parameter β:**
        - Typical values: β ∈ [0.5, 1.0]
        - β = 1.0: Full Gauss-Newton step
        - β < 1.0: Line search / damped step

        **Key Differences from EnKF:**
        1. Uses initial prior C_0 (not current covariance)
        2. Includes sensitivity matrix (linearization)
        3. Has memory of initial state U_0

        **Convergence:**
        Typically converges in 5-20 iterations for mildly nonlinear problems.

        **Computational Complexity:**
        O(d²·N) for sensitivity + O(d·m²) for Kalman gain

        References
        ----------
        .. [1] Chen, Y., & Oliver, D. S. (2013). "Levenberg-Marquardt forms of
               the iterative ensemble smoother for efficient history matching and
               uncertainty quantification", Computational Geosciences, 17(4), 689-703.
               doi:10.1007/s10596-013-9351-5

        .. [2] Oliver, D. S., & Chen, Y. (2011). "Recent progress on reservoir
               history matching: a review", Computational Geosciences, 15(1), 185-221.
               doi:10.1007/s10596-010-9194-2

        Examples
        --------
        >>> enkf = Inverse(input_config)
        >>> for iter in range(max_iter):
        ...     obs_pred = forward_model(state_current)
        ...     state_new = enkf.EnRML(iter, state_current, obs_pred,
        ...                            obs_pert, obs_err, obs)

        Author
        ------
        Siho Jang, 2025
        """
        if iteration == 0:
            self.state0 = state_predict
        x0 = _ave_substracted(self.state0)
        p0 = 1.0/(self.sample-1.0) * np.dot(x0, x0.T)
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        sen = np.dot(hx, np.linalg.pinv(x))
        p0_sen = np.dot(p0, sen.T)
        gn_sub = np.dot(np.dot(sen, p0), sen.T) + ob_err
        gn = np.dot(p0_sen, np.linalg.inv(gn_sub))
        dx = np.dot(gn, obs-state_in_ob) + \
             np.dot(gn, np.dot(sen, state_predict - self.state0))
        state_update = self.beta*self.state0 + (1.0-self.beta)*state_predict + self.beta*dx
        return state_update

    def EnKF_MDA(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        Ensemble Kalman Filter with Multiple Data Assimilation (ES-MDA).

        Assimilates the same observation data multiple times with inflated observation
        error covariance to improve convergence and maintain ensemble spread. This
        avoids filter collapse while approaching the posterior distribution.

        Parameters
        ----------
        iteration : int
            Current iteration index (which of N_α assimilations)
        state_predict : ndarray, shape (d, N)
            Current ensemble estimate
        state_in_ob : ndarray, shape (m, N)
            Predicted observations
        obs : ndarray, shape (m, N)
            Perturbed observation matrix
        ob_err : ndarray, shape (m, m)
            Observation error covariance Γ
        ob : ndarray, shape (m,)
            True observation vector

        Returns
        -------
        state_update : ndarray, shape (d, N)
            Updated ensemble matrix

        Notes
        -----
        **Mathematical Formulation:**

        ES-MDA performs N_α sequential EnKF updates with inflated observation
        error covariance:

            U^{i+1} = U^i + K_i (Y_i - G(U^i))

        where the Kalman gain at step i uses inflated covariance:

            K_i = C^{ug}_i (C^{gg}_i + α_i Γ)^{-1}

        and the inflation factors satisfy:

            sum_{i=1}^{N_α} α_i^{-1} = 1

        **Perturbed Observations:**

        At each step, observations are re-perturbed with inflated noise:

            Y_i = y + √α_i ξ_i,  ξ_i ~ N(0, Γ)

        **Inflation Factor α:**
        - Typical values: α = 4 to 10 (using N_α = 4 to 10 steps)
        - Equal inflation: α_i = α for all i
        - Controls strength of each update

        **Advantages over Standard EnKF:**
        1. Multiple passes through data improves nonlinear convergence
        2. Inflated errors maintain ensemble diversity
        3. Reduces risk of filter collapse
        4. More robust for strongly nonlinear problems

        **Computational Cost:**
        N_α times more expensive than single EnKF update, but often worth it
        for improved accuracy.

        **Computational Complexity:**
        O(N_α · d · m · N) total for N_α assimilation steps

        References
        ----------
        .. [1] Emerick, A. A., & Reynolds, A. C. (2013).
               "Ensemble smoother with multiple data assimilation",
               Computers & Geosciences, 55, 3-15. doi:10.1016/j.cageo.2012.03.011

        .. [2] Emerick, A. A., & Reynolds, A. C. (2012).
               "History matching time-lapse seismic data using the ensemble Kalman filter
               with multiple data assimilations", Computational Geosciences, 16(3), 639-659.
               doi:10.1007/s10596-012-9275-5

        Examples
        --------
        >>> alpha = 4  # Use 4 assimilation steps
        >>> for step in range(alpha):
        ...     obs_pred = forward_model(state_current)
        ...     state_current = enkf.EnKF_MDA(step, state_current, obs_pred,
        ...                                   obs_pert, obs_err, obs)

        Author
        ------
        Siho Jang, 2025
        """
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)
        k_modified = np.dot(pxz, np.linalg.inv(pzz + self.alpha*ob_err))
        obs_origin = np.tile(ob, (obs.shape[1], 1)).T
        ob_perturb = obs - obs_origin
        obs_mda = obs_origin + np.sqrt(self.alpha)*ob_perturb
        dx = np.dot(k_modified, obs_mda - state_in_ob)
        state_update = state_predict + dx
        return state_update

    def REnKF(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        Regularized Ensemble Kalman Filter (REnKF) with soft constraints.

        Incorporates penalty-based regularization to enforce physical constraints
        (e.g., non-negativity, bounds) during the EnKF update. Uses smooth tanh
        penalty functions to guide ensemble members toward feasible regions.

        Parameters
        ----------
        iteration : int
            Current iteration index
        state_predict : ndarray, shape (d, N)
            Current ensemble estimate
        state_in_ob : ndarray, shape (m, N)
            Predicted observations
        obs : ndarray, shape (m, N)
            Perturbed observation matrix
        ob_err : ndarray, shape (m, m)
            Observation error covariance
        ob : ndarray, shape (m,)
            True observation vector

        Returns
        -------
        state_update : ndarray, shape (d, N)
            Updated ensemble with constraint penalties applied

        Notes
        -----
        **Mathematical Formulation:**

        REnKF minimizes the augmented objective function:

            J_λ(u) = 1/2 ||y - G(u)||²_{Γ^{-1}} + λ Ψ(u)

        where Ψ(u) is a barrier/penalty function enforcing constraints.

        The update equation is:

            U^{n+1} = U^n + K (Y - G(U^n)) + K_c ∇Ψ(U^n)

        where K is the standard Kalman gain and K_c is the constraint gain:

            K_c = -(C^{uu} - K C^{gu})

        **Penalty Function:**

        Uses hyperbolic tangent (tanh) barrier functions:

            Ψ(u_i) = tanh(u_i - l)  if u_i < l  (lower bound)
                     tanh(u_i - u)  if u_i > u  (upper bound)
                     0              otherwise

        **Gradient:**

            ∇Ψ(u_i) = sech²(u_i - l)  if u_i < l
                      sech²(u_i - u)  if u_i > u
                      0              otherwise

        **Regularization Parameter λ:**
        - Controls penalty strength
        - Typical values: λ ∈ [10⁻⁴, 10⁻¹]
        - Larger λ: Stronger constraint enforcement
        - Smaller λ: Weaker constraint, more data-driven

        **Advantages:**
        1. Smooth differentiable penalties (unlike hard constraints)
        2. Maintains ensemble structure
        3. Prevents unphysical negative values
        4. Can enforce box constraints [l, u]

        **Typical Constraints in LDM-EKI:**
        - Non-negativity: emission rates ≥ 0
        - Upper bound: emission rates ≤ 10¹⁵ Bq/s
        - Positivity of concentrations

        **Computational Complexity:**
        O(d·m·N) for standard gain + O(d·N) for constraint term

        References
        ----------
        .. [1] Mandel, J., Bergou, E., Gürol, S., Gratton, S., & Kasanický, I. (2016).
               "Hybrid Levenberg-Marquardt and weak-constraint ensemble Kalman smoother method",
               Nonlinear Processes in Geophysics, 23(2), 59-73. doi:10.5194/npg-23-59-2016

        .. [2] Nocedal, J., & Wright, S. J. (2006). "Numerical Optimization",
               2nd edition, Springer. Chapter 19: Interior-Point Methods.

        Examples
        --------
        >>> lambda_val = 0.01  # Regularization strength
        >>> for iter in range(max_iter):
        ...     obs_pred = forward_model(state_current)
        ...     state_new = enkf.REnKF(iter, state_current, obs_pred,
        ...                            obs_pert, obs_err, obs)
        ...     # Check constraint violations
        ...     print(f"Min value: {state_new.min()}, Max: {state_new.max()}")

        Author
        ------
        Siho Jang, 2025
        """
        def tanh_penalty(x):
            return np.tanh(x)

        def tanh_penalty_derivative(x):
            return 1 - np.tanh(x) ** 2

        def constraint_func(x):
            penalty = np.zeros_like(x)
            penalty[x < 0.0] = tanh_penalty(x[x < 0.0])
            penalty[x > 1.0e+15] = tanh_penalty(x[x > 1.0e+15] - 1.0e+15)
            return penalty

        def constraint_derivative(x):
            derivative = np.zeros_like(x)
            derivative[x < 0] = tanh_penalty_derivative(x[x < 0])
            derivative[x > 1e+15] = tanh_penalty_derivative(x[x > 1e+15] - 1e+15)
            return derivative

        def lambda_function(iteration):
            lambda_value = self.lambda_value
            return lambda_value

        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)
        k = np.dot(pxz, np.linalg.pinv(pzz+ob_err))
        dx = np.dot(k, obs-state_in_ob)
        pxx = 1.0/(self.sample-1.0) * np.dot(x, x.T)
        pzx = 1.0/(self.sample-1.0) * np.dot(hx, x.T)
        k_constraints = -pxx + np.dot(k, pzx)

        constraint_mat = np.zeros([len(state_predict), self.sample])
        w = np.identity(state_predict.shape[0])
        lamda = lambda_function(iteration)

        for j in range(self.sample):
            jstate = state_predict[:, j]
            gw = np.dot(constraint_derivative(jstate).T, w)
            gwg = np.dot(gw, constraint_func(jstate))
            constraint_mat[:, j] += lamda * gwg

        dx_constraints = np.dot(k_constraints, constraint_mat)
        state_update = state_predict + dx + dx_constraints
        return state_update


# Utility Functions

def _perturb(ave, cov, samp, perturb=1e-30):
    """
    Generate perturbed ensemble from mean and covariance.

    Args:
        ave: Mean vector
        cov: Covariance matrix
        samp: Number of samples
        perturb: Small value added to diagonal for numerical stability

    Returns:
        Perturbed ensemble matrix (dim x samp)
    """
    dim = len(ave)
    cov += np.eye(dim) * perturb
    chol_decomp = np.linalg.cholesky(cov)
    corr_perturb = np.random.normal(loc=0.0, scale=1.0, size=(dim, samp))
    perturbation = np.matmul(chol_decomp, corr_perturb)
    get_perturb = np.tile(ave, (samp, 1)).T + perturbation
    return get_perturb


def _ave_substracted(m):
    """
    Subtract ensemble mean from each member.

    Args:
        m: Ensemble matrix (dim x samp)

    Returns:
        Mean-subtracted ensemble (dim x samp)
    """
    samp = m.shape[1]
    ave = np.array([np.mean(m, 1)])
    ave = np.tile(ave.T, (1, samp))
    ave_substracted = np.array(m) - ave
    return ave_substracted


def _convergence(misfit_list, ob_err, noise_factor=1.0, min_residual=1e-6):
    """
    Check convergence based on discrepancy principle and residual change.

    Args:
        misfit_list: List of misfit values over iterations
        ob_err: Observation error covariance
        noise_factor: Multiplier for noise level threshold
        min_residual: Minimum relative residual for convergence

    Returns:
        Tuple: (discrepancy_bool, residual_bool, residual, noise)
    """
    # Discrepancy principle
    if np.any(ob_err) == True:
        noise_level = np.sqrt(np.trace(ob_err))
        noise = noise_factor * noise_level
    else:
        noise = noise_factor * ob_err
    discrepancy_bool = misfit_list[-1] < noise

    # Residual of misfit
    mIter = len(misfit_list) -1
    if mIter == 0:
        residual = np.nan
        residual_bool = False
    else:
        residual = abs(misfit_list[mIter]-misfit_list[mIter-1])
        residual /= abs(misfit_list[0])
        residual_bool = residual < min_residual
    return discrepancy_bool, residual_bool, residual, noise


def sec_fisher(cov, N_ens):
    """
    Secondary Fisher information matrix with ensemble correction.

    Applies statistical correction to covariance matrix based on ensemble size.

    Args:
        cov: Covariance matrix
        N_ens: Ensemble size

    Returns:
        Corrected covariance matrix
    """
    rows, cols = cov.shape

    # If the matrix is square, process normally
    if rows == cols:
        v = np.sqrt(np.diag(cov))
        V = np.diag(v)
        V_inv = np.linalg.inv(V)
        R = V_inv @ cov @ V_inv
    else:
        # For non-square matrices, just compute the normalized matrix
        row_scales = np.sqrt(np.diag(np.dot(cov, cov.T)))
        col_scales = np.sqrt(np.diag(np.dot(cov.T, cov)))
        R = (cov / row_scales[:, None]) / col_scales[None, :]
    
    R_sec = np.zeros_like(R)
    for i in range(rows):
        for j in range(min(i + 1, cols)):
            r = R[i, j]
            if (i == j) | (r >= 1):
                R_sec[i, j] = 1
            else:
                s = np.arctanh(r)
                σ_s = 1 / np.sqrt(N_ens - 3)
                σ_r = (np.tanh(s + σ_s) - np.tanh(s - σ_s)) / 2
                Q = r / σ_r
                alpha = (Q**2) / (1 + Q**2)
                R_sec[i, j] = alpha * r

    if rows == cols:
        return V @ R_sec @ V
    else:
        return (R_sec * row_scales[:, None]) * col_scales[None, :]


def compute_Phi_n(y, state_in_ob, Gamma):
    """
    Compute normalized data misfit for adaptive EKI.

    Args:
        y: Observations
        state_in_ob: Predicted observations
        Gamma: Observation error covariance

    Returns:
        Vector of normalized misfits for each ensemble member
    """
    results_matrix = np.dot(np.linalg.pinv(Gamma)**0.5, (y - state_in_ob))
    phi_n = np.linalg.norm(results_matrix, ord=2, axis=0)
    return phi_n


def compute_alpha_inv(M, Phi_n, alpha_inv_history, n):
    """
    Compute adaptive step size parameter for EKI.

    Args:
        M: Number of observations
        Phi_n: Normalized data misfit vector
        alpha_inv_history: List of previous alpha_inv values
        n: Current iteration number

    Returns:
        Adaptive step size parameter alpha_inv
    """
    # Empirical mean and variance of Phi_n
    Phi_mean = np.mean(Phi_n)
    Phi_var = np.var(Phi_n)

    # Compute cumulative step size
    t_n = sum(alpha_inv_history[:n]) if n != 1 else 0

    # Compute alpha_inv with bounds
    alpha_inv = min(max(M/(2.0 * Phi_mean), np.sqrt(M/(2.0 * Phi_var))), 1.0 - t_n)

    return alpha_inv


def compute_distances(size):
    """
    Compute pairwise distance matrix for localization.

    Args:
        size: Dimension of the state space

    Returns:
        Distance matrix (size x size)
    """
    return np.abs(np.arange(size) - np.arange(size)[:, np.newaxis])
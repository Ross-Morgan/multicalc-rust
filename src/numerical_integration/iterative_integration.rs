use core::marker::PhantomData;

use crate::numeric::Numeric;
use crate::numerical_integration::integrator::*;
use crate::numerical_integration::mode::IterativeMethod;
use crate::utils::error_codes::CalcError;

pub const DEFAULT_TOTAL_ITERATIONS: u64 = 100;

/// Configuration shared by the single- and multi-variable iterative integrators.
#[derive(Debug, Clone, Copy)]
pub struct IterativeConfig {
    /// Number of intervals the composite rule walks. See [`DEFAULT_TOTAL_ITERATIONS`].
    pub total_iterations: u64,
    /// The composite rule to use: Booles, Simpsons or Trapezoidal.
    pub integration_method: IterativeMethod,
}

impl Default for IterativeConfig {
    /// Boole's rule with [`DEFAULT_TOTAL_ITERATIONS`] intervals; optimal for most generic equations.
    fn default() -> Self {
        SingleVariableSolver {
            total_iterations: DEFAULT_TOTAL_ITERATIONS,
            integration_method: IterativeMethod::Booles,
        }
    }
}

impl SingleVariableSolver {
    ///returns the total nuber of iterations
    pub fn get_total_iterations(&self) -> u64 {
        self.total_iterations
    }

    ///sets the total nuber of iterations
    pub fn set_total_iterations(&mut self, total_iterations: u64) {
        self.total_iterations = total_iterations;
    }

    ///returns the chosen integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn get_integration_method(&self) -> IterativeMethod {
        self.integration_method
    }

    ///sets the integration method
    ///choices are: Booles, Simpsons and Trapezoidal
    pub fn set_integration_method(&mut self, integration_method: IterativeMethod) {
        self.integration_method = integration_method;
    }

    ///custom constructor. Optimal for fine-tuning for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self {
        SingleVariableSolver {
            total_iterations,
            integration_method,
        }
    }

    /// Checks that the iteration count is non-zero and every limit is well-defined.
    /// The iteration count is checked before the limits so a zero count reports
    /// [`CalcError::IterationsZero`] regardless of the limits.
    fn check_for_errors<T: Numeric, const NUM_INTEGRATIONS: usize>(
        &self,
        integration_limit: &[[T; 2]; NUM_INTEGRATIONS],
    ) -> Result<(), CalcError> {
        if self.total_iterations == 0 {
            return Err(CalcError::IterationsZero);
        }

        for &limit in integration_limit {
            if limit[0] >= limit[1] {
                return Err(INTEGRATION_LIMITS_ILL_DEFINED);
            }
        }

        if NUM_INTEGRATIONS != number_of_integrations {
            return Err(INCORRECT_NUMBER_OF_INTEGRATION_LIMITS);
        }

        Ok(())
    }

    ///returns the numerical integration via Booles' method
    ///number_of_integrations: number of times the equation needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    fn get_booles<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> f64 {
        if number_of_integrations == 1 {
            let (lower_limit, upper_limit) = get_domain_change_limits(&integration_limit[0]);

            let mut ans =
                7.0 * get_domain_change_function_value(func, &integration_limit[0], lower_limit);

            let delta = (upper_limit - lower_limit) / (self.total_iterations as f64);

            let mut current_point = lower_limit;
            let mut multiplier = 32.0;

            for iter in 0..self.total_iterations - 1 {
                current_point += delta;
                ans += multiplier
                    * get_domain_change_function_value(func, &integration_limit[0], current_point);

                if (iter + 2) % 2 != 0 {
                    multiplier = 32.0;
                } else if (iter + 2) % 4 == 0 {
                    multiplier = 14.0;
                } else {
                    multiplier = 12.0;
                }
            }

            ans += 7.0 * get_domain_change_function_value(func, &integration_limit[0], upper_limit);

            return 2.0 * delta * ans / 45.0;
        }

        let mut ans = 7.0 * self.get_booles(number_of_integrations - 1, func, integration_limit);
        let delta = (integration_limit[number_of_integrations - 1][1]
            - integration_limit[number_of_integrations - 1][0])
            / (self.total_iterations as f64);

        let mut multiplier = 32.0;

        for iter in 0..self.total_iterations - 1 {
            ans +=
                multiplier * self.get_booles(number_of_integrations - 1, func, integration_limit);

            if (iter + 2) % 2 != 0 {
                multiplier = 32.0;
            } else if (iter + 2) % 4 == 0 {
                multiplier = 14.0;
            } else {
                multiplier = 12.0
            }
        }

        ans += 7.0 * self.get_booles(number_of_integrations - 1, func, integration_limit);

        2.0 * delta * ans / 45.0
    }

    ///returns the numerical integration via Simsons 3/8th method
    ///number_of_integrations: number of times the equation needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    fn get_simpsons<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> f64 {
        if number_of_integrations == 1 {
            let (lower_limit, upper_limit) = get_domain_change_limits(&integration_limit[0]);

            let mut ans =
                get_domain_change_function_value(func, &integration_limit[0], lower_limit);

            let delta = (upper_limit - lower_limit) / (self.total_iterations as f64);

            let mut multiplier = 3.0;
            let mut current_point = lower_limit;

            for iter in 0..self.total_iterations - 1 {
                current_point += delta;
                ans += multiplier
                    * get_domain_change_function_value(func, &integration_limit[0], current_point);

                if (iter + 2) % 3 == 0 {
                    multiplier = 2.0;
                } else {
                    multiplier = 3.0;
                }
            }

            ans += get_domain_change_function_value(func, &integration_limit[0], upper_limit);

            return 3.0 * delta * ans / 8.0;
        }

        let mut ans = self.get_simpsons(number_of_integrations - 1, func, integration_limit);
        let delta = (integration_limit[number_of_integrations - 1][1]
            - integration_limit[number_of_integrations - 1][0])
            / (self.total_iterations as f64);

        let mut multiplier = 3.0;

        for iter in 0..self.total_iterations - 1 {
            ans +=
                multiplier * self.get_simpsons(number_of_integrations - 1, func, integration_limit);

            if (iter + 2) % 3 == 0 {
                multiplier = 2.0;
            } else {
                multiplier = 3.0;
            }
        }

        ans += self.get_simpsons(number_of_integrations - 1, func, integration_limit);

        3.0 * delta * ans / 8.0
    }

    ///returns the numerical integration via Trapezoidal method
    ///number_of_integrations: number of times the equation needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    fn get_trapezoidal<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> f64 {
        if number_of_integrations == 1 {
            let (lower_limit, upper_limit) = get_domain_change_limits(&integration_limit[0]);

            let mut ans =
                get_domain_change_function_value(func, &integration_limit[0], lower_limit);

            let delta = (upper_limit - lower_limit) / (self.total_iterations as f64);
            let mut current_point = lower_limit;

            for _ in 0..self.total_iterations - 1 {
                current_point += delta;

                ans += 2.0
                    * get_domain_change_function_value(func, &integration_limit[0], current_point);
            }

            ans += get_domain_change_function_value(func, &integration_limit[0], upper_limit);

            return 0.5 * delta * ans;
        }

        let mut ans = self.get_trapezoidal(number_of_integrations - 1, func, integration_limit);

        let delta = (integration_limit[number_of_integrations - 1][1]
            - integration_limit[number_of_integrations - 1][0])
            / (self.total_iterations as f64);

        for _ in 0..self.total_iterations - 1 {
            ans += 2.0 * self.get_trapezoidal(number_of_integrations - 1, func, integration_limit);
        }

        ans += self.get_trapezoidal(number_of_integrations - 1, func, integration_limit);

        0.5 * delta * ans
    }
}

/// Dispatches to the chosen rule, integrating `g` over `[lo, hi]` with `iterations`
/// intervals. The caller decides the domain branch before building `g`, so a finite
/// integral passes `func` straight through with no per-sample transform.
fn integrate_rule<T: Numeric, G: FnMut(T) -> T>(
    method: IterativeMethod,
    iterations: u64,
    lo: T,
    hi: T,
    g: G,
) -> T {
    match method {
        IterativeMethod::Booles => booles(iterations, lo, hi, g),
        IterativeMethod::Simpsons => simpsons(iterations, lo, hi, g),
        IterativeMethod::Trapezoidal => trapezoidal(iterations, lo, hi, g),
    }
}

/// Boole's composite rule over `[lo, hi]`.
fn booles<T: Numeric, G: FnMut(T) -> T>(iterations: u64, lo: T, hi: T, mut g: G) -> T {
    let delta = (hi - lo) / T::from_u64(iterations);
    let mut point = lo;

    let mut ans = T::from_f64(7.0) * g(point);
    let mut multiplier = T::from_f64(32.0);

    for iter in 0..iterations - 1 {
        point += delta;
        ans += multiplier * g(point);

        if (iter + 2) % 2 != 0 {
            multiplier = T::from_f64(32.0);
        } else if (iter + 2) % 4 == 0 {
            multiplier = T::from_f64(14.0);
        } else {
            multiplier = T::from_f64(12.0);
        }
    }

    ans += T::from_f64(7.0) * g(hi);

    T::TWO * delta * ans / T::from_f64(45.0)
}

/// Simpson's 3/8 composite rule over `[lo, hi]`.
fn simpsons<T: Numeric, G: FnMut(T) -> T>(iterations: u64, lo: T, hi: T, mut g: G) -> T {
    let delta = (hi - lo) / T::from_u64(iterations);
    let mut point = lo;

    let mut ans = g(point);
    let mut multiplier = T::from_f64(3.0);

    for iter in 0..iterations - 1 {
        point += delta;
        ans += multiplier * g(point);

        if (iter + 2) % 3 == 0 {
            multiplier = T::TWO;
        } else {
            multiplier = T::from_f64(3.0);
        }
    }

    ans += g(hi);

    T::from_f64(3.0) * delta * ans / T::from_f64(8.0)
}

/// Trapezoidal composite rule over `[lo, hi]`.
fn trapezoidal<T: Numeric, G: FnMut(T) -> T>(iterations: u64, lo: T, hi: T, mut g: G) -> T {
    let delta = (hi - lo) / T::from_u64(iterations);
    let mut point = lo;

    let mut ans = g(point);

    for _ in 0..iterations - 1 {
        point += delta;
        ans += T::TWO * g(point);
    }

    ans += g(hi);

    T::HALF * delta * ans
}

/// Implements the iterative methods for numerical integration for single variable functions
#[derive(Debug, Clone, Copy)]
pub struct IterativeSingle<T = f64> {
    pub config: IterativeConfig,
    _marker: PhantomData<T>,
}

impl<T> Default for IterativeSingle<T> {
    fn default() -> Self {
        IterativeSingle {
            config: IterativeConfig::default(),
            _marker: PhantomData,
        }
    }
}

impl<T> IterativeSingle<T> {
    /// custom constructor. Optimal for fine-tuning for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self {
        IterativeSingle {
            config: IterativeConfig::from_parameters(total_iterations, integration_method),
            _marker: PhantomData,
        }
    }
}

impl<T: Numeric> IterativeSingle<T> {
    /// Integrates the `level`-th limit (1-based). Inner folds of a single-variable
    /// integral are constant in the outer variable, so the inner result is computed
    /// once and reused; an infinite outer limit weights it by `dx/dt`. A finite limit
    /// skips the domain transform entirely.
    fn integrate<F: Fn(T) -> T, const NUM_INTEGRATIONS: usize>(
        &self,
        level: usize,
        func: &F,
        integration_limit: &[[T; 2]; NUM_INTEGRATIONS],
    ) -> T {
        let method = self.config.integration_method;
        let iterations = self.config.total_iterations;

        let domain = match classify(&integration_limit[level - 1]) {
            Ok(d) => d,
            Err(_) => return T::NAN, // limits validated in check_for_errors; unreachable
        };

        if level == 1 {
            return match domain {
                Domain::Finite(a, b) => integrate_rule(method, iterations, a, b, func),
                _ => {
                    let (lo, hi) = t_bounds(&domain);
                    integrate_rule(method, iterations, lo, hi, |t| {
                        let (x, jacobian) = map_sample(&domain, t);
                        func(x) * jacobian
                    })
                }
            };
        }

        let inner = self.integrate(level - 1, func, integration_limit);
        match domain {
            Domain::Finite(a, b) => integrate_rule(method, iterations, a, b, |_| inner),
            _ => {
                let (lo, hi) = t_bounds(&domain);
                integrate_rule(method, iterations, lo, hi, |t| {
                    let (_, jacobian) = map_sample(&domain, t);
                    inner * jacobian
                })
            }
        }
    }
}

impl<T: Numeric> IntegratorSingleVariable for IterativeSingle<T> {
    type Scalar = T;

    /// Integrates `func`, once for each limit in `integration_limit` (so the array length
    /// sets the number of integrations).
    ///
    /// A limit may be finite, or use `f64::INFINITY` / `f64::NEG_INFINITY` for an infinite or
    /// semi-infinite range. Infinite ranges are mapped onto a finite interval and are accurate
    /// only for integrands that decay toward the infinite end.
    ///
    /// # Arguments
    /// * `func` - the function to integrate.
    /// * `integration_limit` - the `[lower, upper]` limit for each level of integration.
    ///
    /// # Errors
    /// [`CalcError::IterationsZero`] if the configured iteration count is zero, or
    /// [`CalcError::IntegrationLimitsIllDefined`] if any limit is ill-defined.
    ///
    /// # Examples
    /// ```
    ///    let my_func = | arg: f64 | -> f64
    ///    {
    ///        return 2.0*arg;
    ///    };
    ///
    /// let my_func = |x: f64| 2.0 * x;
    /// let integrator = IterativeSingle::default();
    ///
    /// let integrator = iterative_integration::SingleVariableSolver::default();  
    ///
    /// let integration_limit = [[0.0, 2.0]; 1]; //desired integration limit
    /// let val = integrator.get(1, &my_func, &integration_limit).unwrap(); //single integration
    /// assert!(f64::abs(val - 4.0) < 1e-6);
    ///
    /// let integration_limit = [[0.0, 2.0], [-1.0, 1.0]]; //desired integration limits
    /// let val = integrator.get(2, &my_func, &integration_limit).unwrap(); //double integration
    /// assert!(f64::abs(val - 8.0) < 1e-6);
    ///
    /// let integration_limit = [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]; //desired integration limits
    /// let val = integrator.get(3, &my_func, &integration_limit).unwrap(); //triple integration
    /// assert!(f64::abs(val - 16.0) < 1e-6);
    ///```
    fn get<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<f64, &'static str> {
        self.check_for_errors(number_of_integrations, integration_limit)?;

        match self.integration_method {
            IterativeMethod::Booles => {
                Ok(self.get_booles(number_of_integrations, func, integration_limit))
            }
            IterativeMethod::Simpsons => {
                Ok(self.get_simpsons(number_of_integrations, func, integration_limit))
            }
            IterativeMethod::Trapezoidal => {
                Ok(self.get_trapezoidal(number_of_integrations, func, integration_limit))
            }
        }
    }
}

/// Implements the iterative methods for numerical integration for multi variable functions
#[derive(Debug, Clone, Copy)]
pub struct IterativeMulti<T = f64> {
    pub config: IterativeConfig,
    _marker: PhantomData<T>,
}

impl<T> Default for IterativeMulti<T> {
    fn default() -> Self {
        MultiVariableSolver {
            total_iterations: DEFAULT_TOTAL_ITERATIONS,
            integration_method: IterativeMethod::Booles,
        }
    }
}

impl MultiVariableSolver {
    ///returns the total number of iterations
    pub fn get_total_iterations(&self) -> u64 {
        self.total_iterations
    }

    ///sets the total number of iterations
    pub fn set_total_iterations(&mut self, total_iterations: u64) {
        self.total_iterations = total_iterations;
    }

    ///returns the chosen integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn get_integration_method(&self) -> IterativeMethod {
        self.integration_method
    }

    ///sets the integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn set_integration_method(&mut self, integration_method: IterativeMethod) {
        self.integration_method = integration_method;
    }

    ///custom constructor, optimal for fine-tuning the integrator for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self {
        MultiVariableSolver {
            total_iterations,
            integration_method,
        }
    }
}

impl<T: Numeric> IterativeMulti<T> {
    /// Integrates the `level`-th limit (1-based) of a partial integral. The sampled
    /// abscissa is written into the integrated variable's slot before recursing, and
    /// an infinite limit weights the whole inner integral by `dx/dt`. A finite limit
    /// skips the domain transform entirely.
    fn integrate<
        F: Fn(&[T; NUM_VARS]) -> T,
        const NUM_VARS: usize,
        const NUM_INTEGRATIONS: usize,
    >(
        &self,
        number_of_integrations: usize,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<(), &'static str> {
        if self.total_iterations == 0 {
            return Err(INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS);
        }

        for &limit in integration_limit {
            if limit[0] >= limit[1] {
                return Err(INTEGRATION_LIMITS_ILL_DEFINED);
            }
        }

        if NUM_INTEGRATIONS != number_of_integrations {
            return Err(INCORRECT_NUMBER_OF_INTEGRATION_LIMITS);
        }

        Ok(())
    }

    ///returns the numerical integration via Booles' method
    ///number_of_integrations: number of times the equation needs to be integrated
    /// idx_to_integrate: the variables' index/indices that needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    fn get_booles<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> f64 {
        if number_of_integrations == 1 {
            // === Base case ===
            let var_idx = idx_to_integrate[0];

            // Map possibly infinite [lower, upper] -> [transformed_lower, transformed_upper]
            let (transformed_lower_limit, transformed_upper_limit) =
                get_domain_change_limits(&integration_limits[0]);

            // Single-variable projection of the multivariable function
            let local_func = |x: f64| {
                let mut cur_vec = *point;
                cur_vec[var_idx] = x;
                func(&cur_vec)
            };

            let delta = (transformed_upper_limit - transformed_lower_limit)
                / (self.total_iterations as f64);

            let mut ans = 7.0
                * get_domain_change_function_value(
                    &local_func,
                    &integration_limits[0],
                    transformed_lower_limit,
                );

            let mut current_point = transformed_lower_limit;
            let mut multiplier = 32.0;

            for iter in 0..self.total_iterations - 1 {
                current_point += delta;

                ans += multiplier
                    * get_domain_change_function_value(
                        &local_func,
                        &integration_limits[0],
                        current_point,
                    );

                if (iter + 2) % 2 != 0 {
                    multiplier = 32.0;
                } else if (iter + 2) % 4 == 0 {
                    multiplier = 14.0;
                } else {
                    multiplier = 12.0;
                }
            }

            ans += 7.0
                * get_domain_change_function_value(
                    &local_func,
                    &integration_limits[0],
                    transformed_upper_limit,
                );

            return 2.0 * delta * ans / 45.0;
        }

        // === Recursive case ===
        let var_idx = idx_to_integrate[number_of_integrations - 1];
        let original_limit = integration_limits[number_of_integrations - 1];
        let n = self.total_iterations;

        // Map possibly infinite [lower, upper] -> [transformed_lower, transformed_upper]
        let (transformed_lower_limit, transformed_upper_limit) =
            get_domain_change_limits(&original_limit);

        let delta = (transformed_upper_limit - transformed_lower_limit) / (n as f64);

        let mapped_outer_integrand = |t: f64| {
            let inner_as_fn_of_x = |x: f64| {
                let mut cur = *point;
                cur[var_idx] = x;
                self.get_booles(
                    number_of_integrations - 1,
                    idx_to_integrate,
                    func,
                    integration_limits,
                    &cur,
                )
            };
            get_domain_change_function_value(&inner_as_fn_of_x, &original_limit, t)
        };

        let mut ans = 7.0 * mapped_outer_integrand(transformed_lower_limit);
        let mut mult = 32.0;

        for i in 1..n {
            let t = transformed_lower_limit + (i as f64) * delta;
            ans += mult * mapped_outer_integrand(t);

            // sequence: 32, 12, 32, 14, ...
            mult = if (i + 1) % 2 != 0 {
                32.0
            } else if (i + 1) % 4 == 0 {
                14.0
            } else {
                12.0
            };
        }

        ans += 7.0 * mapped_outer_integrand(transformed_upper_limit);

        2.0 * delta * ans / 45.0
    }

    /// Returns the numerical integration via Simsons' 3/8th method
    /// number_of_integrations: number of times the equation needs to be integrated
    /// idx_to_integrate: the variables' index/indices that needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    fn get_simpsons<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> f64 {
        if number_of_integrations == 1 {
            // === Base case ===
            let var_idx = idx_to_integrate[0];

            // Map possibly infinite [lower, upper] -> [transformed_lower, transformed_upper]
            let (transformed_lower_limit, transformed_upper_limit) =
                get_domain_change_limits(&integration_limits[0]);

            // Define a pure single-variable view of the multivariable function
            let local_func = |x: f64| {
                let mut cur_vec = *point;
                cur_vec[var_idx] = x;
                func(&cur_vec)
            };

            let delta = (transformed_upper_limit - transformed_lower_limit)
                / (self.total_iterations as f64);

            // Initial term
            let mut ans = get_domain_change_function_value(
                &local_func,
                &integration_limits[0],
                transformed_lower_limit,
            );

            let mut current_point = transformed_lower_limit;
            let mut multiplier = 3.0;

            // Interior points
            for iter in 0..self.total_iterations - 1 {
                current_point += delta;

                ans += multiplier
                    * get_domain_change_function_value(
                        &local_func,
                        &integration_limits[0],
                        current_point,
                    );

                if (iter + 2) % 3 == 0 {
                    multiplier = 2.0;
                } else {
                    multiplier = 3.0;
                }
            }

            // Final term
            ans += get_domain_change_function_value(
                &local_func,
                &integration_limits[0],
                transformed_upper_limit,
            );

            return 3.0 * delta * ans / 8.0;
        }

        // === Recursive case ===
        let var_idx = idx_to_integrate[number_of_integrations - 1];
        let original_limit = integration_limits[number_of_integrations - 1];
        let n = self.total_iterations;

        // Map possibly infinite [lower, upper] -> [transformed_lower, transformed_upper]
        let (transformed_lower_limit, transformed_upper_limit) =
            get_domain_change_limits(&original_limit);

        let delta = (transformed_upper_limit - transformed_lower_limit) / (n as f64);

        let mapped_outer_integrand = |t: f64| {
            let inner_as_fn_of_x = |x: f64| {
                let mut cur = *point;
                cur[var_idx] = x;
                self.get_simpsons(
                    number_of_integrations - 1,
                    idx_to_integrate,
                    func,
                    integration_limits,
                    &cur,
                )
            };

            get_domain_change_function_value(&inner_as_fn_of_x, &original_limit, t)
        };

        let mut ans = mapped_outer_integrand(transformed_lower_limit);
        let mut mult = 3.0;

        for i in 1..n {
            let t = transformed_lower_limit + (i as f64) * delta;
            ans += mult * mapped_outer_integrand(t);
            mult = if (i + 1) % 3 == 0 { 2.0 } else { 3.0 };
        }

        ans += mapped_outer_integrand(transformed_upper_limit);
        3.0 * delta * ans / 8.0
    }

    /// Returns the numerical integration via Trapezoidal method
    /// number_of_integrations: number of times the equation needs to be integrated
    /// idx_to_integrate: the variables' index/indices that needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    fn get_trapezoidal<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> f64 {
        if number_of_integrations == 1 {
            // === Base case ===
            let var_idx = idx_to_integrate[0];
            let original_integration_limits = integration_limits[0];

            // Map possibly infinite [lower, upper] -> [transformed_lower, transformed_upper]
            let (transformed_lower_limit, transformed_upper_limit) =
                get_domain_change_limits(&original_integration_limits);

            let delta = (transformed_upper_limit - transformed_lower_limit)
                / (self.total_iterations as f64);

            let mut current_point = transformed_lower_limit;

            // Helper closure: f(x_var) with other variables fixed
            let local_func = |x: f64| {
                let mut cur_vec = *point;
                cur_vec[var_idx] = x;
                func(&cur_vec)
            };

            let mut ans = get_domain_change_function_value(
                &local_func,
                &original_integration_limits,
                current_point,
            );

            for _ in 0..self.total_iterations - 1 {
                current_point += delta;

                ans += 2.0
                    * get_domain_change_function_value(
                        &local_func,
                        &original_integration_limits,
                        current_point,
                    );
            }

            ans += get_domain_change_function_value(
                &local_func,
                &original_integration_limits,
                transformed_upper_limit,
            );

            return 0.5 * delta * ans;
        }

        // === Recursive case ===
        let var_idx = idx_to_integrate[number_of_integrations - 1];
        let original_limit = integration_limits[number_of_integrations - 1];

        // Map possibly infinite [lower, upper] -> [transformed_lower, transformed_upper]
        let (transformed_lower_limit, transformed_upper_limit) =
            get_domain_change_limits(&original_limit);

        let n = self.total_iterations;
        let delta = (transformed_upper_limit - transformed_lower_limit) / (n as f64);

        // The outer integrand at parameter t is: inner_integral(x(t), other vars fixed) * |dx/dt|
        let mapped_outer_integrand = |t: f64| -> f64 {
            // local pure closure: x -> inner integral value with var_idx set to x
            let inner_as_fn_of_x = |x: f64| {
                let mut cur = *point;
                cur[var_idx] = x;

                // recurse for remaining dimensions
                self.get_trapezoidal(
                    number_of_integrations - 1,
                    idx_to_integrate,
                    func,
                    integration_limits,
                    &cur,
                )
            };

            // apply x(t) mapping and Jacobian
            get_domain_change_function_value(&inner_as_fn_of_x, &original_limit, t)
        };

        // Trapezoidal rule over the transformed t-domain
        let mut ans = mapped_outer_integrand(transformed_lower_limit);

        for i in 1..n {
            let t = transformed_lower_limit + (i as f64) * delta;
            ans += 2.0 * mapped_outer_integrand(t);
        }

        ans += mapped_outer_integrand(transformed_upper_limit);

        0.5 * delta * ans
    }
}

impl<T: Numeric> IntegratorMultiVariable for IterativeMulti<T> {
    type Scalar = T;

    /// Partially integrates `func` over the variables in `idx_to_integrate`, once for each
    /// limit in `integration_limits` (so the array length sets the number of integrations).
    ///
    /// # Arguments
    /// * `idx_to_integrate` - the variable index integrated at each level.
    /// * `func` - the function to integrate.
    /// * `integration_limits` - the `[lower, upper]` limit for each level of integration.
    /// * `point` - the value of every variable. A variable being integrated holds its final
    ///   upper limit; a variable held constant holds that constant.
    ///
    /// # Errors
    /// [`CalcError::IterationsZero`] if the configured iteration count is zero, or
    /// [`CalcError::IntegrationLimitsIllDefined`] if any limit is ill-defined.
    ///
    /// # Examples
    /// ```
    /// let func = | args: &[f64; 3] | -> f64
    ///{
    ///    return 2.0*args[0] + args[1]*args[2];
    ///};
    /// let point = [1.0, 2.0, 3.0];
    /// let integrator = IterativeMulti::default();
    ///
    /// use crate::multicalc::numerical_integration::integrator::*;
    /// use multicalc::numerical_integration::iterative_integration;
    ///
    /// let integrator = iterative_integration::MultiVariableSolver::default();
    ///
    /// let integration_limit = [[0.0, 1.0]; 1]; //desired integation limit
    /// let val = integrator.get(1, [0; 1], &func, &integration_limit, &point).unwrap();
    /// assert!(f64::abs(val - 7.0) < 1e-6);
    /// ```
    fn get<F: Fn(&[T; NUM_VARS]) -> T, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str> {
        self.check_for_errors(number_of_integrations, integration_limits)?;

        match self.integration_method {
            IterativeMethod::Booles => Ok(self.get_booles(
                number_of_integrations,
                idx_to_integrate,
                func,
                integration_limits,
                point,
            )),
            IterativeMethod::Simpsons => Ok(self.get_simpsons(
                number_of_integrations,
                idx_to_integrate,
                func,
                integration_limits,
                point,
            )),
            IterativeMethod::Trapezoidal => Ok(self.get_trapezoidal(
                number_of_integrations,
                idx_to_integrate,
                func,
                integration_limits,
                point,
            )),
        }
    }
}

use core::marker::PhantomData;

use crate::numeric::Numeric;
use crate::numerical_integration::integrator::*;
use crate::numerical_integration::mode::IterativeMethod;
use crate::utils::error_codes::CalcError;

/// Default interval count. A multiple of 12 so Boole (needs a multiple of 4) and
/// Simpson 3/8 (needs a multiple of 3) both align with the composite-rule weights.
pub const DEFAULT_TOTAL_ITERATIONS: u64 = 120;

/// @brief Implements the iterative methods for numerical integration for single variable functions.
#[derive(Clone, Copy)]
pub struct SingleVariableSolver {
    total_iterations: u64,
    integration_method: IterativeMethod,
}

impl Default for SingleVariableSolver {
    /// @brief Default constructor, optimal for most generic equations.
    fn default() -> Self {
        SingleVariableSolver {
            total_iterations: DEFAULT_TOTAL_ITERATIONS,
            integration_method: IterativeMethod::Booles,
        }
    }
}

impl SingleVariableSolver {
    /// @brief Returns the total nuber of iterations.
    pub fn get_total_iterations(&self) -> u64 {
        self.total_iterations
    }

    /// @brief Sets the total nuber of iterations.
    pub fn set_total_iterations(&mut self, total_iterations: u64) {
        self.total_iterations = total_iterations;
    }

    /// @brief Returns the chosen integration method.
    /// @note Possible choices are: Booles, Simpsons and Trapezoidal.
    pub fn get_integration_method(&self) -> IterativeMethod {
        self.integration_method
    }

    /// @brief Sets the integration method.
    pub fn set_integration_method(&mut self, integration_method: IterativeMethod) {
        self.integration_method = integration_method;
    }

    /// @brief Custom constructor. Optimal for fine-tuning for more complex equations.
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

    /// @brief Returns the numerical integration via Simsons 3/8th method.
    /// @param number_of_integrations: number of times the equation needs to be integrated.
    /// @param func: The function to integrate.
    /// @integration_limit: the integration bound(s) for each round of integration.
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

    /// @brief Returns the numerical integration via Trapezoidal method.
    /// @param number_of_integrations: number of times the equation needs to be integrated.
    /// @param func: The function to integrate.
    /// @integration_limit: the integration bound(s) for each round of integration.
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

impl<T: Numeric> IntegratorSingleVariable for IterativeSingle<T> {
    type Scalar = T;

    /// Integrates `func`, once for each limit in `integration_limit` (so the array length
    /// sets the number of integrations).
    ///
    /// @note: Returns a Result<f64, &'static str>,
    /// where possible Err are:
    /// INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS -> if number_of_integrations is zero
    /// INTEGRATION_LIMITS_ILL_DEFINED -> if any integration_limit[i][0] >= integration_limit[i][1] for all possible i
    /// INCORRECT_NUMBER_OF_INTEGRATION_LIMITS -> if size of integration_limit is not equal to number_of_integrations
    ///
    /// @example Assume we want to integrate 2*x . the function would be:
    /// ```
    /// use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
    /// use multicalc::numerical_integration::iterative_integration::IterativeSingle;
    ///
    /// let my_func = |x: f64| 2.0 * x;
    /// let integrator = IterativeSingle::default();
    ///
    /// // single integration of 2x over [0, 2] is 4
    /// let val = integrator.get(&my_func, &[[0.0, 2.0]; 1]).unwrap();
    /// assert!(f64::abs(val - 4.0) < 1e-6);
    ///
    /// // double integration over [0, 2] then [-1, 1] is 8
    /// let val = integrator.get(&my_func, &[[0.0, 2.0], [-1.0, 1.0]]).unwrap();
    /// assert!(f64::abs(val - 8.0) < 1e-6);
    ///
    /// // an infinite limit, for a decaying integrand: integral of e^(-x^2) over the real line is sqrt(pi)
    /// let val = integrator.get(&|x| (-x * x).exp(), &[[f64::NEG_INFINITY, f64::INFINITY]]).unwrap();
    /// assert!(f64::abs(val - std::f64::consts::PI.sqrt()) < 1e-6);
    /// ```
    fn get<F: Fn(T) -> T, const NUM_INTEGRATIONS: usize>(
        &self,
        func: &F,
        integration_limit: &[[T; 2]; NUM_INTEGRATIONS],
    ) -> Result<T, CalcError> {
        self.config.check_for_errors(integration_limit)?;
        Ok(self.integrate(NUM_INTEGRATIONS, func, integration_limit))
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
        IterativeMulti {
            config: IterativeConfig::default(),
            _marker: PhantomData,
        }
    }
}

impl<T> IterativeMulti<T> {
    /// custom constructor, optimal for fine-tuning the integrator for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self {
        IterativeMulti {
            config: IterativeConfig::from_parameters(total_iterations, integration_method),
            _marker: PhantomData,
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
        level: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[T; 2]; NUM_INTEGRATIONS],
        point: &[T; NUM_VARS],
    ) -> T {
        let method = self.config.integration_method;
        let iterations = self.config.total_iterations;

        let domain = match classify(&integration_limits[level - 1]) {
            Ok(d) => d,
            Err(_) => return T::NAN, // limits validated in check_for_errors; unreachable
        };
        let var = idx_to_integrate[level - 1];

        if level == 1 {
            let mut current = *point;
            return match domain {
                Domain::Finite(a, b) => integrate_rule(method, iterations, a, b, |x| {
                    current[var] = x;
                    func(&current)
                }),
                _ => {
                    let (lo, hi) = t_bounds(&domain);
                    integrate_rule(method, iterations, lo, hi, |t| {
                        let (x, jacobian) = map_sample(&domain, t);
                        current[var] = x;
                        func(&current) * jacobian
                    })
                }
            };
        }

        let mut current = *point;
        match domain {
            Domain::Finite(a, b) => integrate_rule(method, iterations, a, b, |x| {
                current[var] = x;
                self.integrate(
                    level - 1,
                    idx_to_integrate,
                    func,
                    integration_limits,
                    &current,
                )
            }),
            _ => {
                let (lo, hi) = t_bounds(&domain);
                integrate_rule(method, iterations, lo, hi, |t| {
                    let (x, jacobian) = map_sample(&domain, t);
                    current[var] = x;
                    let inner = self.integrate(
                        level - 1,
                        idx_to_integrate,
                        func,
                        integration_limits,
                        &current,
                    );
                    inner * jacobian
                })
            }
        }
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
    /// @example Assume we want to integrate 2.0*x + y*z . the function would be:
    /// ```
    /// use multicalc::numerical_integration::integrator::IntegratorMultiVariable;
    /// use multicalc::numerical_integration::iterative_integration::IterativeMulti;
    ///
    /// // f(x, y, z) = 2x + yz, integrated over x in [0, 1] with (y, z) = (2, 3); result is 7
    /// let func = |args: &[f64; 3]| 2.0 * args[0] + args[1] * args[2];
    /// let point = [1.0, 2.0, 3.0];
    /// let integrator = IterativeMulti::default();
    ///
    /// let val = integrator.get([0; 1], &func, &[[0.0, 1.0]; 1], &point).unwrap();
    /// assert!(f64::abs(val - 7.0) < 1e-6);
    /// ```
    fn get<F: Fn(&[T; NUM_VARS]) -> T, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[T; 2]; NUM_INTEGRATIONS],
        point: &[T; NUM_VARS],
    ) -> Result<T, CalcError> {
        self.config.check_for_errors(integration_limits)?;
        Ok(self.integrate(
            NUM_INTEGRATIONS,
            idx_to_integrate,
            func,
            integration_limits,
            point,
        ))
    }
}

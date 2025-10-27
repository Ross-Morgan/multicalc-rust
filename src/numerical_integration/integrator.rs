use crate::numeric::Numeric;
use crate::utils::error_codes::CalcError;

/// Classification of an integration interval, distinguishing finite domains from the
/// three infinite/semi-infinite shapes that need a domain transform.
pub(crate) enum Domain<T: Numeric> {
    Finite(T, T),
    LowerToInf(T),
    UpperToInf(T),
    BothInf,
}

/// Validates a single integration limit and classifies its domain.
///
/// Rejects `NaN` limits, equal/reversed finite limits, and infinite limits whose
/// finite end points the wrong way (e.g. `(a, -inf)` or `(+inf, b)`).
pub(crate) fn classify<T: Numeric>(limit: &[T; 2]) -> Result<Domain<T>, CalcError> {
    let (a, b) = (limit[0], limit[1]);
    if a.is_nan() || b.is_nan() {
        return Err(CalcError::IntegrationLimitsIllDefined);
    }
    match (a.is_finite(), b.is_finite()) {
        (true, true) if a < b => Ok(Domain::Finite(a, b)),
        (true, false) if b > T::ZERO => Ok(Domain::LowerToInf(a)), // (a, +inf); rejects (a, -inf)
        (false, true) if a < T::ZERO => Ok(Domain::UpperToInf(b)), // (-inf, b); rejects (+inf, b)
        (false, false) if a < T::ZERO && b > T::ZERO => Ok(Domain::BothInf), // (-inf, +inf)
        _ => Err(CalcError::IntegrationLimitsIllDefined),
    }
}

/// Returns the `t`-interval the rule walks for a domain. The finite end of a
/// semi-infinite domain sits at `t = 0` and is perfectly regular, so it is included;
/// only an infinite end needs the `T::EPSILON` inset that keeps the transform away from
/// its singular limit.
pub(crate) fn t_bounds<T: Numeric>(d: &Domain<T>) -> (T, T) {
    match d {
        Domain::Finite(a, b) => (*a, *b),
        Domain::LowerToInf(_) => (T::ZERO, T::ONE - T::EPSILON), // finite end t=0, +inf at t=1
        Domain::UpperToInf(_) => (T::ZERO, T::ONE - T::EPSILON), // finite end t=0, -inf at t=1
        Domain::BothInf => (T::EPSILON, T::ONE - T::EPSILON),
    }
}

/// Maps a sample `t` to its position `x` and the Jacobian `dx/dt` for a domain.
/// Finite domains are the identity, so the finite path pays nothing extra.
pub(crate) fn map_sample<T: Numeric>(d: &Domain<T>, t: T) -> (T, T) {
    match *d {
        Domain::Finite(_, _) => (t, T::ONE),
        Domain::LowerToInf(a) => {
            let q = T::ONE - t;
            (a + t / q, T::ONE / (q * q))
        }
        Domain::UpperToInf(b) => {
            let q = T::ONE - t;
            (b - t / q, T::ONE / (q * q))
        }
        Domain::BothInf => {
            let u = T::PI * (t - T::HALF);
            let c = u.cos();
            (u.tan(), T::PI / (c * c))
        }
    }
}

/// Base trait for single variable numerical integration.
pub trait IntegratorSingleVariable {
    /// The scalar the integral is computed in.
    type Scalar: Numeric;

    /// Generic n-th integration of a single variable function. The number of
    /// integrations equals the length of `integration_limit`.
    ///
    /// # Errors
    /// [`CalcError::IterationsZero`] if the configured iteration count is zero, or
    /// [`CalcError::IntegrationLimitsIllDefined`] if any limit is ill-defined.
    fn get<F: Fn(Self::Scalar) -> Self::Scalar, const NUM_INTEGRATIONS: usize>(
        &self,
        func: &F,
        integration_limit: &[[Self::Scalar; 2]; NUM_INTEGRATIONS],
    ) -> Result<Self::Scalar, CalcError>;

    /// Convenience wrapper for a single integral of a single variable function.
    fn get_single<F: Fn(Self::Scalar) -> Self::Scalar>(
        &self,
        func: &F,
        integration_limit: &[Self::Scalar; 2],
    ) -> Result<Self::Scalar, CalcError> {
        self.get(func, &[*integration_limit])
    }

    /// Convenience wrapper for a double integral of a single variable function.
    fn get_double<F: Fn(Self::Scalar) -> Self::Scalar>(
        &self,
        func: &F,
        integration_limit: &[[Self::Scalar; 2]; 2],
    ) -> Result<Self::Scalar, CalcError> {
        self.get(func, integration_limit)
    }
}

/// Base trait for multi-variable numerical integration.
pub trait IntegratorMultiVariable {
    /// The scalar the integral is computed in.
    type Scalar: Numeric;

    /// Generic n-th partial integration of a multi variable function. The number of
    /// integrations equals the length of `integration_limits`.
    ///
    /// @tparam NUM_VARS Number of variables in the function.
    /// @tparam NUM_INTEGRATIONS Number of integrations to perform.
    ///
    /// # Errors
    /// [`CalcError::IterationsZero`] if the configured iteration count is zero, or
    /// [`CalcError::IntegrationLimitsIllDefined`] if any limit is ill-defined.
    fn get<
        F: Fn(&[Self::Scalar; NUM_VARS]) -> Self::Scalar,
        const NUM_VARS: usize,
        const NUM_INTEGRATIONS: usize,
    >(
        &self,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[Self::Scalar; 2]; NUM_INTEGRATIONS],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, CalcError>;

    /// Convenience wrapper for a single partial integral of a multi variable function.
    fn get_single_partial<
        F: Fn(&[Self::Scalar; NUM_VARS]) -> Self::Scalar,
        const NUM_VARS: usize,
    >(
        &self,
        func: &F,
        idx_to_integrate: usize,
        integration_limits: &[Self::Scalar; 2],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, CalcError> {
        self.get([idx_to_integrate], func, &[*integration_limits], point)
    }

    /// Convenience wrapper for a double partial integral of a multi variable function.
    fn get_double_partial<
        F: Fn(&[Self::Scalar; NUM_VARS]) -> Self::Scalar,
        const NUM_VARS: usize,
    >(
        &self,
        func: &F,
        idx_to_integrate: [usize; 2],
        integration_limits: &[[Self::Scalar; 2]; 2],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, CalcError> {
        self.get(idx_to_integrate, func, integration_limits, point)
    }
}

/// Evaluates the transformed function value `f(x(t)) * dx/dt`
/// according to the correct domain-mapping rule.
///
/// Mapping functions:
/// 1. (-∞, ∞):  x = tan(π(t - ½)),     dx/dt = π / cos²(π(t - ½))
/// 2. (a, ∞):   x = a + t/(1 - t),     dx/dt = 1 / (1 - t)²
/// 3. (-∞, b):  x = b - t/(1 - t),     dx/dt = 1 / (1 - t)²
/// 4. Finite (a, b):  x = t,           dx/dt = 1
///
/// # Arguments
/// - `func`: the original function f(x)
/// - `original_integration_limit`: `[a, b]` integration range, possibly infinite
/// - `point`: current evaluation point `t` in the transformed finite domain [0, 1]
///
/// # Returns
/// The function value in transformed domain: f(x(t)) * dx/dt
pub fn get_domain_change_function_value(
    func: &dyn Fn(f64) -> f64,
    original_integration_limit: &[f64; 2],
    point: f64,
) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    let lower_limit = original_integration_limit[0];
    let upper_limit = original_integration_limit[1];

    if lower_limit.is_infinite() && upper_limit.is_infinite() {
        // (-∞, ∞)

        let x = f64::tan(PI * (point - 0.5));
        let jac = PI / f64::cos(PI * (point - 0.5)).powi(2);
        func(x) * jac
    } else if lower_limit.is_finite() && upper_limit.is_infinite() {
        // (lower_limit, ∞)

        let x = lower_limit + point / (1.0 - point);
        let jac = 1.0 / (1.0 - point).powi(2);
        func(x) * jac
    } else if lower_limit.is_infinite() && upper_limit.is_finite() {
        // (-∞, upper_limit)

        let x = upper_limit - point / (1.0 - point);
        let jac = 1.0 / (1.0 - point).powi(2);
        func(x) * jac
    } else {
        func(point)
    }
}

/// Returns the transformed integration limits `(t0, t1)`
/// that map an infinite or semi-infinite domain to a finite one.
///
/// Mapping rules:
/// - Finite (a, b): unchanged → (a, b)
/// - Semi-infinite (a, ∞): t ∈ [0, 1)
/// - Semi-infinite (-∞, b): t ∈ [0, 1)
/// - Infinite (-∞, ∞): t ∈ [0, 1)
pub fn get_domain_change_limits(original_integration_limit: &[f64; 2]) -> (f64, f64) {
    let a = original_integration_limit[0];
    let b = original_integration_limit[1];

    const EPSILON: f64 = core::f64::EPSILON;

    if a.is_infinite() || b.is_infinite() {
        // For all infinite forms, map into [0, 1]
        (EPSILON, 1.0 - EPSILON) //don't actually map to [0,1] but very close to it, because evaluating at infinity gives undefined behavior
    } else {
        // Finite range: leave unchanged
        (a, b)
    }
}

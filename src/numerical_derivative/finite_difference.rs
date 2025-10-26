use crate::numerical_derivative::derivator::*;
use crate::numerical_derivative::mode;
use crate::utils::error_codes::*;

use crate::numeric::Numeric;
use crate::numerical_derivative::derivator::{DerivatorMultiVariable, DerivatorSingleVariable};
use crate::numerical_derivative::mode::{self, FiniteDifferenceMode};
use crate::utils::error_codes::CalcError;

/// Low and high sample offsets (in units of the step size) and the divisor factor
/// for each finite-difference mode.
#[inline]
fn offsets<T: Numeric>(method: FiniteDifferenceMode) -> (T, T, T) {
    match method {
        FiniteDifferenceMode::Forward => (T::ZERO, T::ONE, T::ONE),
        FiniteDifferenceMode::Backward => (-T::ONE, T::ZERO, T::ONE),
        FiniteDifferenceMode::Central => (-T::ONE, T::ONE, T::TWO),
    }
}

/// Configuration shared by the single- and multi-variable finite-difference differentiators.
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifferenceConfig<T = f64> {
    /// The finite-difference step size. See [`mode::DEFAULT_STEP_SIZE`].
    pub step_size: T,
    /// Forward, Backward or Central difference.
    pub method: FiniteDifferenceMode,
    /// Factor the step is scaled by on each recursion level; only matters for third
    /// derivatives and higher. See [`mode::DEFAULT_STEP_SIZE_MULTIPLIER`].
    pub step_size_multiplier: T,
}

impl<T: Numeric> Default for FiniteDifferenceConfig<T> {
    /// Central difference with the default step size and multiplier; best for most cases.
    fn default() -> Self {
        FiniteDifferenceConfig {
            step_size: T::from_f64(mode::DEFAULT_STEP_SIZE),
            method: FiniteDifferenceMode::Central,
            step_size_multiplier: T::from_f64(mode::DEFAULT_STEP_SIZE_MULTIPLIER),
        }
    }
}

impl<T: Numeric> FiniteDifferenceConfig<T> {
    /// Builds a config with explicit parameters.
    pub fn from_parameters(step: T, method: FiniteDifferenceMode, multiplier: T) -> Self {
        FiniteDifferenceConfig {
            step_size: step,
            method,
            step_size_multiplier: multiplier,
        }
    }

    /// Returns [`CalcError::StepSizeZero`] if the step size is zero.
    fn check_step_size(&self) -> Result<(), CalcError> {
        if self.step_size == T::ZERO {
            return Err(CalcError::StepSizeZero);
        }
        Ok(())
    }

/// Finite-difference differentiator for single-variable functions.
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifferenceSingle<T = f64> {
    pub config: FiniteDifferenceConfig<T>,
}

impl<T: Numeric> Default for FiniteDifferenceSingle<T> {
    fn default() -> Self {
        FiniteDifferenceSingle {
            config: FiniteDifferenceConfig::default(),
        }
    }
}

impl<T: Numeric> FiniteDifferenceSingle<T> {
    /// Builds a differentiator with explicit parameters.
    pub fn from_parameters(step: T, method: FiniteDifferenceMode, multiplier: T) -> Self {
        FiniteDifferenceSingle {
            config: FiniteDifferenceConfig::from_parameters(step, method, multiplier),
        }
    }

    #[inline]
    fn diff<F: Fn(T) -> T>(&self, order: usize, func: &F, point: T, step: T) -> T {
        let (lo, hi, denom) = offsets::<T>(self.config.method);

        if order == 1 {
            let f0_args = point;
            let mut f1_args = *point;
            f1_args[idx_to_derivate[0]] += step_size;

            let f0 = func(f0_args);
            let f1 = func(&f1_args);

            return (f1 - f0) / step_size;
        }

        let mut f1_args = *point;
        f1_args[idx_to_derivate[order - 1]] += step_size;

        let f0 = self.get_forward_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            point,
            self.step_size_multiplier * step_size,
        );
        let f1 = self.get_forward_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            &f1_args,
            self.step_size_multiplier * step_size,
        );

        (f1 - f0) / step_size
    }

    /// @brief Returns the partial backward difference for multi-variable functions.
    ///
    /// Computes f'(X) = (f(X) - f(X - h)) / h, where h is the chosen step size.
    ///
    /// @param order The derivative order.
    /// @param func The multi-variable function.
    /// @param idx_to_derivate Array of variable indices to differentiate with respect to.
    /// @param point The evaluation point.
    /// @param step_size The step size.
    ///
    /// @return The computed numerical derivative.
    fn get_backward_difference_multi_variable<const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
        step_size: f64,
    ) -> f64 {
        if order == 1 {
            let mut f0_args = *point;
            f0_args[idx_to_derivate[0]] -= step_size;
            let f1_args = point;

            let f0 = func(&f0_args);
            let f1 = func(f1_args);

            return (f1 - f0) / step_size;
        }

        let mut f0_args = *point;
        f0_args[idx_to_derivate[order - 1]] -= step_size;

        let f0 = self.get_backward_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            &f0_args,
            self.step_size_multiplier * step_size,
        );
        let f1 = self.get_backward_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            point,
            self.step_size_multiplier * step_size,
        );

        (f1 - f0) / step_size
    }

    /// @brief Returns the partial central difference for multi-variable functions.
    ///
    /// Computes f'(X) = (f(X + h) - f(X - h)) / (2h), where h is the chosen step size.
    ///
    /// @param order The derivative order.
    /// @param func The multi-variable function.
    /// @param idx_to_derivate Array of variable indices to differentiate with respect to.
    /// @param point The evaluation point.
    /// @param step_size The step size.
    ///
    /// @return The computed numerical derivative.
    fn get_central_difference_multi_variable<const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
        step_size: f64,
    ) -> f64 {
        if order == 1 {
            let mut f0_args = *point;
            f0_args[idx_to_derivate[0]] -= step_size;

            let mut f1_args = *point;
            f1_args[idx_to_derivate[0]] += step_size;

            let f0 = func(&f0_args);
            let f1 = func(&f1_args);

            return (f1 - f0) / (2.0 * step_size);
        }

        let mut f0_point = *point;
        f0_point[idx_to_derivate[order - 1]] -= step_size;

        let f0 = self.get_central_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            &f0_point,
            self.step_size_multiplier * step_size,
        );

        let mut f1_point = *point;
        f1_point[idx_to_derivate[order - 1]] += step_size;

        let f1 = self.get_central_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            &f1_point,
            self.step_size_multiplier * step_size,
        );

        (f1 - f0) / (2.0 * step_size)
    }

impl<T: Numeric> DerivatorSingleVariable for FiniteDifferenceSingle<T> {
    type Scalar = T;

    fn get<F: Fn(T) -> T>(&self, order: usize, func: &F, point: T) -> Result<T, CalcError> {
        if order == 0 {
            return Err(CalcError::DerivativeOrderZero);
        }
        self.config.check_step_size()?;
        Ok(self.diff(order, func, point, self.config.step_size))
    }
}

        for &idx in idx_to_derivate {
            if idx >= point.len() {
                return Err(INDEX_TO_DERIVATIVE_OUT_OF_RANGE);
            }
        }

        match self.method {
            mode::FiniteDifferenceMode::Forward => Ok(self.get_forward_difference_multi_variable(
                order,
                func,
                idx_to_derivate,
                point,
                self.step_size,
            )),
            mode::FiniteDifferenceMode::Backward => Ok(self
                .get_backward_difference_multi_variable(
                    order,
                    func,
                    idx_to_derivate,
                    point,
                    self.step_size,
                )),
            mode::FiniteDifferenceMode::Central => Ok(self.get_central_difference_multi_variable(
                order,
                func,
                idx_to_derivate,
                point,
                self.step_size,
            )),
        }
        self.config.check_step_size()?;
        Ok(self.diff(order, func, point, self.config.step_size))
    }
}

/// Finite-difference differentiator for multi-variable functions.
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifferenceMulti<T = f64> {
    pub config: FiniteDifferenceConfig<T>,
}

impl<T: Numeric> Default for FiniteDifferenceMulti<T> {
    fn default() -> Self {
        FiniteDifferenceMulti {
            config: FiniteDifferenceConfig::default(),
        }
    }
}

impl<T: Numeric> FiniteDifferenceMulti<T> {
    /// Builds a differentiator with explicit parameters.
    pub fn from_parameters(step: T, method: FiniteDifferenceMode, multiplier: T) -> Self {
        FiniteDifferenceMulti {
            config: FiniteDifferenceConfig::from_parameters(step, method, multiplier),
        }
    }

    #[inline]
    fn diff<F: Fn(&[T; NUM_VARS]) -> T, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[T; NUM_VARS],
        step: T,
    ) -> T {
        let (lo, hi, denom) = offsets::<T>(self.config.method);
        let var = idx_to_differentiate[order - 1];

        let mut low_point = *point;
        low_point[var] += lo * step;
        let mut high_point = *point;
        high_point[var] += hi * step;

        if order == 1 {
            return (func(&high_point) - func(&low_point)) / (denom * step);
        }

        let next = self.config.step_size_multiplier * step;
        let low = self.diff(order - 1, func, idx_to_differentiate, &low_point, next);
        let high = self.diff(order - 1, func, idx_to_differentiate, &high_point, next);
        (high - low) / (denom * step)
    }
}

impl<T: Numeric> DerivatorMultiVariable for FiniteDifferenceMulti<T> {
    type Scalar = T;

    fn get<F: Fn(&[T; NUM_VARS]) -> T, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[T; NUM_VARS],
    ) -> Result<T, CalcError> {
        if NUM_ORDER == 0 {
            return Err(CalcError::DerivativeOrderZero);
        }
        self.config.check_step_size()?;
        for &idx in idx_to_differentiate {
            if idx >= NUM_VARS {
                return Err(CalcError::IndexOutOfRange);
            }
        }
        Ok(self.diff(
            NUM_ORDER,
            func,
            idx_to_differentiate,
            point,
            self.config.step_size,
        ))
    }
}

impl<T: Numeric> FiniteDifferenceMulti<T> {
    /// Builds a differentiator with explicit parameters.
    pub fn from_parameters(step: T, method: FiniteDifferenceMode, multiplier: T) -> Self {
        FiniteDifferenceMulti {
            config: FiniteDifferenceConfig::from_parameters(step, method, multiplier),
        }
    }

    #[inline]
    fn diff<F: Fn(&[T; NUM_VARS]) -> T, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[T; NUM_VARS],
        step: T,
    ) -> T {
        let (lo, hi, denom) = offsets::<T>(self.config.method);
        let var = idx_to_differentiate[order - 1];

        let mut low_point = *point;
        low_point[var] += lo * step;
        let mut high_point = *point;
        high_point[var] += hi * step;

        if order == 1 {
            return (func(&high_point) - func(&low_point)) / (denom * step);
        }

        let next = self.config.step_size_multiplier * step;
        let low = self.diff(order - 1, func, idx_to_differentiate, &low_point, next);
        let high = self.diff(order - 1, func, idx_to_differentiate, &high_point, next);
        (high - low) / (denom * step)
    }
}

impl<T: Numeric> DerivatorMultiVariable for FiniteDifferenceMulti<T> {
    type Scalar = T;

    fn get<F: Fn(&[T; NUM_VARS]) -> T, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[T; NUM_VARS],
    ) -> Result<T, CalcError> {
        if NUM_ORDER == 0 {
            return Err(CalcError::DerivativeOrderZero);
        }
        self.config.check_step_size()?;
        for &idx in idx_to_differentiate {
            if idx >= NUM_VARS {
                return Err(CalcError::IndexOutOfRange);
            }
        }
        Ok(self.diff(
            NUM_ORDER,
            func,
            idx_to_differentiate,
            point,
            self.config.step_size,
        ))
    }
}

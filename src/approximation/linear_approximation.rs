use crate::numeric::Numeric;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use const_poly::function_approximations;

/// A first-order (linear) Taylor approximation of a function about a base point:
/// `f(x) ≈ value + Σ gradient[i] * (x[i] - point[i])`.
#[derive(Debug, Clone, Copy)]
pub struct LinearApproximation<const NUM_VARS: usize, T = f64> {
    point: [T; NUM_VARS],
    value: T,
    gradient: [T; NUM_VARS],
}

/// Goodness-of-fit metrics for a [`LinearApproximation`] over a set of sample points.
#[derive(Debug, Clone, Copy)]
pub struct LinearApproximationPredictionMetrics<T = f64> {
    /// Mean absolute error.
    pub mean_absolute_error: T,
    /// Mean squared error.
    pub mean_squared_error: T,
    /// Root mean squared error.
    pub root_mean_squared_error: T,
    /// Coefficient of determination; `NaN` when the truth is constant over the points.
    pub r_squared: T,
    /// R² adjusted for the number of predictors; `NaN` when there are too few points.
    pub adjusted_r_squared: T,
}

impl<const NUM_VARS: usize, T: Numeric> LinearApproximation<NUM_VARS, T> {
    /// Evaluates the approximation at `x`.
    pub fn predict(&self, x: &[T; NUM_VARS]) -> T {
        let mut result = self.value;
        for ((&g, &xi), &pi) in self.gradient.iter().zip(x).zip(&self.point) {
            result += g * (xi - pi);
        }

        result
    }

    /// The base point the approximation is centered on.
    pub fn point(&self) -> &[T; NUM_VARS] {
        &self.point
    }

    /// The gradient at the base point. These are also the coefficients of the expanded
    /// linear form `intercept + Σ coefficients[i] * x[i]`.
    pub fn coefficients(&self) -> &[T; NUM_VARS] {
        &self.gradient
    }

    /// The intercept of the expanded form `intercept + Σ coefficients[i] * x[i]`.
    pub fn intercept(&self) -> T {
        let mut intercept = self.value;
        for i in 0..NUM_VARS {
            intercept -= self.gradient[i] * self.point[i];
        }
        intercept
    }

    /// Computes goodness-of-fit metrics against `original_function` over `points`.
    ///
    /// `r_squared` is `NaN` when the truth is constant over `points`;
    /// `adjusted_r_squared` is `NaN` when there are too few points.
    pub fn get_prediction_metrics<O: Fn(&[T; NUM_VARS]) -> T, const NUM_POINTS: usize>(
        &self,
        points: &[[T; NUM_VARS]; NUM_POINTS],
        original_function: &O,
    ) -> LinearApproximationPredictionMetrics<T> {
        let (mae, mse, rmse, r_squared, adjusted_r_squared) = crate::approximation::compute_metrics(
            |x| self.predict(x),
            points,
            original_function,
            NUM_VARS, // p = N linear coefficients
        );

        LinearApproximationPredictionMetrics {
            mean_absolute_error: mae.abs(),
            mean_squared_error: mse.abs(),
            root_mean_squared_error: rmse,
            r_squared: r2.abs(),
            adjusted_r_squared: r2_adj.abs(),
        }
    }
}

/// Builds a [`LinearApproximation`] of a function, using any derivator that implements
/// [`DerivatorMultiVariable`].
pub struct LinearApproximator<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable + Default> Default for LinearApproximator<D> {
    fn default() -> Self {
        LinearApproximator {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> LinearApproximator<D> {
    /// Builds an approximator from an explicit derivator.
    pub fn from_derivator(derivator: D) -> Self {
        LinearApproximator { derivator }
    }

    /// Builds a linear (first-order Taylor) approximation of `function` about `point`.
    ///
    /// # Errors
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::approximation::linear_approximation::LinearApproximator;
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    ///
    /// // x + y^2 + z^3
    /// let function_to_approximate = | args: &[f64; 3] | -> f64
    /// {
    ///     return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
    /// };
    ///
    /// let point = [1.0, 2.0, 3.0]; //the point we want to linearize around
    /// let approximator = LinearApproximator::<FiniteDifferenceMulti>::default();
    /// let result = approximator.get(&function_to_approximate, &point).unwrap();
    ///
    pub fn get<const NUM_VARS: usize>(
        &self,
        function: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        point: &[f64; NUM_VARS],
    ) -> Result<LinearApproximationResult<NUM_VARS>, &'static str> {
        let mut slopes_ = [0.0; NUM_VARS];

        let mut intercept_ = function(point);

        for iter in 0..NUM_VARS {
            slopes_[iter] = self.derivator.get(1, function, &[iter], point)?;
            intercept_ -= slopes_[iter] * point[iter];
        }

        Ok(LinearApproximationResult {
            intercept: intercept_,
            coefficients: slopes_,
        })
    }
}

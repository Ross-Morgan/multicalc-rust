use crate::numeric::Numeric;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::numerical_derivative::hessian::Hessian;
use const_poly::function_approximations;

/// A second-order (quadratic) Taylor approximation of a function about a base point:
/// `f(x) ≈ value + Σ gradient[i]·dx[i] + ½ Σ_i Σ_j hessian[i][j]·dx[i]·dx[j]`,
/// where `dx[i] = x[i] - point[i]`.
#[derive(Debug, Clone, Copy)]
pub struct QuadraticApproximation<const NUM_VARS: usize, T = f64> {
    point: [T; NUM_VARS],
    value: T,
    gradient: [T; NUM_VARS],
    hessian: [[T; NUM_VARS]; NUM_VARS],
}

/// Goodness-of-fit metrics for a [`QuadraticApproximation`] over a set of sample points.
#[derive(Debug, Clone, Copy)]
pub struct QuadraticApproximationPredictionMetrics<T = f64> {
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

impl<const NUM_VARS: usize, T: Numeric> QuadraticApproximation<NUM_VARS, T> {
    /// Evaluates the approximation at `x`. The `½` keeps the quadratic term correct for
    /// both diagonal and off-diagonal Hessian entries.
    pub fn predict(&self, x: &[T; NUM_VARS]) -> T {
        let mut result = self.value;
        for (((&gi, &xi), &pi), hrow) in self
            .gradient
            .iter()
            .zip(x)
            .zip(&self.point)
            .zip(&self.hessian)
        {
            let di = xi - pi;
            result += gi * di;
            for ((&hij, &xj), &pj) in hrow.iter().zip(x).zip(&self.point) {
                result += T::HALF * hij * di * (xj - pj);
            }
        }

        result
    }

    /// The base point the approximation is centered on.
    pub fn point(&self) -> &[T; NUM_VARS] {
        &self.point
    }

    /// The gradient at the base point.
    pub fn gradient(&self) -> &[T; NUM_VARS] {
        &self.gradient
    }

    /// The Hessian matrix at the base point.
    pub fn hessian(&self) -> &[[T; NUM_VARS]; NUM_VARS] {
        &self.hessian
    }

    /// Computes goodness-of-fit metrics against `original_function` over `points`.
    ///
    /// `r_squared` is `NaN` when the truth is constant over `points`;
    /// `adjusted_r_squared` is `NaN` when there are too few points.
    pub fn get_prediction_metrics<O: Fn(&[T; NUM_VARS]) -> T, const NUM_POINTS: usize>(
        &self,
        points: &[[T; NUM_VARS]; NUM_POINTS],
        original_function: &O,
    ) -> QuadraticApproximationPredictionMetrics<T> {
        // p = N gradient terms + N(N+1)/2 distinct (symmetric) Hessian terms
        let num_predictors = NUM_VARS + NUM_VARS * (NUM_VARS + 1) / 2;

        let (mae, mse, rmse, r_squared, adjusted_r_squared) = crate::approximation::compute_metrics(
            |x| self.predict(x),
            points,
            original_function,
            num_predictors,
        );

            mae += predicted_y - original_function(point);
            mse += function_approximations::static_powi(predicted_y - original_function(point), 2);
        }

        mae /= NUM_POINTS as f64;
        mse /= NUM_POINTS as f64;

        let rmse = function_approximations::sqrt_approx(mse).abs();

        let mut r2_numerator = 0.0;
        let mut r2_denominator = 0.0;

        for point in points.iter().take(NUM_POINTS) {
            let predicted_y = self.get_prediction_value(point);

            r2_numerator +=
                function_approximations::static_powi(predicted_y - original_function(point), 2);
            r2_denominator = r2_numerator
                + function_approximations::static_powi(mae - original_function(point), 2);
        }

        let r2 = 1.0 - (r2_numerator / r2_denominator);

        let r2_adj = 1.0 - (1.0 - r2) * (NUM_POINTS as f64) / ((NUM_POINTS as f64) - 2.0);

        QuadraticApproximationPredictionMetrics {
            mean_absolute_error: mae.abs(),
            mean_squared_error: mse.abs(),
            root_mean_squared_error: rmse,
            r_squared: r2.abs(),
            adjusted_r_squared: r2_adj.abs(),
        }
    }
}

/// Builds a [`QuadraticApproximation`] of a function, using any derivator that implements
/// [`DerivatorMultiVariable`].
pub struct QuadraticApproximator<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable + Default> Default for QuadraticApproximator<D> {
    fn default() -> Self {
        QuadraticApproximator {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> QuadraticApproximator<D> {
    /// Builds an approximator from an explicit derivator.
    pub fn from_derivator(derivator: D) -> Self {
        QuadraticApproximator { derivator }
    }

    /// Builds a quadratic (second-order Taylor) approximation of `function` about `point`.
    ///
    /// # Errors
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::approximation::quadratic_approximation::QuadraticApproximator;
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    ///
    /// // e^(x/2) + sin(y) + 2z
    /// let function_to_approximate = | args: &[f64; 3] | -> f64
    /// {
    ///     return f64::exp(args[0]/2.0) + f64::sin(args[1]) + 2.0*args[2];
    /// };
    ///
    /// let point = [0.0, 1.57, 10.0]; //the point we want to approximate around
    /// let approximator = QuadraticApproximator::<FiniteDifferenceMulti>::default();
    /// let result = approximator.get(&function_to_approximate, &point).unwrap();
    ///
    /// to see how the [QuadraticApproximationResult::quadratic_coefficients] matrix should be used, refer to [`QuadraticApproximationResult::get_prediction_metrics()`]
    /// or refer to its tests.
    ///
    pub fn get<const NUM_VARS: usize>(
        &self,
        function: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        point: &[f64; NUM_VARS],
    ) -> Result<QuadraticApproximationResult<NUM_VARS>, &'static str> {
        let mut intercept_ = function(point);

        let mut linear_coeffs_ = [0.0; NUM_VARS];

        let hessian_matrix = Hessian::from_derivator(self.derivator).get(function, point)?;

        for iter in 0..NUM_VARS {
            linear_coeffs_[iter] = self.derivator.get(1, function, &[iter], point)?;
            intercept_ -= self.derivator.get(1, function, &[iter], point)? * point[iter];
        }

        let mut quad_coeff = [[0.0; NUM_VARS]; NUM_VARS];

        for row in 0..NUM_VARS {
            for col in row..NUM_VARS {
                quad_coeff[row][col] = hessian_matrix[row][col];

                intercept_ += hessian_matrix[row][col] * point[row] * point[row];

                if row == col {
                    linear_coeffs_[row] -= 2.0 * hessian_matrix[row][col] * point[row]
                } else {
                    linear_coeffs_[row] -= hessian_matrix[row][col] * point[col];
                    linear_coeffs_[col] -= hessian_matrix[row][col] * point[row];
                }
            }
        }

        Ok(QuadraticApproximationResult {
            intercept: intercept_,
            linear_coefficients: linear_coeffs_,
            quadratic_coefficients: quad_coeff,
        })
    }
}

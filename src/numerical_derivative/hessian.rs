use crate::numeric::Numeric;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::utils::error_codes::CalcError;

/// @brief Computes the Hessian matrix for a given function. It can handle single
/// and multivariable equations of any complexity or size.
pub struct Hessian<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable> Default for Hessian<D> {
    /// @brief Default constructor, optimal for most generic cases.
    fn default() -> Self {
        Hessian {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> Hessian<D> {
    /// @brief Custom constructor, optimal for fine-tuning.
    ///
    /// You can create a custom multivariable derivator from this crate
    /// or supply your own by implementing the base traits yourself.
    ///
    /// @param derivator A custom instance implementing the `DerivatorMultiVariable` trait.
    ///
    /// @return A new `Hessian` instance using the provided derivator.
    pub fn from_derivator(derivator: D) -> Self {
        Hessian { derivator }
    }

    /// @brief Returns the Hessian matrix for a given function. It can handle
    /// multivariable functions of any order or complexity.
    ///
    /// The 2-D matrix returned has the structure:
    ///
    /// [[d²f/dx₁², d²f/dx₁dx₂, ... , d²f/dx₁dxₙ],
    ///  [...                                 ...],
    ///  [d²f/dxₙdx₁, d²f/dxₙdx₂, ... , d²f/dxₙ²]]
    ///
    /// where `N` is the total number of variables.
    ///
    /// @tparam NUM_VARS The number of variables in the function.
    /// @param function The target multivariable function.
    /// @param vector_of_points The point at which the Hessian matrix should be evaluated.
    ///
    /// @return Result containing the symmetric Hessian matrix as `[[f64; NUM_VARS]; NUM_VARS]`,
    /// or an error string if the computation fails.
    ///
    /// @note
    /// Possible error codes include:
    /// - `NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO`: If the derivative step size is zero.
    ///
    /// @example
    /// ```
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    /// use multicalc::numerical_derivative::hessian::Hessian;
    ///
    /// let my_func = |args: &[f64; 2]| -> f64 {
    ///     args[1] * args[0].sin() + 2.0 * args[0] * args[1].exp()
    /// };
    ///
    /// let points = [1.0, 2.0]; // The point around which we want the Hessian matrix.
    /// let hessian = Hessian::<MultiVariableSolver>::default();
    ///
    /// let hessian = Hessian::<FiniteDifferenceMulti>::default();
    /// let result = hessian.get(&my_func, &[1.0, 2.0]).unwrap();
    /// assert!(f64::abs(result[0][0] - (-2.0 * f64::sin(1.0))) < 1e-5);
    /// ```
    pub fn get<F: Fn(&[D::Scalar; NUM_VARS]) -> D::Scalar, const NUM_VARS: usize>(
        &self,
        function: &F,
        vector_of_points: &[D::Scalar; NUM_VARS],
    ) -> Result<[[D::Scalar; NUM_VARS]; NUM_VARS], CalcError> {
        let mut result = [[<D::Scalar as Numeric>::NAN; NUM_VARS]; NUM_VARS];

        // explicit indices are needed for the symmetric mirror write `result[col][row]`
        #[allow(clippy::needless_range_loop)]
        for row_index in 0..NUM_VARS {
            for col_index in 0..NUM_VARS {
                if result[row_index][col_index].is_nan() {
                    result[row_index][col_index] = self.derivator.get_double_partial(
                        function,
                        &[row_index, col_index],
                        vector_of_points,
                    )?;

                    result[col_index][row_index] = result[row_index][col_index];
                    //exploit the fact that a hessian is a symmetric matrix
                }
            }
        }

        return Ok(result);
    }
}

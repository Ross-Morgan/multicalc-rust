use crate::numerical_derivative::derivator::DerivatorMultiVariable;

use num_complex::ComplexFloat;

///computes the hessian matrix for a given function
/// Can handle single and multivariable equations of any complexity or size
pub struct Hessian<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable> Default for Hessian<D> {
    ///the default constructor, optimal for most generic cases
    fn default() -> Self {
        return Hessian {
            derivator: D::default(),
        };
    }
}

impl<D: DerivatorMultiVariable> Hessian<D> {
    ///custom constructor, optimal for fine tuning
    /// You can create a custom multivariable derivator from this crate
    /// or supply your own by implementing the base traits yourself
    pub fn from_derivator(derivator: D) -> Self {
        return Hessian { derivator };
    }

    /// Returns the hessian matrix for a given function
    /// Can handle multivariable functions of any order or complexity
    ///
    /// The result is the symmetric matrix of second partial derivatives, so entry `[i][j]`
    /// is `d²(function)/d(variable i) d(variable j)`. Only the upper triangle and diagonal
    /// are evaluated; the rest is mirrored, relying on the symmetry of the Hessian.
    ///
    /// # Arguments
    /// * `function` - the scalar function to differentiate.
    /// * `vector_of_points` - the point at which the derivatives are taken.
    ///
    /// # Errors
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    /// use multicalc::numerical_derivative::hessian::Hessian;
    ///
    /// // f(x, y) = y*sin(x) + 2*x*e^y
    /// let my_func = |args: &[f64; 2]| args[1] * args[0].sin() + 2.0 * args[0] * args[1].exp();
    ///
    /// let hessian = Hessian::<FiniteDifferenceMulti>::default();
    /// let result = hessian.get(&my_func, &[1.0, 2.0]).unwrap();
    /// assert!(f64::abs(result[0][0] - (-2.0 * f64::sin(1.0))) < 1e-5);
    /// ```
    ///
    pub fn get<T: ComplexFloat, const NUM_VARS: usize>(
        &self,
        function: &dyn Fn(&[T; NUM_VARS]) -> T,
        vector_of_points: &[T; NUM_VARS],
    ) -> Result<[[T; NUM_VARS]; NUM_VARS], &'static str> {
        let mut result = [[T::from(f64::NAN).unwrap(); NUM_VARS]; NUM_VARS];

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

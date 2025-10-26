use crate::numerical_derivative::finite_difference::MultiVariableSolver;
use const_poly::Polynomial;

///computes the hessian matrix for a given function
/// Can handle single and multivariable equations of any complexity or size
pub struct Hessian {
    derivator: MultiVariableSolver,
}

impl Default for Hessian {
    ///the default constructor, optimal for most generic cases
    fn default() -> Self {
        return Hessian {
            derivator: MultiVariableSolver::default(),
        };
    }
}

impl Hessian {
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
    pub const fn get<const NUM_VARS: usize>(
        &self,
        function: &Polynomial<NUM_VARS>,
        vector_of_points: &[f64; NUM_VARS],
    ) -> Result<[[f64; NUM_VARS]; NUM_VARS], &'static str> {
        let mut result = [[0.0; NUM_VARS]; NUM_VARS];

        let mut row_index = 0;

        while row_index < NUM_VARS {
            let mut col_index = 0;
            while col_index < NUM_VARS {
                // compute only upper triangle (symmetric Hessian)
                if col_index >= row_index {
                    let res = self.derivator.get_double_partial(
                        function,
                        &[row_index, col_index],
                        vector_of_points,
                    );

                    match res {
                        Ok(value) => {
                            result[row_index][col_index] = value;
                            result[col_index][row_index] = value;
                        }
                        Err(e) => return Err(e),
                    }
                }

                col_index += 1;
            }
            row_index += 1;
        }

        Ok(result)
    }
}

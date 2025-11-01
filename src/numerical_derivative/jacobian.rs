use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::utils::error_codes::*;

#[cfg(feature = "heap")]
use std::{boxed::Box, vec::Vec};

pub struct Jacobian<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable> Default for Jacobian<D> {
    fn default() -> Self {
        Jacobian {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> Jacobian<D> {
    ///custom constructor, optimal for fine tuning
    /// You can create a custom multivariable derivator from this crate
    /// or supply your own by implementing the base traits yourself
    pub fn from_derivator(derivator: D) -> Self {
        Jacobian { derivator }
    }

    /// Returns the jacobian matrix for a given vector of functions
    /// Can handle multivariable functions of any order or complexity
    ///
    /// The result has one row per function and one column per variable, so entry `[m][n]`
    /// is `d(function m)/d(variable n)`.
    ///
    /// # Arguments
    /// * `function_matrix` - the functions whose partial derivatives form the rows.
    /// * `vector_of_points` - the point at which the derivatives are taken.
    ///
    /// # Errors
    /// [`CalcError::EmptyFunctionSet`] if `function_matrix` is empty, or
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    /// use multicalc::numerical_derivative::jacobian::Jacobian;
    ///
    /// // the vector function (x*y*z, x^2 + y^2)
    /// let my_func1 = |args: &[f64; 3]| args[0] * args[1] * args[2];
    /// let my_func2 = |args: &[f64; 3]| args[0] * args[0] + args[1] * args[1];
    /// let function_matrix: [&dyn Fn(&[f64; 3]) -> f64; 2] = [&my_func1, &my_func2];
    ///
    /// let jacobian = Jacobian::<FiniteDifferenceMulti>::default();
    /// let result = jacobian.get(&function_matrix, &[1.0, 2.0, 3.0]).unwrap();
    /// // result is [[6, 3, 2], [2, 4, 0]]
    /// assert!(f64::abs(result[0][0] - 6.0) < 1e-6);
    /// ```
    ///
    pub fn get<const NUM_FUNCS: usize, const NUM_VARS: usize>(
        &self,
        function_matrix: &[&dyn Fn(&[f64; NUM_VARS]) -> f64; NUM_FUNCS],
        vector_of_points: &[f64; NUM_VARS],
    ) -> Result<[[f64; NUM_VARS]; NUM_FUNCS], &'static str> {
        if function_matrix.is_empty() {
            return Err(VECTOR_OF_FUNCTIONS_CANNOT_BE_EMPTY);
        }

        let mut result = [[0.0; NUM_VARS]; NUM_FUNCS];

        for row_index in 0..NUM_FUNCS {
            for col_index in 0..NUM_VARS {
                result[row_index][col_index] = self.derivator.get_single_partial(
                    &function_matrix[row_index],
                    col_index,
                    vector_of_points,
                )?;
            }
        }

        Ok(result)
    }

    /// Same as [`Jacobian::get`] but returns a heap-allocated `Vec<Vec<_>>`, which avoids a
    /// stack overflow on large inputs. Requires the `alloc` feature (off by default).
    ///
    /// The result has one row per function and one column per variable, so entry `[m][n]`
    /// is `d(function m)/d(variable n)`.
    ///
    /// # Arguments
    /// * `function_matrix` - the functions whose partial derivatives form the rows.
    /// * `vector_of_points` - the point at which the derivatives are taken.
    ///
    /// NOTE: Returns a Result<T, &'static str>
    /// Possible &'static str are:
    /// VectorOfFunctionsCannotBeEmpty -> if function_matrix argument is an empty array
    /// NumberOfStepsCannotBeZero -> if the derivative step size is zero
    #[cfg(feature = "heap")]
    pub fn get_on_heap<const NUM_VARS: usize>(
        &self,
        function_matrix: &Vec<Box<dyn Fn(&[f64; NUM_VARS]) -> f64>>,
        vector_of_points: &[f64; NUM_VARS],
    ) -> Result<Vec<Vec<T>>, &'static str> {
        if function_matrix.is_empty() {
            return Err(CalcError::EmptyFunctionSet);
        }

        let mut result: Vec<Vec<D::Scalar>> = Vec::new();

        let mut result: Vec<Vec<f64>> = Vec::new();

        for row_index in 0..num_funcs {
            let mut cur_row: Vec<f64> = Vec::new();
            for col_index in 0..NUM_VARS {
                cur_row.push(self.derivator.get_single_partial(
                    func,
                    col_index,
                    vector_of_points,
                )?);
            }
            result.push(cur_row);
        }

        Ok(result)
    }
}

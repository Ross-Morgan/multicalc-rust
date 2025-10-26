use crate::numerical_integration::iterative_integration::DEFAULT_TOTAL_ITERATIONS;
use crate::utils::error_codes::*;
use const_poly::Polynomial;

/// Builds the curve position [transformations[0](t), ..., transformations[N-1](t)].
fn curve_point<T: Numeric, const N: usize>(transformations: &[&dyn Fn(T) -> T; N], t: T) -> [T; N] {
    let mut point = [T::ZERO; N];
    for i in 0..N {
        point[i] = transformations[i](t);
    }
    point
}

/// Trapezoidal integration of the `idx`-th field component along the parametrized curve.
/// Generic over the dimension `N`, so the 2D and 3D paths share one body.
fn get_partial<T: Numeric, const N: usize>(
    vector_field: &[&dyn Fn(&[T; N]) -> T; N],
    transformations: &[&dyn Fn(T) -> T; N],
    integration_limit: &[T; 2],
    total_iterations: u64,
    idx: usize,
) -> Result<T, CalcError> {
    if total_iterations == 0 {
        return Err(CalcError::IterationsZero);
    }
    // rejects NaN, equal, and reversed limits (partial_cmp is None for NaN)
    if !matches!(
        integration_limit[0].partial_cmp(&integration_limit[1]),
        Some(core::cmp::Ordering::Less)
    ) {
        return Err(CalcError::IntegrationLimitsIllDefined);
    }

    let delta = (integration_limit[1] - integration_limit[0]) / T::from_u64(total_iterations);
    let mut t = integration_limit[0];
    let mut ans = T::ZERO;

    //use the trapezoidal rule for line integrals, caching the shared endpoint so each
    //curve point and field value is evaluated once per node rather than twice
    //https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter25/section04.html
    let mut left = curve_point(transformations, t);
    let mut left_value = vector_field[idx](&left);

    for _ in 0..total_iterations {
        let right = curve_point(transformations, t + delta);
        let right_value = vector_field[idx](&right);

        ans += (right[idx] - left[idx]) * (left_value + right_value) / T::TWO;

        t += delta;
        left = right;
        left_value = right_value;
    }

    Ok(ans)
}

/// Computes the line integral of a 2D vector field along a parametrized curve.
///
/// NOTE: Returns a Result<f64, &'static str>
/// Possible &'static str are:
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
///
/// # Arguments
/// * `vector_field` - the two field components, each taking the curve position `[x, y]`.
/// * `transformations` - the two transforms mapping `t` to `x` and to `y`.
/// * `integration_limit` - the `[lower, upper]` range of the parameter `t`.
///
/// # Errors
/// [`CalcError::IntegrationLimitsIllDefined`] if the lower limit is not strictly less than the
/// upper limit.
///
/// # Examples
/// ```
/// use multicalc::vector_field::line_integral;
///
/// // the field (y, -x) along the unit circle (cos t, sin t), for t in [0, 2*pi]
/// let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] =
///     [&(|args: &[f64; 2]| args[1]), &(|args: &[f64; 2]| -args[0])];
/// let transformation_matrix: [&dyn Fn(f64) -> f64; 2] =
///     [&(|t: f64| t.cos()), &(|t: f64| t.sin())];
///
/// let val = line_integral::get_2d(&vector_field_matrix, &transformation_matrix, &[0.0, 6.28]).unwrap();
/// // the line integral is -2*pi
/// assert!(f64::abs(val + 6.28) < 0.01);
/// ```
pub fn get_2d(
    vector_field: &[&Polynomial<2>; 2],
    transformations: &[&Polynomial<1>; 2],
    integration_limit: &[f64; 2],
) -> Result<f64, &'static str> {
    return get_2d_custom(
        vector_field,
        transformations,
        integration_limit,
        DEFAULT_TOTAL_ITERATIONS,
    )
}

///same as [get_2d()] but with the option to change the total iterations used, reserved for more advanced user
/// The argument 'n' denotes the number of steps to be used. However, for [`mode::IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// NOTE: Returns a Result<f64, &'static str>
/// Possible &'static str are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_2d_custom(
    vector_field: &[&Polynomial<2>; 2],
    transformations: &[&Polynomial<1>; 2],
    integration_limit: &[f64; 2],
    total_iterations: u64,
) -> Result<f64, &'static str> {
    return Ok(get_partial_2d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        0,
    )? + get_partial_2d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        1,
    )?)
}

/// NOTE: Returns a Result<f64, &'static str>
/// Possible &'static str are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_partial_2d(
    vector_field: &[&Polynomial<2>; 2],
    transformations: &[&Polynomial<1>; 2],
    integration_limit: &[f64; 2],
    max_iterations: u64,
    idx: usize,
) -> Result<f64, &'static str> {
    if max_iterations == 0 {
        return Err(INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS);
    }
    if integration_limit[0].abs() >= integration_limit[1].abs() {
        return Err(INTEGRATION_LIMITS_ILL_DEFINED);
    }

    let mut ans = 0.0;

    let mut cur_point = integration_limit[0];

    let delta = (integration_limit[1] - integration_limit[0]) / (max_iterations as f64);

    //use the trapezoidal rule for line integrals
    //https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter25/section04.html
    for _ in 0..max_iterations {
        let coords = get_transformed_coordinates_2d(transformations, cur_point, delta);

        ans = ans
            + (coords[idx + 2] - coords[idx])
                * (vector_field[idx].evaluate(&[coords[2], coords[3]])
                    + vector_field[idx].evaluate(&[coords[0], coords[1]]))
                / (2.0);

        cur_point = cur_point + delta;
    }

    return Ok(ans);
}

///same as [`get_2d`] but for parametrized curves in a 3D vector field
/// NOTE: Returns a Result<f64, &'static str>
/// Possible &'static str are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_3d(
    vector_field: &[&Polynomial<3>; 3],
    transformations: &[&Polynomial<1>; 3],
    integration_limit: &[f64; 2],
) -> Result<f64, &'static str> {
    return get_3d_custom(
        vector_field,
        transformations,
        integration_limit,
        DEFAULT_TOTAL_ITERATIONS,
    )
}

///same as [get_3d()] but with the option to change the total iterations used, reserved for more advanced user
/// The argument 'n' denotes the number of steps to be used. However, for [`mode::IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// NOTE: Returns a Result<f64, &'static str>
/// Possible &'static str are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_3d_custom(
    vector_field: &[&Polynomial<3>; 3],
    transformations: &[&Polynomial<1>; 3],
    integration_limit: &[f64; 2],
    total_iterations: u64,
) -> Result<f64, &'static str> {
    return Ok(get_partial_3d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        0,
    )? + get_partial_3d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        1,
    )? + get_partial_3d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        2,
    )?)
}

/// NOTE: Returns a Result<f64, &'static str>
/// Possible &'static str are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_partial_3d(
    vector_field: &[&Polynomial<3>; 3],
    transformations: &[&Polynomial<1>; 3],
    integration_limit: &[f64; 2],
    steps: u64,
    idx: usize,
) -> Result<f64, &'static str> {
    if steps == 0 {
        return Err(INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS);
    }
    if integration_limit[0].abs() >= integration_limit[1].abs() {
        return Err(INTEGRATION_LIMITS_ILL_DEFINED);
    }

    let mut ans = 0.0;

    let mut cur_point = integration_limit[0];

    let delta = (integration_limit[1] - integration_limit[0]) / (steps as f64);

    //use the trapezoidal rule for line integrals
    //https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter25/section04.html
    for _ in 0..steps {
        let coords = get_transformed_coordinates_3d(transformations, cur_point, delta);

        ans = ans
            + (coords[idx + 3] - coords[idx])
                * (vector_field[idx].evaluate(&[coords[3], coords[4], coords[5]])
                    + vector_field[idx].evaluate(&[coords[0], coords[1], coords[2]]))
                / (2.0);

        cur_point = cur_point + delta;
    }

    return Ok(ans);
}

fn get_transformed_coordinates_2d(
    transformations: &[&Polynomial<1>; 2],
    cur_point: f64,
    delta: f64,
) -> [f64; 4] {
    let mut ans = [0.0; 4];

    ans[0] = transformations[0].evaluate_scalar(cur_point); //x at t
    ans[1] = transformations[1].evaluate_scalar(cur_point); //y at t

    ans[2] = transformations[0].evaluate_scalar(cur_point + delta); //x at t + delta
    ans[3] = transformations[1].evaluate_scalar(cur_point + delta); //y at t + delta

    return ans;
}

fn get_transformed_coordinates_3d(
    transformations: &[&Polynomial<1>; 3],
    cur_point: f64,
    delta: f64,
) -> [f64; 6] {
    let mut ans = [0.0; 6];

    ans[0] = transformations[0].evaluate_scalar(cur_point); //x at t
    ans[1] = transformations[1].evaluate_scalar(cur_point); //y at t
    ans[2] = transformations[1].evaluate_scalar(cur_point); //z at t

    ans[3] = transformations[0].evaluate_scalar(cur_point + delta); //x at t + delta
    ans[4] = transformations[1].evaluate_scalar(cur_point + delta); //y at t + delta
    ans[5] = transformations[1].evaluate_scalar(cur_point + delta); //z at t + delta

    return ans;
}

// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Hand-rolled numerical primitives — no external math dependencies.
//!
//! The crate is zero-dependency, so every optimiser and solver is implemented
//! from its primary source. All routines are pure functions, deterministic
//! (same input produces bit-identical output), and `std`-only.
//!
//! # Contents
//!
//! - [`nelder_mead`] — downhill-simplex minimisation (Nelder & Mead 1965).
//!   Gradient-free; used for the low-dimensional outer problems
//!   (2-D quasi-explicit, 2-3-D SSVI).
//! - [`brent_root`] — bracketed root-finding (Brent 1973). Combines bisection,
//!   secant, and inverse quadratic interpolation; guaranteed convergence on
//!   any sign-changing bracket.
//! - [`solve_spd_3`] / [`solve_spd`] — symmetric positive-definite linear
//!   solves by Cholesky decomposition; the exact inner solver for weighted
//!   normal equations.
//! - [`levenberg_marquardt`] — damped Gauss-Newton for nonlinear
//!   least-squares (Levenberg 1944; Marquardt 1963), with the gain-ratio
//!   damping update.
//!
//! # References
//!
//! - Nelder, J. A. & Mead, R., "A simplex method for function minimization",
//!   *The Computer Journal* 7(4):308-313 (1965).
//! - Brent, R. P., *Algorithms for Minimization Without Derivatives*,
//!   Prentice-Hall (1973), Chapter 4.
//! - Levenberg, K., "A method for the solution of certain non-linear problems
//!   in least squares", *Quarterly of Applied Mathematics* 2(2):164-168 (1944).
//! - Marquardt, D. W., "An algorithm for least-squares estimation of nonlinear
//!   parameters", *Journal of SIAM* 11(2):431-441 (1963).

// ─── Nelder-Mead downhill simplex ────────────────────────────────────────────

/// Standard Nelder-Mead reflection coefficient `alpha`.
const NM_REFLECT: f64 = 1.0;
/// Standard Nelder-Mead expansion coefficient `gamma`.
const NM_EXPAND: f64 = 2.0;
/// Standard Nelder-Mead contraction coefficient `rho`.
const NM_CONTRACT: f64 = 0.5;
/// Standard Nelder-Mead shrink coefficient `sigma`.
const NM_SHRINK: f64 = 0.5;

/// Converts a `usize` index or count to `f64` losslessly.
///
/// Splits the value into a high and low `u32` half and recombines through
/// `f64::from`, both of which are exact conversions. The result is exact for
/// every `usize` below `2^53` (the `f64` mantissa width) — i.e. for every
/// grid index and count this crate produces — and avoids the precision-loss
/// `as`-cast lint entirely.
#[inline]
#[must_use]
pub fn index_to_f64(i: usize) -> f64 {
    let value = i as u64;
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

/// Outcome of a [`nelder_mead`] minimisation.
///
/// Carries the best point found, its objective value, the iteration count,
/// and whether the convergence test was met before the iteration cap.
#[derive(Debug, Clone, PartialEq)]
pub struct NelderMeadResult {
    /// The minimising point.
    pub x: Vec<f64>,
    /// The objective value at [`Self::x`].
    pub fx: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// `true` if the simplex satisfied the tolerance test before the cap.
    pub converged: bool,
}

/// Minimises a scalar objective by the Nelder-Mead downhill simplex method.
///
/// Builds an initial simplex of `n + 1` vertices around `start` (each axis
/// perturbed by a step proportional to `|x_j|`, or by an absolute step for a
/// zero coordinate) and applies reflection, expansion, contraction, and
/// shrink steps with the standard coefficients `(1, 2, 0.5, 0.5)`
/// (Nelder & Mead 1965). Termination is declared when the spread of objective
/// values across the simplex falls below `tol`, or after `max_iter`
/// iterations.
///
/// # Arguments
///
/// * `objective` — the function to minimise.
/// * `start` — the initial guess; its length sets the dimension `n`.
/// * `tol` — convergence tolerance on the objective spread.
/// * `max_iter` — iteration cap.
///
/// # Examples
///
/// ```
/// use regit_svi::math::nelder_mead;
///
/// // Minimise the 2-D sphere; the minimum is the origin.
/// let res = nelder_mead(|x| x[0] * x[0] + x[1] * x[1], &[1.0, 1.0], 1e-12, 500);
/// assert!(res.fx < 1e-10);
/// assert!(res.converged);
/// ```
#[must_use]
#[allow(clippy::too_many_lines)] // One cohesive simplex algorithm; splitting hurts readability.
pub fn nelder_mead<F>(objective: F, start: &[f64], tol: f64, max_iter: usize) -> NelderMeadResult
where
    F: Fn(&[f64]) -> f64,
{
    let n = start.len();
    if n == 0 {
        return NelderMeadResult {
            x: Vec::new(),
            fx: objective(&[]),
            iterations: 0,
            converged: true,
        };
    }

    // Build the initial simplex: start, plus one perturbed vertex per axis.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(start.to_vec());
    for j in 0..n {
        let mut v = start.to_vec();
        let step = if v[j].abs() > 1e-8 {
            0.05 * v[j].abs()
        } else {
            0.000_25
        };
        v[j] += step;
        simplex.push(v);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| objective(v)).collect();
    let mut order: Vec<usize> = (0..=n).collect();

    let mut iterations = 0;
    let mut converged = false;

    while iterations < max_iter {
        iterations += 1;

        // Sort vertices by objective value (ascending).
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        // Convergence: both the objective spread and the geometric diameter
        // of the simplex must fall below tolerance. Checking only the
        // objective spread would stop prematurely when two vertices straddle
        // the minimum at equal heights.
        let spread = (fvals[worst] - fvals[best]).abs();
        let mut diameter = 0.0_f64;
        for v in &simplex {
            let mut dist = 0.0;
            for j in 0..n {
                let d = v[j] - simplex[best][j];
                dist += d * d;
            }
            diameter = diameter.max(dist.sqrt());
        }
        if spread <= tol && diameter <= tol.sqrt().max(tol) {
            converged = true;
            break;
        }

        // Centroid of all vertices except the worst.
        let mut centroid = vec![0.0; n];
        for (idx, &v) in order.iter().enumerate() {
            if idx == n {
                continue;
            }
            for j in 0..n {
                centroid[j] += simplex[v][j];
            }
        }
        let inv_n = 1.0 / index_to_f64(n);
        for c in &mut centroid {
            *c *= inv_n;
        }

        // Reflection.
        let reflected = axpy(&centroid, NM_REFLECT, &centroid, &simplex[worst]);
        let f_reflected = objective(&reflected);

        if f_reflected < fvals[best] {
            // Expansion.
            let expanded = axpy(&centroid, NM_EXPAND, &reflected, &centroid);
            let f_expanded = objective(&expanded);
            if f_expanded < f_reflected {
                simplex[worst] = expanded;
                fvals[worst] = f_expanded;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = f_reflected;
            }
        } else if f_reflected < fvals[second_worst] {
            // Accept the reflected point.
            simplex[worst] = reflected;
            fvals[worst] = f_reflected;
        } else {
            // Contraction.
            let (contracted, f_contracted) = if f_reflected < fvals[worst] {
                // Outside contraction.
                let c = axpy(&centroid, NM_CONTRACT, &reflected, &centroid);
                let fc = objective(&c);
                (c, fc)
            } else {
                // Inside contraction.
                let c = axpy(&centroid, NM_CONTRACT, &simplex[worst], &centroid);
                let fc = objective(&c);
                (c, fc)
            };

            if f_contracted < f_reflected.min(fvals[worst]) {
                simplex[worst] = contracted;
                fvals[worst] = f_contracted;
            } else {
                // Shrink towards the best vertex.
                let best_pt = simplex[best].clone();
                for &v in &order[1..] {
                    for j in 0..n {
                        simplex[v][j] = best_pt[j] + NM_SHRINK * (simplex[v][j] - best_pt[j]);
                    }
                    fvals[v] = objective(&simplex[v]);
                }
            }
        }
    }

    // Final ordering.
    order.sort_by(|&a, &b| {
        fvals[a]
            .partial_cmp(&fvals[b])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let best = order[0];

    NelderMeadResult {
        x: simplex[best].clone(),
        fx: fvals[best],
        iterations,
        converged,
    }
}

/// Computes `centroid + coeff * (point - reference)`, the affine step shared
/// by the reflection, expansion, and contraction operations.
fn axpy(centroid: &[f64], coeff: f64, point: &[f64], reference: &[f64]) -> Vec<f64> {
    centroid
        .iter()
        .zip(point.iter())
        .zip(reference.iter())
        .map(|((&c, &p), &r)| c + coeff * (p - r))
        .collect()
}

// ─── Brent root-finder ───────────────────────────────────────────────────────

/// Finds a root of `f` inside the bracket `[a, b]` by Brent's method.
///
/// `f(a)` and `f(b)` must have opposite signs. The method combines bisection,
/// the secant rule, and inverse quadratic interpolation: it keeps the
/// guaranteed convergence of bisection while accelerating to super-linear
/// order near a simple root (Brent 1973). Returns the root once the bracket
/// width drops below `tol` or `f` evaluates exactly to zero.
///
/// # Errors
///
/// Returns `None` if `f(a)` and `f(b)` do not bracket a sign change.
///
/// # Examples
///
/// ```
/// use regit_svi::math::brent_root;
///
/// // Root of x^2 - 2 on [0, 2] is sqrt(2).
/// let root = brent_root(|x| x * x - 2.0, 0.0, 2.0, 1e-12, 100).unwrap();
/// assert!((root - 2.0_f64.sqrt()).abs() < 1e-10);
/// ```
// Brent's method names its bracket and history points a, b, c, d, s as in
// the primary source (Brent 1973, Chapter 4); the single-char lint is noise
// for this canonical algorithm.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn brent_root<F>(f: F, a: f64, b: f64, tol: f64, max_iter: usize) -> Option<f64>
where
    F: Fn(f64) -> f64,
{
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);

    if fa == 0.0 {
        return Some(a);
    }
    if fb == 0.0 {
        return Some(b);
    }
    if fa * fb > 0.0 {
        return None;
    }

    // Ensure |f(b)| <= |f(a)| so b is the better current estimate.
    if fa.abs() < fb.abs() {
        core::mem::swap(&mut a, &mut b);
        core::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = a;
    let mut mflag = true;

    for _ in 0..max_iter {
        if (b - a).abs() <= tol || fb == 0.0 {
            return Some(b);
        }

        let mut s = if (fa - fc).abs() > f64::EPSILON && (fb - fc).abs() > f64::EPSILON {
            // Inverse quadratic interpolation.
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // Secant rule.
            b - fb * (b - a) / (fb - fa)
        };

        // Decide whether to keep the interpolant or fall back to bisection.
        let lo = (3.0 * a + b) / 4.0;
        let bound_lo = lo.min(b);
        let bound_hi = lo.max(b);
        let use_bisection = !(bound_lo..=bound_hi).contains(&s)
            || (mflag && (s - b).abs() >= (b - c).abs() / 2.0)
            || (!mflag && (s - b).abs() >= (c - d).abs() / 2.0)
            || (mflag && (b - c).abs() < tol)
            || (!mflag && (c - d).abs() < tol);

        if use_bisection {
            s = f64::midpoint(a, b);
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if fa.abs() < fb.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut fa, &mut fb);
        }
    }

    Some(b)
}

// ─── Symmetric positive-definite linear solves (Cholesky) ────────────────────

/// Solves the `3x3` symmetric positive-definite system `A x = rhs` by
/// Cholesky decomposition.
///
/// `a` holds the six independent entries of the symmetric matrix in the
/// order `[a00, a01, a02, a11, a12, a22]`. The system is exact (no
/// iteration): `A = L Lᵀ` is formed and the two triangular solves are
/// back-substituted.
///
/// # Errors
///
/// Returns `None` if `A` is not positive definite (a non-positive pivot is
/// encountered), which the caller treats as a singular normal-equations
/// system.
///
/// # Examples
///
/// ```
/// use regit_svi::math::solve_spd_3;
///
/// // Identity system: x = rhs.
/// let x = solve_spd_3(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], &[2.0, 3.0, 5.0]).unwrap();
/// assert!((x[0] - 2.0).abs() < 1e-15);
/// assert!((x[2] - 5.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn solve_spd_3(a: &[f64; 6], rhs: &[f64; 3]) -> Option<[f64; 3]> {
    let mat = [[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]];
    let sol = solve_spd(&mat.iter().flatten().copied().collect::<Vec<_>>(), rhs, 3)?;
    Some([sol[0], sol[1], sol[2]])
}

/// Solves the `n x n` symmetric positive-definite system `A x = rhs` by
/// Cholesky decomposition `A = L Lᵀ`.
///
/// `a` is the matrix in row-major order (`n * n` entries); only the lower
/// triangle is read. The factorisation and the two triangular solves are
/// exact up to floating-point round-off.
///
/// # Errors
///
/// Returns `None` if `A` is not positive definite or the dimensions are
/// inconsistent.
///
/// # Examples
///
/// ```
/// use regit_svi::math::solve_spd;
///
/// // 2x2 SPD system [[4, 2], [2, 3]] x = [10, 8] -> x = [1.4, 1.733...].
/// let x = solve_spd(&[4.0, 2.0, 2.0, 3.0], &[10.0, 8.0], 2).unwrap();
/// assert!((4.0 * x[0] + 2.0 * x[1] - 10.0).abs() < 1e-12);
/// assert!((2.0 * x[0] + 3.0 * x[1] - 8.0).abs() < 1e-12);
/// ```
// The Cholesky factor L and the solve vectors x, y carry the standard
// single-letter names of matrix-computation texts (Golub & Van Loan).
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn solve_spd(a: &[f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    if a.len() != n * n || rhs.len() != n {
        return None;
    }

    // Cholesky: L is lower-triangular, A = L Lᵀ.
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                let pivot = l[j * n + j];
                if pivot == 0.0 {
                    return None;
                }
                l[i * n + j] = sum / pivot;
            }
        }
    }

    // Forward solve L y = rhs.
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }

    // Back solve Lᵀ x = y.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }

    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

// ─── Levenberg-Marquardt ─────────────────────────────────────────────────────

/// Outcome of a [`levenberg_marquardt`] minimisation.
#[derive(Debug, Clone, PartialEq)]
pub struct LevenbergMarquardtResult {
    /// The minimising parameter vector.
    pub params: Vec<f64>,
    /// The weighted sum of squared residuals at [`Self::params`].
    pub cost: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// `true` if a convergence test was met before the iteration cap.
    pub converged: bool,
}

/// Minimises a weighted nonlinear least-squares objective by the
/// Levenberg-Marquardt algorithm.
///
/// The objective is `F(p) = sum_i weight_i * residual_i(p)^2`. Each iteration
/// solves the damped normal equations
///
/// ```text
/// (JᵀW J + mu * diag(JᵀW J)) delta = -JᵀW r
/// ```
///
/// and adapts the damping `mu` by the gain ratio between the actual and the
/// predicted reduction in cost (Marquardt 1963): a successful step shrinks
/// `mu`, a rejected step grows it. `residual` must return, for a parameter
/// vector, the per-observation `(residual, weight, jacobian_row)` triples,
/// where `jacobian_row[j] = d(residual)/d(p_j)`.
///
/// # Arguments
///
/// * `residual` — closure mapping `p` to a vector of `(r_i, w_i, dr_i/dp)`.
/// * `start` — the initial parameter vector.
/// * `tol` — convergence tolerance on the gradient and step norms.
/// * `max_iter` — iteration cap.
///
/// # Examples
///
/// ```
/// use regit_svi::math::levenberg_marquardt;
///
/// // Fit y = a, the mean of three points; the minimiser is their average.
/// let data = [1.0_f64, 2.0, 3.0];
/// let res = levenberg_marquardt(
///     |p: &[f64]| {
///         data.iter()
///             .map(|&y| (p[0] - y, 1.0, vec![1.0]))
///             .collect::<Vec<_>>()
///     },
///     &[0.0],
///     1e-12,
///     100,
/// );
/// assert!((res.params[0] - 2.0).abs() < 1e-8);
/// ```
#[must_use]
pub fn levenberg_marquardt<F>(
    residual: F,
    start: &[f64],
    tol: f64,
    max_iter: usize,
) -> LevenbergMarquardtResult
where
    F: Fn(&[f64]) -> Vec<(f64, f64, Vec<f64>)>,
{
    let n = start.len();
    let mut params = start.to_vec();
    let mut mu = 1e-3;
    let nu_growth = 2.0;

    let cost_of =
        |obs: &[(f64, f64, Vec<f64>)]| -> f64 { obs.iter().map(|(r, w, _)| w * r * r).sum() };

    let mut obs = residual(&params);
    let mut cost = cost_of(&obs);
    let mut iterations = 0;
    let mut converged = false;

    while iterations < max_iter {
        iterations += 1;

        // Assemble JᵀW J (n x n) and -JᵀW r (n).
        let mut jtj = vec![0.0; n * n];
        let mut jtr = vec![0.0; n];
        for (r, w, jac) in &obs {
            for i in 0..n {
                jtr[i] -= w * jac[i] * r;
                for j in 0..n {
                    jtj[i * n + j] += w * jac[i] * jac[j];
                }
            }
        }

        // Gradient-norm convergence test.
        let grad_norm = jtr.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm <= tol || cost <= tol * tol {
            converged = true;
            break;
        }

        // Inner loop: grow mu until the damped step reduces the cost.
        let mut accepted = false;
        for _ in 0..30 {
            // Damped system: (JᵀWJ + mu * diag(JᵀWJ)).
            let mut damped = jtj.clone();
            for i in 0..n {
                damped[i * n + i] += mu * jtj[i * n + i].max(1e-12);
            }

            let Some(delta) = solve_spd(&damped, &jtr, n) else {
                mu *= nu_growth;
                continue;
            };

            let trial: Vec<f64> = params
                .iter()
                .zip(delta.iter())
                .map(|(&p, &d)| p + d)
                .collect();
            let trial_obs = residual(&trial);
            let trial_cost = cost_of(&trial_obs);

            // Predicted reduction from the local quadratic model.
            let predicted: f64 = (0..n)
                .map(|i| delta[i] * (mu * jtj[i * n + i].max(1e-12) * delta[i] + jtr[i]))
                .sum::<f64>()
                / 2.0;
            let actual = cost - trial_cost;
            let gain = if predicted > 0.0 {
                actual / predicted
            } else {
                -1.0
            };

            if gain > 0.0 && trial_cost < cost {
                // Accept the step; shrink the damping.
                let step_norm = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
                params = trial;
                obs = trial_obs;
                cost = trial_cost;
                let shrink = (1.0 - (2.0 * gain - 1.0).powi(3)).max(1.0 / 3.0);
                mu *= shrink;
                accepted = true;
                if step_norm <= tol {
                    converged = true;
                }
                break;
            }
            // Reject the step; grow the damping.
            mu *= nu_growth;
        }

        if !accepted {
            // No damped step could reduce the cost: the iteration has
            // reached a stationary point of the local model.
            converged = true;
            break;
        }
        if converged {
            break;
        }
    }

    LevenbergMarquardtResult {
        params,
        cost,
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Rosenbrock function: a standard non-convex optimiser test case
    /// with a single global minimum at `(1, 1)`.
    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    #[test]
    fn nelder_mead_minimises_sphere() {
        let res = nelder_mead(|x| x[0] * x[0] + x[1] * x[1], &[3.0, -2.0], 1e-14, 1000);
        assert!(res.fx < 1e-10, "fx = {}", res.fx);
        assert!(res.x[0].abs() < 1e-5);
        assert!(res.x[1].abs() < 1e-5);
        assert!(res.converged);
    }

    #[test]
    fn nelder_mead_minimises_rosenbrock() {
        let res = nelder_mead(rosenbrock, &[-1.2, 1.0], 1e-14, 5000);
        assert!((res.x[0] - 1.0).abs() < 1e-3, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-3, "x1 = {}", res.x[1]);
        assert!(res.fx < 1e-6);
    }

    #[test]
    fn nelder_mead_handles_one_dimension() {
        let res = nelder_mead(|x| (x[0] - 4.0) * (x[0] - 4.0), &[0.0], 1e-14, 1000);
        assert!((res.x[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn nelder_mead_empty_start() {
        let res = nelder_mead(|_| 7.0, &[], 1e-12, 10);
        assert!((res.fx - 7.0).abs() < 1e-15);
        assert!(res.converged);
    }

    #[test]
    fn brent_finds_sqrt_two() {
        let root = brent_root(|x| x * x - 2.0, 0.0, 2.0, 1e-14, 200).unwrap();
        assert!((root - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn brent_finds_cubic_root() {
        // x^3 - x - 2 has a single real root near 1.5213797.
        let root = brent_root(|x| x * x * x - x - 2.0, 1.0, 2.0, 1e-14, 200).unwrap();
        assert!((root - 1.521_379_706_804_567_6).abs() < 1e-10);
    }

    #[test]
    fn brent_endpoint_root() {
        let root = brent_root(|x| x - 3.0, 3.0, 5.0, 1e-12, 50).unwrap();
        assert!((root - 3.0).abs() < 1e-15);
    }

    #[test]
    fn brent_rejects_no_bracket() {
        assert!(brent_root(|x| x * x + 1.0, -1.0, 1.0, 1e-12, 50).is_none());
    }

    #[test]
    fn brent_finds_transcendental_root() {
        // cos(x) - x has its fixed point near 0.7390851.
        let root = brent_root(|x| x.cos() - x, 0.0, 1.0, 1e-14, 200).unwrap();
        assert!((root - 0.739_085_133_215_160_6).abs() < 1e-10);
    }

    #[test]
    fn solve_spd_3_identity() {
        let x = solve_spd_3(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], &[2.0, 3.0, 5.0]).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-15);
        assert!((x[1] - 3.0).abs() < 1e-15);
        assert!((x[2] - 5.0).abs() < 1e-15);
    }

    #[test]
    fn solve_spd_3_known_system() {
        // A = [[4,1,1],[1,3,0],[1,0,2]], rhs chosen so x = [1,2,3].
        let a = [4.0, 1.0, 1.0, 3.0, 0.0, 2.0];
        let rhs = [4.0 + 2.0 + 3.0, 1.0 + 6.0 + 0.0, 1.0 + 0.0 + 6.0];
        let x = solve_spd_3(&a, &rhs).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn solve_spd_rejects_non_spd() {
        // Indefinite matrix: a negative pivot is encountered.
        assert!(solve_spd(&[1.0, 2.0, 2.0, 1.0], &[1.0, 1.0], 2).is_none());
    }

    #[test]
    fn solve_spd_rejects_bad_dims() {
        assert!(solve_spd(&[1.0, 0.0, 0.0, 1.0], &[1.0], 2).is_none());
    }

    #[test]
    fn levenberg_marquardt_fits_mean() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let res = levenberg_marquardt(
            |p: &[f64]| {
                data.iter()
                    .map(|&y| (p[0] - y, 1.0, vec![1.0]))
                    .collect::<Vec<_>>()
            },
            &[0.0],
            1e-14,
            200,
        );
        assert!((res.params[0] - 2.5).abs() < 1e-8);
        assert!(res.converged);
    }

    #[test]
    fn levenberg_marquardt_fits_line() {
        // Fit y = a + b*x to exact data y = 2 + 3x.
        let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let res = levenberg_marquardt(
            |p: &[f64]| {
                xs.iter()
                    .map(|&x| {
                        let y = 2.0 + 3.0 * x;
                        (p[0] + p[1] * x - y, 1.0, vec![1.0, x])
                    })
                    .collect::<Vec<_>>()
            },
            &[0.0, 0.0],
            1e-14,
            300,
        );
        assert!((res.params[0] - 2.0).abs() < 1e-6, "a = {}", res.params[0]);
        assert!((res.params[1] - 3.0).abs() < 1e-6, "b = {}", res.params[1]);
    }

    #[test]
    fn levenberg_marquardt_nonlinear_exponential() {
        // Fit y = exp(a*x) to data generated with a = 0.5.
        let xs = [0.0_f64, 0.5, 1.0, 1.5, 2.0];
        let res = levenberg_marquardt(
            |p: &[f64]| {
                xs.iter()
                    .map(|&x| {
                        let model = (p[0] * x).exp();
                        let y = (0.5 * x).exp();
                        (model - y, 1.0, vec![x * model])
                    })
                    .collect::<Vec<_>>()
            },
            &[0.1],
            1e-14,
            300,
        );
        assert!((res.params[0] - 0.5).abs() < 1e-5, "a = {}", res.params[0]);
    }
}

// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Quasi-explicit slice calibration (de Marco & Martini / Zeliade 2009).
//!
//! Direct least-squares over all five raw parameters is non-convex and
//! sensitive to the starting point. The quasi-explicit method removes that
//! fragility by exploiting a change of variables that makes the problem
//! linear in three of the five parameters.
//!
//! # Reduction
//!
//! Fix the two nonlinear parameters `m` and `sigma`. Substituting
//! `y = (k - m)/sigma` gives `sqrt((k-m)^2 + sigma^2) = sigma*sqrt(y^2 + 1)`,
//! so with `c = b*sigma` and `d = rho*b*sigma` the model becomes affine in
//! `(a, d, c)`:
//!
//! ```text
//! w(y) = a + d*y + c*sqrt(y^2 + 1)
//! ```
//!
//! # Inner problem — convex, solved exactly
//!
//! For fixed `(m, sigma)`, minimise the weighted residual over `(a, d, c)` on
//! the convex Zeliade domain `D`:
//!
//! ```text
//! 0 <= c <= 4*sigma
//! |d| <= c          and      |d| <= 4*sigma - c
//! 0 <= a <= max_i w_i
//! ```
//!
//! `f` is a convex quadratic, so its minimum over the polytope `D` is either
//! the unconstrained stationary point (from the 3x3 normal equations) or, if
//! that is infeasible, lies on a face — found by enumerating every face,
//! edge, and vertex of `D` and taking the feasible minimiser. With three
//! variables this enumeration is small and exact.
//!
//! # Outer problem — two-dimensional
//!
//! Let `f*(m, sigma)` be the optimal inner residual. The 2-D, mildly
//! non-convex problem `min f*(m, sigma)` is solved with the Nelder-Mead
//! simplex, multi-started across a small grid of `(m, sigma)` seeds.
//!
//! # Recovery
//!
//! From the optimal `(a, c, d, m, sigma)`: `b = c/sigma`, `rho = d/c`
//! (`b = 0`, `rho = 0` when `c = 0`).
//!
//! # References
//!
//! - De Marco, S. & Martini, C., "Quasi-explicit calibration of Gatheral's
//!   SVI model", Zeliade Systems White Paper ZWP-0005 (2009).

use crate::calibration::CalibrationResult;
use crate::errors::CalibrationError;
use crate::math::{nelder_mead, solve_spd_3};
use crate::raw::RawSvi;
use crate::types::Quote;

/// Minimum number of quotes the five-parameter raw SVI model can be fit to.
const MIN_QUOTES: usize = 5;
/// Nelder-Mead tolerance for the 2-D outer search.
const OUTER_TOL: f64 = 1e-12;
/// Nelder-Mead iteration cap for the outer search.
const OUTER_MAX_ITER: usize = 2000;

/// Calibrates a raw SVI slice to market quotes by the quasi-explicit method.
///
/// Runs the inner convex QP in `(a, d, c)` for each `(m, sigma)` proposed by a
/// multi-started Nelder-Mead outer search, recovers `b` and `rho`, and
/// returns the fitted slice with its RMSE and butterfly flag.
///
/// # Errors
///
/// - [`CalibrationError::EmptyQuotes`] if `quotes` is empty.
/// - [`CalibrationError::TooFewQuotes`] if fewer than five quotes are given.
/// - [`CalibrationError::AllWeightsZero`] if every fitting weight is zero.
/// - [`CalibrationError::Param`] if the recovered parameters are invalid.
///
/// # Examples
///
/// ```
/// use regit_svi::types::Quote;
/// use regit_svi::calibration::quasi_explicit::calibrate;
///
/// let quotes = [
///     Quote::new(-0.20, 0.0512, 1.0).unwrap(),
///     Quote::new(-0.10, 0.0432, 1.0).unwrap(),
///     Quote::new( 0.00, 0.0400, 1.0).unwrap(),
///     Quote::new( 0.10, 0.0420, 1.0).unwrap(),
///     Quote::new( 0.20, 0.0480, 1.0).unwrap(),
/// ];
/// let fit = calibrate(&quotes).unwrap();
/// assert!(fit.rmse < 1e-2);
/// ```
pub fn calibrate(quotes: &[Quote]) -> Result<CalibrationResult, CalibrationError> {
    if quotes.is_empty() {
        return Err(CalibrationError::EmptyQuotes);
    }
    if quotes.len() < MIN_QUOTES {
        return Err(CalibrationError::TooFewQuotes {
            got: quotes.len(),
            need: MIN_QUOTES,
        });
    }
    if quotes.iter().all(|q| q.weight <= 0.0) {
        return Err(CalibrationError::AllWeightsZero);
    }

    let k_min = quotes.iter().map(|q| q.k).fold(f64::INFINITY, f64::min);
    let k_max = quotes.iter().map(|q| q.k).fold(f64::NEG_INFINITY, f64::max);
    let k_span = (k_max - k_min).max(1e-3);
    let w_max = quotes
        .iter()
        .map(|q| q.w)
        .fold(0.0_f64, f64::max)
        .max(1e-12);

    // The outer objective: optimal inner residual at (m, sigma).
    // Parametrise sigma = exp(sigma_hat) to keep it strictly positive.
    let outer = |p: &[f64]| -> f64 {
        let m = p[0];
        let sigma = p[1].exp();
        if !m.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return f64::INFINITY;
        }
        inner_solve(quotes, m, sigma, w_max).0
    };

    // Multi-start grid of (m, sigma) seeds covering the quoted range.
    let m_seeds = [
        k_min,
        0.5 * (k_min + k_max),
        k_max,
        k_min - 0.25 * k_span,
        k_max + 0.25 * k_span,
    ];
    let sigma_seeds = [0.1 * k_span, 0.3 * k_span, k_span, 2.0 * k_span];

    let mut best_obj = f64::INFINITY;
    let mut best_m = 0.5 * (k_min + k_max);
    let mut best_sigma = 0.3 * k_span;

    for &m0 in &m_seeds {
        for &s0 in &sigma_seeds {
            let start = [m0, s0.max(1e-6).ln()];
            let res = nelder_mead(outer, &start, OUTER_TOL, OUTER_MAX_ITER);
            if res.fx < best_obj {
                best_obj = res.fx;
                best_m = res.x[0];
                best_sigma = res.x[1].exp();
            }
        }
    }

    // Recover the full parameter set from the best (m, sigma).
    let (resid, a, d, c) = inner_solve(quotes, best_m, best_sigma, w_max);
    let b = if best_sigma > 0.0 {
        c / best_sigma
    } else {
        0.0
    };
    let rho = if c.abs() > 1e-300 {
        (d / c).clamp(-0.999_999, 0.999_999)
    } else {
        0.0
    };

    let total_weight: f64 = quotes.iter().map(|q| q.weight).sum();
    let rmse = if total_weight > 0.0 {
        (resid / total_weight).sqrt()
    } else {
        0.0
    };

    let slice = RawSvi::new(a, b, rho, best_m, best_sigma).map_err(CalibrationError::Param)?;
    Ok(CalibrationResult::new(slice, rmse))
}

/// Solves the inner convex QP for fixed `(m, sigma)`.
///
/// Returns `(residual, a, d, c)`: the optimal weighted sum of squared
/// residuals and the affine coefficients of `w(y) = a + d*y + c*sqrt(y^2+1)`,
/// minimised over the Zeliade domain `D` (MATH.md §10).
///
/// The face/edge/vertex enumeration of the 3-D polytope `D` is exhaustive by
/// nature, so the function is necessarily long; it is one cohesive solver.
#[allow(clippy::too_many_lines)]
fn inner_solve(quotes: &[Quote], m: f64, sigma: f64, w_max: f64) -> (f64, f64, f64, f64) {
    // Design data: phi_i = (1, y_i, z_i) with y_i = (k_i-m)/sigma,
    // z_i = sqrt(y_i^2 + 1).
    let mut rows: Vec<([f64; 3], f64, f64)> = Vec::with_capacity(quotes.len());
    for q in quotes {
        if q.weight <= 0.0 {
            continue;
        }
        let y = (q.k - m) / sigma;
        let z = (y * y + 1.0).sqrt();
        rows.push(([1.0, y, z], q.w, q.weight));
    }
    if rows.is_empty() {
        return (f64::INFINITY, 0.0, 0.0, 0.0);
    }

    // Weighted normal-equations matrices: A (symmetric 3x3) and rhs (3).
    // A_jl = sum_i weight_i * phi_ij * phi_il
    // rhs_j = sum_i weight_i * phi_ij * w_i
    let mut a_mat = [0.0_f64; 6]; // [a00, a01, a02, a11, a12, a22]
    let mut rhs = [0.0_f64; 3];
    for (phi, w, weight) in &rows {
        let ww = *weight;
        a_mat[0] += ww * phi[0] * phi[0];
        a_mat[1] += ww * phi[0] * phi[1];
        a_mat[2] += ww * phi[0] * phi[2];
        a_mat[3] += ww * phi[1] * phi[1];
        a_mat[4] += ww * phi[1] * phi[2];
        a_mat[5] += ww * phi[2] * phi[2];
        rhs[0] += ww * phi[0] * w;
        rhs[1] += ww * phi[1] * w;
        rhs[2] += ww * phi[2] * w;
    }

    // Residual of a candidate (a, d, c) — used to compare feasible vertices.
    let residual_of = |a: f64, d: f64, c: f64| -> f64 {
        rows.iter()
            .map(|(phi, w, weight)| {
                let model = a * phi[0] + d * phi[1] + c * phi[2];
                let r = model - w;
                weight * r * r
            })
            .sum()
    };

    // Zeliade domain D for (a, d, c):
    //   0 <= a <= w_max
    //   0 <= c <= 4*sigma
    //   |d| <= c   and   |d| <= 4*sigma - c
    let c_hi = 4.0 * sigma;
    let feasible = |a: f64, d: f64, c: f64| -> bool {
        let eps = 1e-9 * (1.0 + c_hi + w_max);
        a >= -eps
            && a <= w_max + eps
            && c >= -eps
            && c <= c_hi + eps
            && d.abs() <= c + eps
            && d.abs() <= c_hi - c + eps
    };

    let mut best_resid = f64::INFINITY;
    let mut best = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut consider = |a: f64, d: f64, c: f64| {
        if a.is_finite() && d.is_finite() && c.is_finite() && feasible(a, d, c) {
            let r = residual_of(a, d, c);
            if r < best_resid {
                best_resid = r;
                best = (a, d, c);
            }
        }
    };

    // (1) Unconstrained stationary point — solve the 3x3 normal equations.
    if let Some(x) = solve_spd_3(&a_mat, &rhs) {
        consider(x[0], x[1], x[2]);
    }

    // (2) Faces: fix one variable / one inequality to its boundary and solve
    //     the reduced 2x2 least-squares problem in the other two.
    //
    // The polytope D has these face-defining equalities:
    //   a = 0, a = w_max,
    //   c = 0, c = c_hi,
    //   d = c, d = -c, d = c_hi - c, d = -(c_hi - c).
    //
    // For each, solve the unconstrained reduced problem; clamp/clip is not
    // needed because all edges and vertices are enumerated separately below.

    // Faces a = const.
    for &a_fix in &[0.0, w_max] {
        // Minimise over (d, c): normal equations for the 2-vector (d, c).
        // A2 = [[a11, a12],[a12, a22]], rhs2 = [rhs1 - a01*a_fix, rhs2 - a02*a_fix].
        if let Some((d, c)) = solve_2x2(
            a_mat[3],
            a_mat[4],
            a_mat[5],
            rhs[1] - a_mat[1] * a_fix,
            rhs[2] - a_mat[2] * a_fix,
        ) {
            consider(a_fix, d, c);
        }
    }
    // Faces c = const.
    for &c_fix in &[0.0, c_hi] {
        // Minimise over (a, d).
        if let Some((a, d)) = solve_2x2(
            a_mat[0],
            a_mat[1],
            a_mat[3],
            rhs[0] - a_mat[2] * c_fix,
            rhs[1] - a_mat[4] * c_fix,
        ) {
            consider(a, d, c_fix);
        }
    }
    // Faces d = s*c (s = +/-1): substitute d = s*c, minimise over (a, c).
    for &s in &[1.0_f64, -1.0] {
        // Model column for c becomes (s*phi_d + phi_c); rebuild 2x2 in (a, c).
        // A00 = a00, A01 = s*a01 + a02, A11 = a22 + 2s*a12 + a11.
        let a00 = a_mat[0];
        let a01 = s * a_mat[1] + a_mat[2];
        let a11 = a_mat[5] + 2.0 * s * a_mat[4] + a_mat[3];
        let r0 = rhs[0];
        let r1 = s * rhs[1] + rhs[2];
        if let Some((a, c)) = solve_2x2(a00, a01, a11, r0, r1) {
            consider(a, s * c, c);
        }
    }
    // Faces d = s*(c_hi - c): substitute d = s*c_hi - s*c, minimise over (a, c).
    for &s in &[1.0_f64, -1.0] {
        // d = s*c_hi - s*c. Model: a*1 + (s*c_hi - s*c)*phi_d + c*phi_c.
        // = a + s*c_hi*phi_d + c*(phi_c - s*phi_d).
        // 2x2 in (a, c): column_a = 1, column_c = phi_c - s*phi_d.
        // A00 = a00, A01 = a02 - s*a01, A11 = a22 - 2s*a12 + a11.
        // rhs: r0 = rhs0 - s*c_hi*rhs1 ; r1 = (rhs2 - s*rhs1) - s*c_hi*(a12 - s*a11)
        let a00 = a_mat[0];
        let a01 = a_mat[2] - s * a_mat[1];
        let a11 = a_mat[5] - 2.0 * s * a_mat[4] + a_mat[3];
        let r0 = rhs[0] - s * c_hi * a_mat[1];
        let r1 = (rhs[2] - s * rhs[1]) - s * c_hi * (a_mat[4] - s * a_mat[3]);
        if let Some((a, c)) = solve_2x2(a00, a01, a11, r0, r1) {
            consider(a, s * c_hi - s * c, c);
        }
    }

    // (3) Edges: pairs of equalities. Solve the resulting 1-D least squares.
    //     The relevant edges are intersections of a = const with the (d, c)
    //     boundary lines, and of c = const with the d boundary lines.
    for &a_fix in &[0.0, w_max] {
        for &c_fix in &[0.0, c_hi] {
            // Minimise over d alone.
            // residual sum d^2*a11 - 2*d*(rhs1 - a01*a_fix - a12*c_fix) + const.
            let denom = a_mat[3];
            if denom > 0.0 {
                let d = (rhs[1] - a_mat[1] * a_fix - a_mat[4] * c_fix) / denom;
                consider(a_fix, d, c_fix);
            }
        }
        for &s in &[1.0_f64, -1.0] {
            // a fixed, d = s*c: minimise over c.
            let a11 = a_mat[5] + 2.0 * s * a_mat[4] + a_mat[3];
            if a11 > 0.0 {
                let c = (s * (rhs[1] - a_mat[1] * a_fix) + (rhs[2] - a_mat[2] * a_fix)) / a11;
                consider(a_fix, s * c, c);
            }
            // a fixed, d = s*(c_hi - c): minimise over c.
            let a11b = a_mat[5] - 2.0 * s * a_mat[4] + a_mat[3];
            if a11b > 0.0 {
                let r = (rhs[2] - a_mat[2] * a_fix)
                    - s * (rhs[1] - a_mat[1] * a_fix)
                    - s * c_hi * (a_mat[4] - s * a_mat[3]);
                let c = r / a11b;
                consider(a_fix, s * c_hi - s * c, c);
            }
        }
    }

    // (4) Vertices: all corners of D. With the box on a and c, and the
    //     wing constraints |d| <= c, |d| <= c_hi - c, the vertices in (d, c)
    //     are (d, c) in {(0, 0), (0, c_hi), (c, c) with c = c_hi/2 -> d = +-c_hi/2}.
    let half = c_hi / 2.0;
    let dc_vertices = [(0.0, 0.0), (0.0, c_hi), (half, half), (-half, half)];
    for &a_fix in &[0.0, w_max] {
        for &(d, c) in &dc_vertices {
            consider(a_fix, d, c);
        }
    }
    // Also: a free at each (d, c) vertex.
    for &(d, c) in &dc_vertices {
        // minimise over a alone.
        if a_mat[0] > 0.0 {
            let a = (rhs[0] - a_mat[1] * d - a_mat[2] * c) / a_mat[0];
            consider(a, d, c);
        }
    }

    (best_resid, best.0, best.1, best.2)
}

/// Solves the symmetric `2x2` system `[[a00, a01],[a01, a11]] x = [r0, r1]`.
///
/// Returns `None` if the system is singular or not positive definite.
fn solve_2x2(a00: f64, a01: f64, a11: f64, r0: f64, r1: f64) -> Option<(f64, f64)> {
    let det = a00 * a11 - a01 * a01;
    if det.abs() < 1e-300 || !det.is_finite() {
        return None;
    }
    let x0 = (a11 * r0 - a01 * r1) / det;
    let x1 = (a00 * r1 - a01 * r0) / det;
    if x0.is_finite() && x1.is_finite() {
        Some((x0, x1))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generates a synthetic slice of quotes from known raw SVI parameters.
    fn synthetic(svi: &RawSvi, ks: &[f64]) -> Vec<Quote> {
        ks.iter()
            .map(|&k| Quote::new(k, svi.total_variance(k), 1.0).unwrap())
            .collect()
    }

    #[test]
    fn rejects_empty() {
        assert!(matches!(calibrate(&[]), Err(CalibrationError::EmptyQuotes)));
    }

    #[test]
    fn rejects_too_few_quotes() {
        let q = Quote::new(0.0, 0.04, 1.0).unwrap();
        assert!(matches!(
            calibrate(&[q, q, q]),
            Err(CalibrationError::TooFewQuotes { .. })
        ));
    }

    #[test]
    fn rejects_all_zero_weights() {
        let quotes: Vec<Quote> = (0..6)
            .map(|i| Quote::new(f64::from(i) * 0.1 - 0.3, 0.04, 0.0).unwrap())
            .collect();
        assert!(matches!(
            calibrate(&quotes),
            Err(CalibrationError::AllWeightsZero)
        ));
    }

    #[test]
    fn recovers_synthetic_parameters() {
        let truth = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.15).unwrap();
        let ks = [-0.4, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4];
        let quotes = synthetic(&truth, &ks);
        let fit = calibrate(&quotes).unwrap();

        // RMSE should be near machine zero for noise-free synthetic data.
        assert!(fit.rmse < 1e-5, "rmse = {}", fit.rmse);
        // The fitted slice must reproduce total variance everywhere.
        for &k in &[-0.6, -0.2, 0.0, 0.2, 0.6] {
            let err = (fit.slice.total_variance(k) - truth.total_variance(k)).abs();
            assert!(err < 1e-4, "k = {k}, err = {err}");
        }
    }

    #[test]
    fn recovers_symmetric_smile() {
        let truth = RawSvi::new(0.03, 0.3, 0.0, 0.0, 0.2).unwrap();
        let ks = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5];
        let quotes = synthetic(&truth, &ks);
        let fit = calibrate(&quotes).unwrap();
        assert!(fit.rmse < 1e-5, "rmse = {}", fit.rmse);
        assert!(fit.slice.rho.abs() < 1e-2, "rho = {}", fit.slice.rho);
    }

    #[test]
    fn recovers_positive_skew() {
        let truth = RawSvi::new(0.05, 0.35, 0.4, -0.1, 0.18).unwrap();
        let ks = [-0.4, -0.2, -0.05, 0.05, 0.2, 0.4, 0.6];
        let quotes = synthetic(&truth, &ks);
        let fit = calibrate(&quotes).unwrap();
        assert!(fit.rmse < 1e-4, "rmse = {}", fit.rmse);
        for &k in &[-0.3, 0.0, 0.3] {
            let err = (fit.slice.total_variance(k) - truth.total_variance(k)).abs();
            assert!(err < 1e-3, "k = {k}, err = {err}");
        }
    }

    #[test]
    fn graceful_degradation_with_noise() {
        let truth = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.15).unwrap();
        let ks = [-0.4, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4];
        // Deterministic pseudo-noise from a small LCG.
        let mut state = 12_345_u64;
        let quotes: Vec<Quote> = ks
            .iter()
            .map(|&k| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let noise =
                    (f64::from((state >> 40) as u32) / f64::from(u32::MAX) - 0.5) * 2.0 * 5e-4;
                Quote::new(k, truth.total_variance(k) + noise, 1.0).unwrap()
            })
            .collect();
        let fit = calibrate(&quotes).unwrap();
        // With small noise the fit should still be close.
        assert!(fit.rmse < 1e-2, "rmse = {}", fit.rmse);
        let atm_err = (fit.slice.total_variance(0.0) - truth.total_variance(0.0)).abs();
        assert!(atm_err < 5e-3, "atm err = {atm_err}");
    }

    #[test]
    fn solve_2x2_identity() {
        let (x, y) = solve_2x2(1.0, 0.0, 1.0, 3.0, 7.0).unwrap();
        assert!((x - 3.0).abs() < 1e-15);
        assert!((y - 7.0).abs() < 1e-15);
    }

    #[test]
    fn solve_2x2_rejects_singular() {
        assert!(solve_2x2(1.0, 1.0, 1.0, 1.0, 1.0).is_none());
    }
}

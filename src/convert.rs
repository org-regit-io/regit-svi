// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Conversions between the Raw, Jump-Wings, and SSVI parametrisations.
//!
//! Three parametrisations are supported; the conversions form this graph:
//!
//! ```text
//!    Raw  <------------------>  Jump-Wings        (bijective, per MATH.md §4)
//!     ^
//!     |  slice-at-fixed-theta
//!     |
//!   SSVI                                          (one-directional: SSVI -> Raw)
//! ```
//!
//! - **Raw <-> JW** — bijective for any maturity `t > 0`, subject to the
//!   existence condition `|beta| <= 1` on the JW side.
//! - **SSVI -> Raw** — every SSVI slice at fixed `theta` is a raw SVI slice
//!   (see [`crate::ssvi::Ssvi::slice_at`]).
//! - **Raw -> SSVI** is not defined: SSVI is a constrained three-parameter
//!   family, so a generic raw slice has no SSVI pre-image.
//!
//! # Raw -> JW (forward map)
//!
//! Given `{a, b, rho, m, sigma}` and maturity `t`, with
//! `w0 = a + b*(-rho*m + sqrt(m^2+sigma^2))`:
//!
//! ```text
//! v_t       = w0 / t
//! psi_t     = (b/(2*sqrt(w0))) * (rho - m/sqrt(m^2+sigma^2))
//! p_t       = (b/sqrt(w0)) * (1 - rho)
//! c_t       = (b/sqrt(w0)) * (1 + rho)
//! v_tilde_t = (a + b*sigma*sqrt(1-rho^2)) / t
//! ```
//!
//! # JW -> Raw (inverse map)
//!
//! Given `{v_t, psi_t, p_t, c_t, v_tilde_t}` and maturity `t`, with
//! `w = v_t * t`:
//!
//! ```text
//! b     = (sqrt(w)/2) * (c_t + p_t)
//! rho   = 1 - p_t*sqrt(w)/b
//! beta  = rho - 2*psi_t*sqrt(w)/b
//! alpha = sign(beta) * sqrt(1/beta^2 - 1)
//! m     = (v_t - v_tilde_t)*t
//!         / ( b*(-rho + sign(alpha)*sqrt(1+alpha^2) - alpha*sqrt(1-rho^2)) )
//! sigma = alpha * m
//! a     = v_tilde_t*t - b*sigma*sqrt(1-rho^2)
//! ```
//!
//! `alpha` is real only if `|beta| <= 1`; otherwise the JW tuple has no raw
//! pre-image and a [`ConvertError`] is returned. When `v_t = v_tilde_t` the
//! ATM point coincides with the vertex (`m = 0`), handled as a separate
//! branch (MATH.md §4).
//!
//! # References
//!
//! - Gatheral, J. & Jacquier, A., "Arbitrage-free SVI volatility surfaces",
//!   *Quantitative Finance* 14(1):59-71 (2014), Sections 3.2 and 4.

use crate::errors::{ConvertError, ParamError};
use crate::jw::SviJw;
use crate::raw::RawSvi;
use crate::ssvi::Ssvi;

/// Converts a raw SVI slice to the Jump-Wings parametrisation at maturity `t`.
///
/// Implements the forward map of MATH.md §4. The result is always a valid
/// Jump-Wings tuple for a valid raw slice and a positive maturity.
///
/// # Errors
///
/// - [`ConvertError::NonPositiveAtmVariance`] if `t <= 0` or the ATM total
///   variance `w0` is not strictly positive (a degenerate slice).
/// - [`ConvertError::Param`] if the resulting tuple fails validation.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::convert::raw_to_jw;
///
/// let raw = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.12).unwrap();
/// let jw = raw_to_jw(&raw, 1.0).unwrap();
/// // v_t is the ATM variance.
/// assert!((jw.v_t - raw.total_variance(0.0)).abs() < 1e-12);
/// ```
pub fn raw_to_jw(raw: &RawSvi, t: f64) -> Result<SviJw, ConvertError> {
    if t <= 0.0 || !t.is_finite() {
        return Err(ConvertError::NonPositiveAtmVariance { w: t });
    }
    let w0 = raw.atm_total_variance();
    if w0 <= 0.0 || !w0.is_finite() {
        return Err(ConvertError::NonPositiveAtmVariance { w: w0 });
    }
    let sqrt_w0 = w0.sqrt();
    let root_m = (raw.m * raw.m + raw.sigma * raw.sigma).sqrt();

    let v_t = w0 / t;
    let psi_t = (raw.b / (2.0 * sqrt_w0)) * (raw.rho - raw.m / root_m);
    let p_t = (raw.b / sqrt_w0) * (1.0 - raw.rho);
    let c_t = (raw.b / sqrt_w0) * (1.0 + raw.rho);
    let v_tilde_t = raw.w_min() / t;

    SviJw::new(v_t, psi_t, p_t, c_t, v_tilde_t, t).map_err(ConvertError::Param)
}

/// Converts a Jump-Wings slice back to the raw SVI parametrisation.
///
/// Implements the inverse map of MATH.md §4, including the `|beta| <= 1`
/// existence check and the `m = 0` degenerate branch (which arises when
/// `v_t = v_tilde_t`, i.e. the ATM point coincides with the vertex).
///
/// # Errors
///
/// - [`ConvertError::DegenerateJw`] if `b = 0` (`c_t + p_t = 0`) makes the
///   inverse indeterminate.
/// - [`ConvertError::JwHasNoRawPreimage`] if `|beta| > 1`.
/// - [`ConvertError::Param`] if the resulting raw slice fails validation.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::convert::{raw_to_jw, jw_to_raw};
///
/// let raw = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.12).unwrap();
/// let back = jw_to_raw(&raw_to_jw(&raw, 1.0).unwrap()).unwrap();
/// // Raw -> JW -> Raw is the identity.
/// assert!((back.a - raw.a).abs() < 1e-9);
/// assert!((back.sigma - raw.sigma).abs() < 1e-9);
/// ```
// The raw SVI parameters (a, b, m, t, w) keep their canonical single-letter
// names from MATH.md §4; renaming them would obscure the inverse-map formula.
#[allow(clippy::many_single_char_names)]
pub fn jw_to_raw(jw: &SviJw) -> Result<RawSvi, ConvertError> {
    let t = jw.t;
    let w = jw.v_t * t;
    if w <= 0.0 || !w.is_finite() {
        return Err(ConvertError::NonPositiveAtmVariance { w });
    }
    let sqrt_w = w.sqrt();

    // b = (sqrt(w)/2) * (c_t + p_t).
    let b = (sqrt_w / 2.0) * (jw.c_t + jw.p_t);
    if b <= 0.0 || !b.is_finite() {
        return Err(ConvertError::DegenerateJw);
    }

    // rho = 1 - p_t*sqrt(w)/b   ( = (c_t - p_t)/(c_t + p_t) ).
    let rho = 1.0 - jw.p_t * sqrt_w / b;

    // beta = rho - 2*psi_t*sqrt(w)/b.
    let beta = rho - 2.0 * jw.psi_t * sqrt_w / b;
    if beta.abs() > 1.0 {
        return Err(ConvertError::JwHasNoRawPreimage { beta });
    }

    let one_minus_rho2 = 1.0 - rho * rho;
    let sqrt_one_minus_rho2 = one_minus_rho2.sqrt();
    let numerator = (jw.v_t - jw.v_tilde_t) * t;

    // beta = m / sqrt(m^2 + sigma^2), so beta = 0 is exactly the m = 0 case.
    let (m, sigma) = if beta.abs() < 1e-12 {
        // Degenerate branch m = 0: the ATM point is the vertex.
        // From w0 - w_min = b*sigma*(1 - sqrt(1-rho^2)):
        // sigma = (v_t - v_tilde_t)*t / ( b*(1 - sqrt(1-rho^2)) ).
        let denom = b * (1.0 - sqrt_one_minus_rho2);
        if denom.abs() < 1e-300 {
            // rho = 0 and m = 0: w0 = w_min, so sigma is unconstrained by the
            // Jump-Wings tuple — the inverse map is genuinely ambiguous.
            return Err(ConvertError::DegenerateJw);
        }
        let sigma = numerator / denom;
        (0.0, sigma)
    } else {
        // alpha = sign(beta) * sqrt(1/beta^2 - 1) = sigma / m.
        let alpha = sign(beta) * (1.0 / (beta * beta) - 1.0).sqrt();
        // m = (v_t - v_tilde_t)*t
        //     / ( b*(-rho + sign(alpha)*sqrt(1+alpha^2) - alpha*sqrt(1-rho^2)) ).
        let bracket =
            -rho + sign(alpha) * (1.0 + alpha * alpha).sqrt() - alpha * sqrt_one_minus_rho2;
        let denom = b * bracket;
        if denom.abs() < 1e-12 || !denom.is_finite() {
            // bracket = 0: the vertex coincides with the ATM point while
            // m != 0, so the Jump-Wings tuple maps a one-parameter family of
            // raw slices — the inverse map is ambiguous.
            return Err(ConvertError::DegenerateJw);
        }
        let m = numerator / denom;
        let sigma = alpha * m;
        (m, sigma)
    };

    // a = v_tilde_t*t - b*sigma*sqrt(1-rho^2).
    let a = jw.v_tilde_t * t - b * sigma * sqrt_one_minus_rho2;

    if sigma <= 0.0 {
        return Err(ConvertError::Param(ParamError::NonPositiveSigma { sigma }));
    }

    RawSvi::new(a, b, rho, m, sigma).map_err(ConvertError::Param)
}

/// Converts an SSVI surface at fixed `theta` to a raw SVI slice.
///
/// A thin wrapper over [`Ssvi::slice_at`], provided so all conversions are
/// reachable from one module.
///
/// # Errors
///
/// Returns [`ConvertError::Param`] for the same conditions as
/// [`Ssvi::slice_at`].
///
/// # Examples
///
/// ```
/// use regit_svi::ssvi::{Phi, Ssvi};
/// use regit_svi::convert::ssvi_to_raw;
///
/// let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
/// let raw = ssvi_to_raw(&ssvi, 0.04).unwrap();
/// assert!(raw.total_variance(0.0) > 0.0);
/// ```
pub fn ssvi_to_raw(ssvi: &Ssvi, theta: f64) -> Result<RawSvi, ConvertError> {
    ssvi.slice_at(theta).map_err(ConvertError::Param)
}

/// Returns `+1.0` for non-negative input and `-1.0` for negative input.
///
/// Matches the `sign` used in the JW inverse map (MATH.md §4); zero maps to
/// `+1.0`, consistent with `f64::signum` for `+0.0`.
#[inline]
fn sign(x: f64) -> f64 {
    if x < 0.0 { -1.0 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssvi::Phi;

    /// Asserts two raw slices agree to a tolerance on every parameter.
    fn assert_raw_close(a: &RawSvi, b: &RawSvi, tol: f64) {
        assert!((a.a - b.a).abs() < tol, "a: {} vs {}", a.a, b.a);
        assert!((a.b - b.b).abs() < tol, "b: {} vs {}", a.b, b.b);
        assert!((a.rho - b.rho).abs() < tol, "rho: {} vs {}", a.rho, b.rho);
        assert!((a.m - b.m).abs() < tol, "m: {} vs {}", a.m, b.m);
        assert!(
            (a.sigma - b.sigma).abs() < tol,
            "sigma: {} vs {}",
            a.sigma,
            b.sigma
        );
    }

    #[test]
    fn raw_jw_roundtrip_identity() {
        let raw = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.12).unwrap();
        let jw = raw_to_jw(&raw, 1.0).unwrap();
        let back = jw_to_raw(&jw).unwrap();
        assert_raw_close(&raw, &back, 1e-9);
    }

    #[test]
    fn raw_jw_roundtrip_positive_rho() {
        let raw = RawSvi::new(0.05, 0.3, 0.4, -0.1, 0.2).unwrap();
        let jw = raw_to_jw(&raw, 2.0).unwrap();
        let back = jw_to_raw(&jw).unwrap();
        assert_raw_close(&raw, &back, 1e-8);
    }

    #[test]
    fn raw_jw_roundtrip_various_maturities() {
        for &t in &[0.25, 0.5, 1.0, 2.0, 5.0] {
            let raw = RawSvi::new(0.03, 0.35, -0.2, 0.02, 0.15).unwrap();
            let jw = raw_to_jw(&raw, t).unwrap();
            let back = jw_to_raw(&jw).unwrap();
            assert_raw_close(&raw, &back, 1e-8);
        }
    }

    #[test]
    fn raw_to_jw_v_t_is_atm_variance() {
        let raw = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.12).unwrap();
        let jw = raw_to_jw(&raw, 1.0).unwrap();
        assert!((jw.v_t * jw.t - raw.atm_total_variance()).abs() < 1e-12);
        assert!((jw.v_tilde_t * jw.t - raw.w_min()).abs() < 1e-12);
    }

    #[test]
    fn raw_to_jw_rejects_bad_maturity() {
        let raw = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.12).unwrap();
        assert!(raw_to_jw(&raw, 0.0).is_err());
    }

    #[test]
    fn jw_to_raw_handles_m_zero_branch() {
        // A raw slice with m = 0 exactly (and rho != 0) is the genuine m = 0
        // degenerate branch (beta = m/sqrt(m^2+sigma^2) = 0). The inverse map
        // recovers it through the dedicated sigma formula.
        let raw = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.15).unwrap();
        let jw = raw_to_jw(&raw, 1.0).unwrap();
        assert!(
            jw.psi_t.abs() > 1e-9,
            "m = 0 with rho != 0 has non-zero skew"
        );
        let back = jw_to_raw(&jw).unwrap();
        assert!(
            back.m.abs() < 1e-9,
            "recovered m should be 0, got {}",
            back.m
        );
        for &k in &[-0.3, 0.0, 0.3] {
            assert!(
                (back.total_variance(k) - raw.total_variance(k)).abs() < 1e-8,
                "k = {k}"
            );
        }
    }

    #[test]
    fn jw_to_raw_rejects_ambiguous_vertex_at_atm() {
        // A raw slice whose vertex sits at k = 0 with m != 0 produces a
        // Jump-Wings tuple that maps a one-parameter family of raw slices,
        // so the inverse map must report it as ambiguous.
        let rho = -0.3_f64;
        let sigma = 0.15_f64;
        let m = rho * sigma / (1.0 - rho * rho).sqrt();
        let raw = RawSvi::new(0.04, 0.4, rho, m, sigma).unwrap();
        let jw = raw_to_jw(&raw, 1.0).unwrap();
        assert!((jw.v_t - jw.v_tilde_t).abs() < 1e-12);
        assert!(matches!(jw_to_raw(&jw), Err(ConvertError::DegenerateJw)));
    }

    #[test]
    fn jw_to_raw_rejects_no_preimage() {
        // psi_t too large -> |beta| > 1.
        let jw = SviJw::new_unchecked(0.04, 5.0, 0.3, 0.25, 0.035, 1.0);
        assert!(matches!(
            jw_to_raw(&jw),
            Err(ConvertError::JwHasNoRawPreimage { .. })
        ));
    }

    #[test]
    fn jw_to_raw_rejects_zero_b() {
        // c_t = p_t = 0 -> b = 0.
        let jw = SviJw::new_unchecked(0.04, 0.0, 0.0, 0.0, 0.04, 1.0);
        assert!(matches!(jw_to_raw(&jw), Err(ConvertError::DegenerateJw)));
    }

    #[test]
    fn ssvi_to_raw_matches_slice_at() {
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        let raw = ssvi_to_raw(&ssvi, 0.04).unwrap();
        let direct = ssvi.slice_at(0.04).unwrap();
        assert_raw_close(&raw, &direct, 1e-15);
    }

    #[test]
    fn ssvi_to_raw_rejects_bad_theta() {
        let ssvi = Ssvi::new(-0.3, Phi::heston(1.0).unwrap()).unwrap();
        assert!(ssvi_to_raw(&ssvi, 0.0).is_err());
    }

    #[test]
    fn jw_roundtrip_total_variance_preserved() {
        // The strongest invariant: w(k) is unchanged by Raw -> JW -> Raw.
        let raw = RawSvi::new(0.045, 0.38, -0.25, 0.03, 0.14).unwrap();
        let back = jw_to_raw(&raw_to_jw(&raw, 1.5).unwrap()).unwrap();
        for &k in &[-1.0, -0.4, 0.0, 0.4, 1.0] {
            assert!(
                (back.total_variance(k) - raw.total_variance(k)).abs() < 1e-9,
                "k = {k}"
            );
        }
    }
}

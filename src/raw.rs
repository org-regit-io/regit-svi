// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Raw SVI parametrisation of a single maturity slice.
//!
//! The raw SVI parametrisation expresses the total implied variance of one
//! slice as
//!
//! ```text
//! w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )
//! ```
//!
//! with five parameters `{a, b, rho, m, sigma}`:
//!
//! | Parameter | Domain      | Role                                    |
//! |-----------|-------------|-----------------------------------------|
//! | `a`       | `a in R`    | Vertical translation — variance level   |
//! | `b`       | `b >= 0`    | Slope of the wings                      |
//! | `rho`     | `|rho| < 1` | Counter-clockwise rotation — skew       |
//! | `m`       | `m in R`    | Horizontal translation — shifts smile   |
//! | `sigma`   | `sigma > 0` | ATM curvature — vertex smoothness       |
//!
//! Writing `u = k - m` and `r = sqrt(u^2 + sigma^2)`, the closed-form
//! derivatives are
//!
//! ```text
//! w'(k)  = b * ( rho + u / r )
//! w''(k) = b * sigma^2 / r^3
//! ```
//!
//! Because `b >= 0` and `sigma > 0`, `w''(k) > 0` everywhere: a raw SVI slice
//! is strictly convex in `k`. The slice attains its minimum
//! `w_min = a + b*sigma*sqrt(1-rho^2)` at `k_min = m - rho*sigma/sqrt(1-rho^2)`,
//! so non-negative variance holds iff `w_min >= 0`.
//!
//! # References
//!
//! - Gatheral, J., "A parsimonious arbitrage-free implied volatility
//!   parameterization with application to the valuation of volatility
//!   derivatives", Global Derivatives & Risk Management, Madrid (2004).
//! - Gatheral, J. & Jacquier, A., "Arbitrage-free SVI volatility surfaces",
//!   *Quantitative Finance* 14(1):59-71 (2014), Section 3.1.

use crate::errors::ParamError;

/// A raw SVI slice — five parameters describing one maturity's smile.
///
/// Constructed through [`RawSvi::new`] (validated against the raw SVI domain)
/// or [`RawSvi::new_unchecked`] (caller-asserted). Once built, all evaluation
/// methods are total: they never panic and never return `NaN` for finite `k`.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
///
/// let svi = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).unwrap();
/// // ATM total variance.
/// let w0 = svi.total_variance(0.0);
/// assert!(w0 > 0.0);
/// // Strict convexity: w'' is positive everywhere.
/// assert!(svi.w_double_prime(0.0) > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawSvi {
    /// Vertical translation `a` — the overall variance level.
    pub a: f64,
    /// Wing slope `b >= 0`.
    pub b: f64,
    /// Correlation / skew `rho`, with `|rho| < 1`.
    pub rho: f64,
    /// Horizontal translation `m`.
    pub m: f64,
    /// ATM curvature `sigma > 0`.
    pub sigma: f64,
}

impl RawSvi {
    /// Creates a validated raw SVI slice.
    ///
    /// Validation enforces the raw SVI domain (MATH.md §2): `b >= 0`,
    /// `|rho| < 1`, `sigma > 0`, all inputs finite, and the non-negative
    /// variance condition `a + b*sigma*sqrt(1-rho^2) >= 0`.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if any parameter is `NaN` or infinite.
    /// - [`ParamError::NegativeSlope`] if `b < 0`.
    /// - [`ParamError::CorrelationOutOfRange`] if `|rho| >= 1`.
    /// - [`ParamError::NonPositiveSigma`] if `sigma <= 0`.
    /// - [`ParamError::NegativeMinVariance`] if `w_min < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    /// use regit_svi::errors::ParamError;
    ///
    /// assert!(RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).is_ok());
    /// assert!(matches!(
    ///     RawSvi::new(0.04, -0.1, 0.0, 0.0, 0.1),
    ///     Err(ParamError::NegativeSlope { .. }),
    /// ));
    /// ```
    pub fn new(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Result<Self, ParamError> {
        for (name, value) in [("a", a), ("b", b), ("rho", rho), ("m", m), ("sigma", sigma)] {
            if !value.is_finite() {
                return Err(ParamError::NonFinite { name });
            }
        }
        if b < 0.0 {
            return Err(ParamError::NegativeSlope { b });
        }
        if rho.abs() >= 1.0 {
            return Err(ParamError::CorrelationOutOfRange { rho });
        }
        if sigma <= 0.0 {
            return Err(ParamError::NonPositiveSigma { sigma });
        }
        let candidate = Self {
            a,
            b,
            rho,
            m,
            sigma,
        };
        let w_min = candidate.w_min();
        if w_min < 0.0 {
            return Err(ParamError::NegativeMinVariance { w_min });
        }
        Ok(candidate)
    }

    /// Creates a raw SVI slice without validation.
    ///
    /// The caller asserts the raw SVI domain holds. Used on hot paths and by
    /// calibration code that has already constrained its search space.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new_unchecked(0.04, 0.4, -0.3, 0.0, 0.1);
    /// assert!((svi.a - 0.04).abs() < 1e-15);
    /// ```
    #[must_use]
    pub const fn new_unchecked(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Self {
        Self {
            a,
            b,
            rho,
            m,
            sigma,
        }
    }

    /// Re-checks the raw SVI domain for an existing slice.
    ///
    /// # Errors
    ///
    /// Returns the same [`ParamError`] variants as [`RawSvi::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new_unchecked(0.04, 0.4, -0.3, 0.0, 0.1);
    /// assert!(svi.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), ParamError> {
        Self::new(self.a, self.b, self.rho, self.m, self.sigma).map(|_| ())
    }

    /// Total implied variance `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// // With rho = 0 and m = 0, w(0) = a + b*sigma.
    /// let svi = RawSvi::new(0.04, 0.4, 0.0, 0.0, 0.1).unwrap();
    /// assert!((svi.total_variance(0.0) - (0.04 + 0.4 * 0.1)).abs() < 1e-15);
    /// ```
    #[must_use]
    #[inline]
    pub fn total_variance(&self, k: f64) -> f64 {
        let u = k - self.m;
        let r = u.mul_add(u, self.sigma * self.sigma).sqrt();
        self.b.mul_add(self.rho.mul_add(u, r), self.a)
    }

    /// First derivative `w'(k) = b * (rho + u/r)`, `u = k - m`,
    /// `r = sqrt(u^2 + sigma^2)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// // At the vertex k_min the slope vanishes.
    /// let svi = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).unwrap();
    /// assert!(svi.w_prime(svi.k_min()).abs() < 1e-12);
    /// ```
    #[must_use]
    #[inline]
    pub fn w_prime(&self, k: f64) -> f64 {
        let u = k - self.m;
        let r = u.mul_add(u, self.sigma * self.sigma).sqrt();
        self.b * (self.rho + u / r)
    }

    /// Second derivative `w''(k) = b * sigma^2 / r^3`, `r = sqrt(u^2 + sigma^2)`.
    ///
    /// Strictly positive for any valid slice (`b >= 0`, `sigma > 0`), so the
    /// slice is strictly convex.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).unwrap();
    /// assert!(svi.w_double_prime(0.5) > 0.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn w_double_prime(&self, k: f64) -> f64 {
        let u = k - self.m;
        let s2 = self.sigma * self.sigma;
        let r2 = u.mul_add(u, s2);
        let r = r2.sqrt();
        self.b * s2 / (r2 * r)
    }

    /// Black implied volatility at log-moneyness `k` and maturity `t`:
    /// `sqrt(w(k) / t)`.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NonPositiveMaturity`] if `t <= 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new(0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
    /// // Flat w = 0.04 -> vol = 0.2 at t = 1.
    /// assert!((svi.implied_vol(0.0, 1.0).unwrap() - 0.2).abs() < 1e-12);
    /// ```
    pub fn implied_vol(&self, k: f64, t: f64) -> Result<f64, ParamError> {
        if t <= 0.0 || !t.is_finite() {
            return Err(ParamError::NonPositiveMaturity { t });
        }
        Ok((self.total_variance(k) / t).sqrt())
    }

    /// Log-moneyness of the variance minimum:
    /// `k_min = m - rho*sigma/sqrt(1-rho^2)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// // With rho = 0 the vertex sits at k = m.
    /// let svi = RawSvi::new(0.04, 0.4, 0.0, 0.05, 0.1).unwrap();
    /// assert!((svi.k_min() - 0.05).abs() < 1e-15);
    /// ```
    #[must_use]
    #[inline]
    pub fn k_min(&self) -> f64 {
        let denom = (1.0 - self.rho * self.rho).sqrt();
        self.m - self.rho * self.sigma / denom
    }

    /// Minimum total variance `w_min = a + b*sigma*sqrt(1-rho^2)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).unwrap();
    /// // w_min is attained at k_min.
    /// assert!((svi.w_min() - svi.total_variance(svi.k_min())).abs() < 1e-12);
    /// ```
    #[must_use]
    #[inline]
    pub fn w_min(&self) -> f64 {
        self.b
            .mul_add(self.sigma * (1.0 - self.rho * self.rho).sqrt(), self.a)
    }

    /// ATM total variance `w(0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).unwrap();
    /// assert!((svi.atm_total_variance() - svi.total_variance(0.0)).abs() < 1e-15);
    /// ```
    #[must_use]
    #[inline]
    pub fn atm_total_variance(&self) -> f64 {
        self.total_variance(0.0)
    }

    /// ATM variance skew `w'(0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).unwrap();
    /// assert!((svi.atm_skew() - svi.w_prime(0.0)).abs() < 1e-15);
    /// ```
    #[must_use]
    #[inline]
    pub fn atm_skew(&self) -> f64 {
        self.w_prime(0.0)
    }

    /// ATM variance curvature `w''(0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).unwrap();
    /// assert!(svi.atm_curvature() > 0.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn atm_curvature(&self) -> f64 {
        self.w_double_prime(0.0)
    }

    /// Asymptotic slope of the left wing `b*(rho - 1)` (non-positive).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).unwrap();
    /// assert!(svi.left_wing_slope() <= 0.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn left_wing_slope(&self) -> f64 {
        self.b * (self.rho - 1.0)
    }

    /// Asymptotic slope of the right wing `b*(rho + 1)` (non-negative).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    ///
    /// let svi = RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).unwrap();
    /// assert!(svi.right_wing_slope() >= 0.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn right_wing_slope(&self) -> f64 {
        self.b * (self.rho + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative valid slice used across the tests.
    fn slice() -> RawSvi {
        RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.12).unwrap()
    }

    #[test]
    fn new_accepts_valid_slice() {
        assert!(RawSvi::new(0.04, 0.4, -0.3, 0.0, 0.1).is_ok());
    }

    #[test]
    fn new_rejects_negative_b() {
        assert!(matches!(
            RawSvi::new(0.04, -0.1, 0.0, 0.0, 0.1),
            Err(ParamError::NegativeSlope { .. })
        ));
    }

    #[test]
    fn new_rejects_rho_out_of_range() {
        assert!(matches!(
            RawSvi::new(0.04, 0.4, 1.0, 0.0, 0.1),
            Err(ParamError::CorrelationOutOfRange { .. })
        ));
        assert!(matches!(
            RawSvi::new(0.04, 0.4, -1.2, 0.0, 0.1),
            Err(ParamError::CorrelationOutOfRange { .. })
        ));
    }

    #[test]
    fn new_rejects_non_positive_sigma() {
        assert!(matches!(
            RawSvi::new(0.04, 0.4, 0.0, 0.0, 0.0),
            Err(ParamError::NonPositiveSigma { .. })
        ));
    }

    #[test]
    fn new_rejects_negative_min_variance() {
        // a very negative -> w_min < 0.
        assert!(matches!(
            RawSvi::new(-1.0, 0.4, 0.0, 0.0, 0.1),
            Err(ParamError::NegativeMinVariance { .. })
        ));
    }

    #[test]
    fn new_rejects_non_finite() {
        assert!(matches!(
            RawSvi::new(f64::NAN, 0.4, 0.0, 0.0, 0.1),
            Err(ParamError::NonFinite { .. })
        ));
    }

    #[test]
    fn total_variance_golden_value() {
        // rho = 0, m = 0: w(0) = a + b*sigma. w(k) symmetric.
        let svi = RawSvi::new(0.04, 0.4, 0.0, 0.0, 0.1).unwrap();
        assert!((svi.total_variance(0.0) - 0.08).abs() < 1e-15);
        assert!((svi.total_variance(0.5) - svi.total_variance(-0.5)).abs() < 1e-15);
    }

    #[test]
    fn total_variance_hand_computed() {
        // a=0.04, b=0.4, rho=-0.3, m=0.05, sigma=0.12, k=0.2.
        // u = 0.15, r = sqrt(0.0225 + 0.0144) = sqrt(0.0369).
        let svi = slice();
        let u = 0.2 - 0.05;
        let r = (u * u + 0.12 * 0.12_f64).sqrt();
        let expected = 0.04 + 0.4 * ((-0.3) * u + r);
        assert!((svi.total_variance(0.2) - expected).abs() < 1e-15);
    }

    #[test]
    fn derivative_matches_finite_difference() {
        let svi = slice();
        let h = 1e-6;
        for &k in &[-0.3, -0.1, 0.0, 0.15, 0.4] {
            let fd = (svi.total_variance(k + h) - svi.total_variance(k - h)) / (2.0 * h);
            assert!((svi.w_prime(k) - fd).abs() < 1e-6, "k = {k}");
        }
    }

    #[test]
    fn second_derivative_matches_finite_difference() {
        let svi = slice();
        let h = 1e-4;
        for &k in &[-0.3, -0.1, 0.0, 0.15, 0.4] {
            let fd = (svi.total_variance(k + h) - 2.0 * svi.total_variance(k)
                + svi.total_variance(k - h))
                / (h * h);
            assert!((svi.w_double_prime(k) - fd).abs() < 1e-4, "k = {k}");
        }
    }

    #[test]
    fn second_derivative_strictly_positive() {
        let svi = slice();
        for &k in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            assert!(svi.w_double_prime(k) > 0.0, "k = {k}");
        }
    }

    #[test]
    fn vertex_is_the_minimum() {
        let svi = slice();
        let km = svi.k_min();
        assert!(svi.w_prime(km).abs() < 1e-12);
        let wmin = svi.w_min();
        assert!((wmin - svi.total_variance(km)).abs() < 1e-12);
        // Any other point has larger w.
        assert!(svi.total_variance(km + 0.1) > wmin);
        assert!(svi.total_variance(km - 0.1) > wmin);
    }

    #[test]
    fn implied_vol_roundtrip() {
        let svi = RawSvi::new(0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
        let vol = svi.implied_vol(0.0, 1.0).unwrap();
        assert!((vol - 0.2).abs() < 1e-12);
    }

    #[test]
    fn implied_vol_rejects_bad_maturity() {
        let svi = slice();
        assert!(matches!(
            svi.implied_vol(0.0, 0.0),
            Err(ParamError::NonPositiveMaturity { .. })
        ));
    }

    #[test]
    fn atm_accessors_agree_with_evaluation() {
        let svi = slice();
        assert!((svi.atm_total_variance() - svi.total_variance(0.0)).abs() < 1e-15);
        assert!((svi.atm_skew() - svi.w_prime(0.0)).abs() < 1e-15);
        assert!((svi.atm_curvature() - svi.w_double_prime(0.0)).abs() < 1e-15);
    }

    #[test]
    fn wing_slopes_have_correct_sign() {
        let svi = slice();
        assert!(svi.left_wing_slope() <= 0.0);
        assert!(svi.right_wing_slope() >= 0.0);
        // Asymptotic check: slope of w far in the wings.
        let far = (svi.total_variance(1000.0) - svi.total_variance(999.0)) / 1.0;
        assert!((far - svi.right_wing_slope()).abs() < 1e-3);
    }

    #[test]
    fn validate_round_trips() {
        let svi = RawSvi::new_unchecked(0.04, 0.4, -0.3, 0.0, 0.1);
        assert!(svi.validate().is_ok());
        let bad = RawSvi::new_unchecked(0.04, -1.0, 0.0, 0.0, 0.1);
        assert!(bad.validate().is_err());
    }
}

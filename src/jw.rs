// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! SVI Jump-Wings parametrisation — trader-facing slice parameters.
//!
//! Raw SVI parameters have no direct financial meaning: `m` and `sigma` are
//! not quantities a trader observes. The Jump-Wings (JW) parametrisation
//! re-expresses one slice in five quantities read off the market, and is
//! maturity-dependent (it carries the maturity `t` explicitly).
//!
//! With `w := w(0)` the ATM total variance at maturity `t`:
//!
//! | JW parameter | Meaning                                                    |
//! |--------------|------------------------------------------------------------|
//! | `v_t`        | ATM variance: `v_t = w(0) / t`                             |
//! | `psi_t`      | ATM skew: `psi_t = w'(0) / (2*sqrt(w(0)))`                 |
//! | `p_t`        | Left (put) wing slope: `p_t = b*(1-rho)/sqrt(w(0))`        |
//! | `c_t`        | Right (call) wing slope: `c_t = b*(1+rho)/sqrt(w(0))`      |
//! | `v_tilde_t`  | Minimum variance: `v_tilde_t = w_min / t`                  |
//!
//! The closed-form maps between Raw and Jump-Wings live in
//! [`crate::convert`]; this module only defines the struct and its
//! accessors.
//!
//! # References
//!
//! - Gatheral, J. & Jacquier, A., "Arbitrage-free SVI volatility surfaces",
//!   *Quantitative Finance* 14(1):59-71 (2014), Section 3.2.

use crate::errors::ParamError;

/// An SVI slice in the Jump-Wings parametrisation, tagged with its maturity.
///
/// All five quantities are maturity-specific; `t` is stored so the slice can
/// be converted back to the raw parametrisation unambiguously
/// (see [`crate::convert`]).
///
/// # Examples
///
/// ```
/// use regit_svi::jw::SviJw;
///
/// let jw = SviJw::new(0.04, -0.1, 0.3, 0.25, 0.035, 1.0).unwrap();
/// assert!((jw.v_t - 0.04).abs() < 1e-15);
/// assert!((jw.t - 1.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SviJw {
    /// ATM variance `v_t = w(0) / t`.
    pub v_t: f64,
    /// ATM skew `psi_t = w'(0) / (2*sqrt(w(0)))`.
    pub psi_t: f64,
    /// Left (put) wing slope `p_t = b*(1-rho)/sqrt(w(0))`.
    pub p_t: f64,
    /// Right (call) wing slope `c_t = b*(1+rho)/sqrt(w(0))`.
    pub c_t: f64,
    /// Minimum variance `v_tilde_t = w_min / t`.
    pub v_tilde_t: f64,
    /// Maturity `t > 0` the slice belongs to.
    pub t: f64,
}

impl SviJw {
    /// Creates a validated Jump-Wings slice.
    ///
    /// Validation enforces what a JW tuple must satisfy independently of any
    /// raw pre-image: all values finite, `v_t > 0` and `v_tilde_t >= 0`
    /// (variances), `p_t >= 0` and `c_t >= 0` (wing slopes are non-negative),
    /// and `t > 0`. The further existence condition `|beta| <= 1` is checked
    /// only when converting to the raw parametrisation — see
    /// [`crate::convert::jw_to_raw`].
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if any field is `NaN` or infinite.
    /// - [`ParamError::NonPositiveMaturity`] if `t <= 0`.
    /// - [`ParamError::NegativeTotalVariance`] if `v_t <= 0` or `v_tilde_t < 0`.
    /// - [`ParamError::NegativeSlope`] if `p_t < 0` or `c_t < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::jw::SviJw;
    ///
    /// assert!(SviJw::new(0.04, -0.1, 0.3, 0.25, 0.035, 1.0).is_ok());
    /// assert!(SviJw::new(-0.04, 0.0, 0.3, 0.25, 0.0, 1.0).is_err());
    /// ```
    pub fn new(
        v_t: f64,
        psi_t: f64,
        p_t: f64,
        c_t: f64,
        v_tilde_t: f64,
        t: f64,
    ) -> Result<Self, ParamError> {
        for (name, value) in [
            ("v_t", v_t),
            ("psi_t", psi_t),
            ("p_t", p_t),
            ("c_t", c_t),
            ("v_tilde_t", v_tilde_t),
            ("t", t),
        ] {
            if !value.is_finite() {
                return Err(ParamError::NonFinite { name });
            }
        }
        if t <= 0.0 {
            return Err(ParamError::NonPositiveMaturity { t });
        }
        if v_t <= 0.0 {
            return Err(ParamError::NegativeTotalVariance { w: v_t });
        }
        if v_tilde_t < 0.0 {
            return Err(ParamError::NegativeTotalVariance { w: v_tilde_t });
        }
        if p_t < 0.0 {
            return Err(ParamError::NegativeSlope { b: p_t });
        }
        if c_t < 0.0 {
            return Err(ParamError::NegativeSlope { b: c_t });
        }
        Ok(Self {
            v_t,
            psi_t,
            p_t,
            c_t,
            v_tilde_t,
            t,
        })
    }

    /// Creates a Jump-Wings slice without validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::jw::SviJw;
    ///
    /// let jw = SviJw::new_unchecked(0.04, -0.1, 0.3, 0.25, 0.035, 1.0);
    /// assert!((jw.v_t - 0.04).abs() < 1e-15);
    /// ```
    #[must_use]
    pub const fn new_unchecked(
        v_t: f64,
        psi_t: f64,
        p_t: f64,
        c_t: f64,
        v_tilde_t: f64,
        t: f64,
    ) -> Self {
        Self {
            v_t,
            psi_t,
            p_t,
            c_t,
            v_tilde_t,
            t,
        }
    }

    /// Re-checks the Jump-Wings domain for an existing slice.
    ///
    /// # Errors
    ///
    /// Returns the same [`ParamError`] variants as [`SviJw::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::jw::SviJw;
    ///
    /// let jw = SviJw::new_unchecked(0.04, -0.1, 0.3, 0.25, 0.035, 1.0);
    /// assert!(jw.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), ParamError> {
        Self::new(
            self.v_t,
            self.psi_t,
            self.p_t,
            self.c_t,
            self.v_tilde_t,
            self.t,
        )
        .map(|_| ())
    }

    /// ATM total variance `w(0) = v_t * t`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::jw::SviJw;
    ///
    /// let jw = SviJw::new(0.04, -0.1, 0.3, 0.25, 0.035, 2.0).unwrap();
    /// assert!((jw.atm_total_variance() - 0.08).abs() < 1e-15);
    /// ```
    #[must_use]
    #[inline]
    pub fn atm_total_variance(&self) -> f64 {
        self.v_t * self.t
    }

    /// Minimum total variance `w_min = v_tilde_t * t`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::jw::SviJw;
    ///
    /// let jw = SviJw::new(0.04, -0.1, 0.3, 0.25, 0.035, 2.0).unwrap();
    /// assert!((jw.min_total_variance() - 0.07).abs() < 1e-15);
    /// ```
    #[must_use]
    #[inline]
    pub fn min_total_variance(&self) -> f64 {
        self.v_tilde_t * self.t
    }

    /// ATM implied volatility `sqrt(v_t)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::jw::SviJw;
    ///
    /// let jw = SviJw::new(0.04, -0.1, 0.3, 0.25, 0.035, 1.0).unwrap();
    /// assert!((jw.atm_vol() - 0.2).abs() < 1e-12);
    /// ```
    #[must_use]
    #[inline]
    pub fn atm_vol(&self) -> f64 {
        self.v_t.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_valid_tuple() {
        assert!(SviJw::new(0.04, -0.1, 0.3, 0.25, 0.035, 1.0).is_ok());
    }

    #[test]
    fn new_rejects_non_positive_maturity() {
        assert!(matches!(
            SviJw::new(0.04, -0.1, 0.3, 0.25, 0.035, 0.0),
            Err(ParamError::NonPositiveMaturity { .. })
        ));
    }

    #[test]
    fn new_rejects_non_positive_v_t() {
        assert!(matches!(
            SviJw::new(0.0, -0.1, 0.3, 0.25, 0.0, 1.0),
            Err(ParamError::NegativeTotalVariance { .. })
        ));
    }

    #[test]
    fn new_rejects_negative_v_tilde() {
        assert!(matches!(
            SviJw::new(0.04, -0.1, 0.3, 0.25, -0.01, 1.0),
            Err(ParamError::NegativeTotalVariance { .. })
        ));
    }

    #[test]
    fn new_rejects_negative_wing_slopes() {
        assert!(matches!(
            SviJw::new(0.04, -0.1, -0.1, 0.25, 0.035, 1.0),
            Err(ParamError::NegativeSlope { .. })
        ));
        assert!(matches!(
            SviJw::new(0.04, -0.1, 0.3, -0.1, 0.035, 1.0),
            Err(ParamError::NegativeSlope { .. })
        ));
    }

    #[test]
    fn new_rejects_non_finite() {
        assert!(matches!(
            SviJw::new(f64::NAN, -0.1, 0.3, 0.25, 0.035, 1.0),
            Err(ParamError::NonFinite { .. })
        ));
    }

    #[test]
    fn accessors() {
        let jw = SviJw::new(0.04, -0.1, 0.3, 0.25, 0.035, 2.0).unwrap();
        assert!((jw.atm_total_variance() - 0.08).abs() < 1e-15);
        assert!((jw.min_total_variance() - 0.07).abs() < 1e-15);
        assert!((jw.atm_vol() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn validate_round_trips() {
        let jw = SviJw::new_unchecked(0.04, -0.1, 0.3, 0.25, 0.035, 1.0);
        assert!(jw.validate().is_ok());
        let bad = SviJw::new_unchecked(-1.0, 0.0, 0.3, 0.25, 0.0, 1.0);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn is_copy() {
        let jw = SviJw::new(0.04, -0.1, 0.3, 0.25, 0.035, 1.0).unwrap();
        let copy = jw;
        assert_eq!(jw, copy);
    }
}

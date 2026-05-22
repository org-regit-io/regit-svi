// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Core market types: log-moneyness, total implied variance, and quotes.
//!
//! SVI parametrises one maturity slice at a time. Fix a time to expiry
//! `T > 0` and a forward price `F`. For a strike `K` the **log-moneyness**
//! is `k = ln(K / F)`, with `k = 0` at-the-money-forward.
//!
//! SVI does not parametrise Black implied volatility directly — it
//! parametrises the **total implied variance**:
//!
//! ```text
//! w(k) = sigma_BS(k)^2 * T
//! ```
//!
//! `w` is the natural object: it is additive in maturity for a flat surface,
//! the no-arbitrage conditions take their simplest form in `w`, and `w >= 0`
//! is the only domain requirement. Implied volatility is recovered by
//! `sigma_BS(k) = sqrt(w(k) / T)`.
//!
//! # References
//!
//! - Gatheral, J., *The Volatility Surface: A Practitioner's Guide*,
//!   Wiley (2006), Chapter 3.

use crate::errors::ParamError;

/// A single market quote: a log-moneyness, an observed total implied
/// variance, and a non-negative fitting weight.
///
/// A **slice** is a set of quotes sharing one maturity. The weight is any
/// non-negative number expressing the relative trust placed in the quote
/// during calibration — common choices are option vega or the inverse of the
/// bid-ask spread. A weight of `0.0` excludes the quote from the fit.
///
/// # Invariants
///
/// Constructed through [`Quote::new`] (checked) or [`Quote::new_unchecked`]
/// (caller-asserted), a `Quote` always satisfies `w >= 0`, `weight >= 0`, and
/// all three fields finite.
///
/// # Examples
///
/// ```
/// use regit_svi::types::Quote;
///
/// let q = Quote::new(-0.10, 0.0432, 1.0).unwrap();
/// assert!((q.k + 0.10).abs() < 1e-15);
/// assert!((q.w - 0.0432).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quote {
    /// Log-moneyness `k = ln(K / F)`.
    pub k: f64,
    /// Observed total implied variance `w = sigma_BS^2 * T`.
    pub w: f64,
    /// Non-negative fitting weight (e.g. vega or inverse bid-ask spread).
    pub weight: f64,
}

impl Quote {
    /// Creates a validated market quote.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if any field is `NaN` or infinite.
    /// - [`ParamError::NegativeTotalVariance`] if `w < 0`.
    /// - [`ParamError::NegativeWeight`] if `weight < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::types::Quote;
    /// use regit_svi::errors::ParamError;
    ///
    /// assert!(Quote::new(0.0, 0.04, 1.0).is_ok());
    /// assert_eq!(
    ///     Quote::new(0.0, -0.04, 1.0),
    ///     Err(ParamError::NegativeTotalVariance { w: -0.04 }),
    /// );
    /// ```
    pub fn new(k: f64, w: f64, weight: f64) -> Result<Self, ParamError> {
        if !k.is_finite() {
            return Err(ParamError::NonFinite { name: "k" });
        }
        if !w.is_finite() {
            return Err(ParamError::NonFinite { name: "w" });
        }
        if !weight.is_finite() {
            return Err(ParamError::NonFinite { name: "weight" });
        }
        if w < 0.0 {
            return Err(ParamError::NegativeTotalVariance { w });
        }
        if weight < 0.0 {
            return Err(ParamError::NegativeWeight { weight });
        }
        Ok(Self { k, w, weight })
    }

    /// Creates a quote without validation.
    ///
    /// The caller asserts `w >= 0`, `weight >= 0`, and all fields finite.
    /// Used on hot paths where the inputs are already known to be valid.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::types::Quote;
    ///
    /// let q = Quote::new_unchecked(0.0, 0.04, 1.0);
    /// assert!((q.w - 0.04).abs() < 1e-15);
    /// ```
    #[must_use]
    pub const fn new_unchecked(k: f64, w: f64, weight: f64) -> Self {
        Self { k, w, weight }
    }

    /// Returns the Black implied volatility implied by this quote at maturity
    /// `t`, i.e. `sqrt(w / t)`.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NonPositiveMaturity`] if `t <= 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::types::Quote;
    ///
    /// let q = Quote::new(0.0, 0.04, 1.0).unwrap();
    /// let vol = q.implied_vol(1.0).unwrap();
    /// assert!((vol - 0.20).abs() < 1e-12);
    /// ```
    pub fn implied_vol(&self, t: f64) -> Result<f64, ParamError> {
        if t <= 0.0 || !t.is_finite() {
            return Err(ParamError::NonPositiveMaturity { t });
        }
        Ok((self.w / t).sqrt())
    }
}

/// Builds a slice of quotes from `(k, w, weight)` triples, validating each.
///
/// # Errors
///
/// Propagates the first [`ParamError`] from [`Quote::new`].
///
/// # Examples
///
/// ```
/// use regit_svi::types::quotes_from_triples;
///
/// let slice = quotes_from_triples(&[
///     (-0.10, 0.0432, 1.0),
///     ( 0.00, 0.0400, 1.0),
///     ( 0.10, 0.0420, 1.0),
/// ]).unwrap();
/// assert_eq!(slice.len(), 3);
/// ```
pub fn quotes_from_triples(triples: &[(f64, f64, f64)]) -> Result<Vec<Quote>, ParamError> {
    triples
        .iter()
        .map(|&(k, w, weight)| Quote::new(k, w, weight))
        .collect()
}

/// Converts an implied volatility to a total implied variance: `sigma^2 * t`.
///
/// # Examples
///
/// ```
/// use regit_svi::types::total_variance_from_vol;
///
/// let w = total_variance_from_vol(0.20, 1.0);
/// assert!((w - 0.04).abs() < 1e-15);
/// ```
#[must_use]
#[inline]
pub fn total_variance_from_vol(vol: f64, t: f64) -> f64 {
    vol * vol * t
}

/// Converts a log-moneyness `k = ln(K / F)` from a strike and forward.
///
/// # Examples
///
/// ```
/// use regit_svi::types::log_moneyness;
///
/// let k = log_moneyness(110.0, 100.0);
/// assert!((k - (110.0_f64 / 100.0).ln()).abs() < 1e-15);
/// ```
#[must_use]
#[inline]
pub fn log_moneyness(strike: f64, forward: f64) -> f64 {
    (strike / forward).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::ParamError;

    #[test]
    fn quote_new_valid() {
        let q = Quote::new(-0.1, 0.0432, 2.0).unwrap();
        assert!((q.k + 0.1).abs() < 1e-15);
        assert!((q.w - 0.0432).abs() < 1e-15);
        assert!((q.weight - 2.0).abs() < 1e-15);
    }

    #[test]
    fn quote_new_rejects_negative_variance() {
        assert_eq!(
            Quote::new(0.0, -0.01, 1.0),
            Err(ParamError::NegativeTotalVariance { w: -0.01 })
        );
    }

    #[test]
    fn quote_new_rejects_negative_weight() {
        assert_eq!(
            Quote::new(0.0, 0.04, -1.0),
            Err(ParamError::NegativeWeight { weight: -1.0 })
        );
    }

    #[test]
    fn quote_new_rejects_non_finite() {
        assert_eq!(
            Quote::new(f64::NAN, 0.04, 1.0),
            Err(ParamError::NonFinite { name: "k" })
        );
        assert_eq!(
            Quote::new(0.0, f64::INFINITY, 1.0),
            Err(ParamError::NonFinite { name: "w" })
        );
        assert_eq!(
            Quote::new(0.0, 0.04, f64::NAN),
            Err(ParamError::NonFinite { name: "weight" })
        );
    }

    #[test]
    fn quote_new_unchecked() {
        let q = Quote::new_unchecked(0.1, 0.05, 0.5);
        assert!((q.k - 0.1).abs() < 1e-15);
    }

    #[test]
    fn quote_implied_vol_roundtrip() {
        let q = Quote::new(0.0, 0.09, 1.0).unwrap();
        let vol = q.implied_vol(1.0).unwrap();
        assert!((vol - 0.30).abs() < 1e-12);
    }

    #[test]
    fn quote_implied_vol_rejects_bad_maturity() {
        let q = Quote::new(0.0, 0.04, 1.0).unwrap();
        assert!(matches!(
            q.implied_vol(0.0),
            Err(ParamError::NonPositiveMaturity { .. })
        ));
        assert!(matches!(
            q.implied_vol(-1.0),
            Err(ParamError::NonPositiveMaturity { .. })
        ));
    }

    #[test]
    fn quotes_from_triples_builds_slice() {
        let slice = quotes_from_triples(&[(-0.1, 0.05, 1.0), (0.1, 0.05, 1.0)]).unwrap();
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn quotes_from_triples_propagates_error() {
        let bad = quotes_from_triples(&[(0.0, -1.0, 1.0)]);
        assert!(bad.is_err());
    }

    #[test]
    fn total_variance_from_vol_works() {
        assert!((total_variance_from_vol(0.20, 2.0) - 0.08).abs() < 1e-15);
    }

    #[test]
    fn log_moneyness_atm_is_zero() {
        assert!(log_moneyness(100.0, 100.0).abs() < 1e-15);
    }

    #[test]
    fn quote_is_copy() {
        let q = Quote::new(0.0, 0.04, 1.0).unwrap();
        let copy = q;
        assert_eq!(q, copy);
    }
}

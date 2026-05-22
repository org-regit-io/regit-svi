// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Multi-slice volatility surface assembly and interpolation.
//!
//! A surface is an ordered set of calibrated slices at maturities
//! `t_1 < ... < t_n`. To evaluate total variance at an arbitrary `(k, T)`:
//!
//! - **Maturity inside the grid.** Locate the bracketing slices
//!   `t_j <= T < t_{j+1}` and interpolate linearly in total variance along
//!   constant `k`:
//!
//!   ```text
//!   w(k, T) = ( (t_{j+1} - T)*w(k, t_j) + (T - t_j)*w(k, t_{j+1}) )
//!             / (t_{j+1} - t_j)
//!   ```
//!
//!   Linear interpolation in `w` is monotone in `T`, so it introduces no
//!   calendar-spread arbitrage provided the bracketing slices are themselves
//!   ordered.
//!
//! - **Maturity outside the grid.** Total variance is extrapolated flat in
//!   implied volatility (constant `sigma_BS` beyond the first / last slice),
//!   the conservative market default.
//!
//! For an SSVI-backed surface, evaluation is direct from the closed form at
//! the interpolated `theta_T`, and no per-`k` interpolation is needed.
//!
//! # References
//!
//! - Gatheral, J., *The Volatility Surface: A Practitioner's Guide*,
//!   Wiley (2006), Chapter 3.

use crate::arbitrage::calendar_scan;
use crate::errors::ParamError;
use crate::raw::RawSvi;
use crate::ssvi::Ssvi;

/// The backing representation of a [`Surface`].
#[derive(Debug, Clone, PartialEq)]
enum Backing {
    /// A set of raw slices, each tagged with its maturity (ascending order).
    Slices(Vec<(f64, RawSvi)>),
    /// An SSVI surface plus its `(maturity, theta)` term structure.
    Ssvi {
        /// The SSVI parametrisation.
        ssvi: Ssvi,
        /// `(maturity, theta)` knots, ascending in maturity.
        term: Vec<(f64, f64)>,
    },
}

/// An arbitrage-checked volatility surface.
///
/// Built from either a set of calibrated raw slices ([`Surface::from_slices`])
/// or an SSVI parametrisation ([`Surface::from_ssvi`]). Evaluation at any
/// `(k, T)` returns total variance ([`Surface::total_variance`]) or implied
/// volatility ([`Surface::implied_vol`]).
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::surface::Surface;
///
/// let s1 = RawSvi::new(0.02, 0.3, -0.2, 0.0, 0.1).unwrap();
/// let s2 = RawSvi::new(0.05, 0.3, -0.2, 0.0, 0.1).unwrap();
/// let surface = Surface::from_slices(vec![(0.5, s1), (1.5, s2)]).unwrap();
/// // Total variance at an interpolated maturity.
/// let w = surface.total_variance(0.0, 1.0);
/// assert!(w > s1.total_variance(0.0) && w < s2.total_variance(0.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Surface {
    backing: Backing,
}

impl Surface {
    /// Builds a surface from `(maturity, slice)` pairs.
    ///
    /// The pairs are sorted into ascending maturity order. Maturities must be
    /// strictly positive and distinct.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if a maturity is not finite.
    /// - [`ParamError::NonPositiveMaturity`] if a maturity is `<= 0` or two
    ///   maturities coincide.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    /// use regit_svi::surface::Surface;
    ///
    /// let s = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
    /// assert!(Surface::from_slices(vec![(1.0, s)]).is_ok());
    /// assert!(Surface::from_slices(vec![(0.0, s)]).is_err());
    /// ```
    pub fn from_slices(mut slices: Vec<(f64, RawSvi)>) -> Result<Self, ParamError> {
        if slices.is_empty() {
            return Err(ParamError::NonPositiveMaturity { t: 0.0 });
        }
        for &(t, _) in &slices {
            if !t.is_finite() {
                return Err(ParamError::NonFinite { name: "maturity" });
            }
            if t <= 0.0 {
                return Err(ParamError::NonPositiveMaturity { t });
            }
        }
        slices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        for pair in slices.windows(2) {
            if (pair[1].0 - pair[0].0).abs() < 1e-12 {
                return Err(ParamError::NonPositiveMaturity { t: pair[1].0 });
            }
        }
        Ok(Self {
            backing: Backing::Slices(slices),
        })
    }

    /// Builds a surface from an SSVI parametrisation and its `(maturity,
    /// theta)` term structure.
    ///
    /// The term structure is sorted into ascending maturity order; `theta`
    /// must be non-decreasing (a non-decreasing ATM term structure is one of
    /// the SSVI no-calendar conditions).
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonPositiveMaturity`] if the term structure is empty or
    ///   contains a non-positive / duplicate maturity.
    /// - [`ParamError::NonPositiveTheta`] if a `theta` is non-positive.
    /// - [`ParamError::NonFinite`] if a knot is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::{Phi, Ssvi};
    /// use regit_svi::surface::Surface;
    ///
    /// let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
    /// let surface = Surface::from_ssvi(ssvi, vec![(0.5, 0.02), (1.0, 0.04)]).unwrap();
    /// assert!(surface.total_variance(0.0, 0.75) > 0.0);
    /// ```
    pub fn from_ssvi(ssvi: Ssvi, mut term: Vec<(f64, f64)>) -> Result<Self, ParamError> {
        if term.is_empty() {
            return Err(ParamError::NonPositiveMaturity { t: 0.0 });
        }
        for &(t, theta) in &term {
            if !t.is_finite() || !theta.is_finite() {
                return Err(ParamError::NonFinite {
                    name: "term structure",
                });
            }
            if t <= 0.0 {
                return Err(ParamError::NonPositiveMaturity { t });
            }
            if theta <= 0.0 {
                return Err(ParamError::NonPositiveTheta { theta });
            }
        }
        term.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        for pair in term.windows(2) {
            if (pair[1].0 - pair[0].0).abs() < 1e-12 {
                return Err(ParamError::NonPositiveMaturity { t: pair[1].0 });
            }
        }
        Ok(Self {
            backing: Backing::Ssvi { ssvi, term },
        })
    }

    /// The number of maturity knots in the surface.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    /// use regit_svi::surface::Surface;
    ///
    /// let s = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
    /// let surface = Surface::from_slices(vec![(0.5, s), (1.0, s)]).unwrap();
    /// assert_eq!(surface.len(), 2);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.backing {
            Backing::Slices(s) => s.len(),
            Backing::Ssvi { term, .. } => term.len(),
        }
    }

    /// Returns `true` if the surface has no maturity knots.
    ///
    /// A [`Surface`] is always constructed with at least one knot, so this
    /// returns `false` for every value built through the public constructors.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    /// use regit_svi::surface::Surface;
    ///
    /// let s = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
    /// let surface = Surface::from_slices(vec![(1.0, s)]).unwrap();
    /// assert!(!surface.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total implied variance `w(k, T)` at log-moneyness `k` and maturity `T`.
    ///
    /// Inside the maturity grid the value is linearly interpolated in total
    /// variance; outside it the implied volatility is held flat (constant
    /// `sigma_BS`). An SSVI-backed surface evaluates the closed form at the
    /// interpolated `theta`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    /// use regit_svi::surface::Surface;
    ///
    /// let s1 = RawSvi::new(0.02, 0.3, -0.2, 0.0, 0.1).unwrap();
    /// let s2 = RawSvi::new(0.05, 0.3, -0.2, 0.0, 0.1).unwrap();
    /// let surface = Surface::from_slices(vec![(0.5, s1), (1.5, s2)]).unwrap();
    /// // Midpoint maturity -> midpoint total variance at constant k.
    /// let mid = surface.total_variance(0.0, 1.0);
    /// let expect = 0.5 * (s1.total_variance(0.0) + s2.total_variance(0.0));
    /// assert!((mid - expect).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn total_variance(&self, k: f64, t: f64) -> f64 {
        match &self.backing {
            Backing::Slices(slices) => Self::total_variance_slices(slices, k, t),
            Backing::Ssvi { ssvi, term } => {
                let theta = interpolate_theta(term, t);
                ssvi.total_variance(k, theta)
            }
        }
    }

    /// Black implied volatility `sigma_BS(k, T) = sqrt(w(k, T) / T)`.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NonPositiveMaturity`] if `T <= 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    /// use regit_svi::surface::Surface;
    ///
    /// let s = RawSvi::new(0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
    /// let surface = Surface::from_slices(vec![(1.0, s)]).unwrap();
    /// // Flat w = 0.04 at t = 1 -> vol = 0.2.
    /// assert!((surface.implied_vol(0.0, 1.0).unwrap() - 0.2).abs() < 1e-12);
    /// ```
    pub fn implied_vol(&self, k: f64, t: f64) -> Result<f64, ParamError> {
        if t <= 0.0 || !t.is_finite() {
            return Err(ParamError::NonPositiveMaturity { t });
        }
        Ok((self.total_variance(k, t) / t).sqrt())
    }

    /// Checks the surface for calendar-spread arbitrage across adjacent
    /// maturity knots.
    ///
    /// For a slice-backed surface, runs the [`calendar_scan`] on every
    /// adjacent pair over `[k_lo, k_hi]`; the surface is calendar-free if
    /// every pair is. For an SSVI-backed surface, the closed-form Theorem 4.1
    /// conditions are used.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    /// use regit_svi::surface::Surface;
    ///
    /// let s1 = RawSvi::new(0.02, 0.3, -0.2, 0.0, 0.1).unwrap();
    /// let s2 = RawSvi::new(0.05, 0.3, -0.2, 0.0, 0.1).unwrap();
    /// let surface = Surface::from_slices(vec![(0.5, s1), (1.5, s2)]).unwrap();
    /// assert!(surface.is_calendar_free(-0.5, 0.5));
    /// ```
    #[must_use]
    pub fn is_calendar_free(&self, k_lo: f64, k_hi: f64) -> bool {
        match &self.backing {
            Backing::Slices(slices) => slices
                .windows(2)
                .all(|p| calendar_scan(&p[0].1, &p[1].1, k_lo, k_hi).is_free),
            Backing::Ssvi { ssvi, term } => {
                let thetas: Vec<f64> = term.iter().map(|&(_, theta)| theta).collect();
                ssvi.is_calendar_free(&thetas)
            }
        }
    }

    /// Evaluates a slice-backed surface at `(k, t)` with maturity
    /// interpolation and flat-vol extrapolation.
    fn total_variance_slices(slices: &[(f64, RawSvi)], k: f64, t: f64) -> f64 {
        let n = slices.len();
        let (t0, first) = slices[0];
        let (tn, last) = slices[n - 1];

        if t <= t0 {
            // Flat implied volatility below the first maturity:
            // w(k, t) = (w(k, t0) / t0) * t.
            return (first.total_variance(k) / t0) * t.max(0.0);
        }
        if t >= tn {
            // Flat implied volatility above the last maturity.
            return (last.total_variance(k) / tn) * t;
        }

        // Bracketing slices t_j <= t < t_{j+1}; linear interpolation in w.
        for pair in slices.windows(2) {
            let (tj, sj) = pair[0];
            let (tj1, sj1) = pair[1];
            if t >= tj && t <= tj1 {
                let wj = sj.total_variance(k);
                let wj1 = sj1.total_variance(k);
                let frac = (t - tj) / (tj1 - tj);
                return wj + frac * (wj1 - wj);
            }
        }
        // Unreachable for a sorted, bracketing grid; fall back to the last.
        last.total_variance(k)
    }
}

/// Interpolates the `theta` term structure at maturity `t`.
///
/// Linear interpolation between adjacent `(maturity, theta)` knots, with flat
/// extrapolation of `theta / t` (constant ATM implied variance) outside the
/// grid.
fn interpolate_theta(term: &[(f64, f64)], t: f64) -> f64 {
    let n = term.len();
    let (t_first, theta_first) = term[0];
    let (t_last, theta_last) = term[n - 1];

    if t <= t_first {
        return (theta_first / t_first) * t.max(0.0);
    }
    if t >= t_last {
        return (theta_last / t_last) * t;
    }
    for pair in term.windows(2) {
        let (t_lo, theta_lo) = pair[0];
        let (t_hi, theta_hi) = pair[1];
        if t >= t_lo && t <= t_hi {
            let frac = (t - t_lo) / (t_hi - t_lo);
            return theta_lo + frac * (theta_hi - theta_lo);
        }
    }
    theta_last
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssvi::Phi;

    #[test]
    fn from_slices_sorts_and_validates() {
        let s = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(2.0, s), (0.5, s), (1.0, s)]).unwrap();
        assert_eq!(surface.len(), 3);
    }

    #[test]
    fn from_slices_rejects_bad_maturity() {
        let s = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
        assert!(Surface::from_slices(vec![(0.0, s)]).is_err());
        assert!(Surface::from_slices(vec![(1.0, s), (1.0, s)]).is_err());
        assert!(Surface::from_slices(vec![]).is_err());
    }

    #[test]
    fn interpolation_is_linear_in_w() {
        let s1 = RawSvi::new(0.02, 0.3, -0.2, 0.0, 0.1).unwrap();
        let s2 = RawSvi::new(0.06, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(1.0, s1), (3.0, s2)]).unwrap();
        // At t = 2 (midpoint), w is the average at every k.
        for &k in &[-0.3, 0.0, 0.3] {
            let mid = surface.total_variance(k, 2.0);
            let expect = 0.5 * (s1.total_variance(k) + s2.total_variance(k));
            assert!((mid - expect).abs() < 1e-12, "k = {k}");
        }
    }

    #[test]
    fn interpolation_recovers_knot_values() {
        let s1 = RawSvi::new(0.02, 0.3, -0.2, 0.0, 0.1).unwrap();
        let s2 = RawSvi::new(0.06, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(1.0, s1), (3.0, s2)]).unwrap();
        assert!((surface.total_variance(0.1, 1.0) - s1.total_variance(0.1)).abs() < 1e-12);
        assert!((surface.total_variance(0.1, 3.0) - s2.total_variance(0.1)).abs() < 1e-12);
    }

    #[test]
    fn extrapolation_is_flat_in_vol() {
        let s = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(1.0, s)]).unwrap();
        // Below: w(k, 0.5) = w(k, 1) * 0.5 (constant vol).
        let w_half = surface.total_variance(0.0, 0.5);
        assert!((w_half - s.total_variance(0.0) * 0.5).abs() < 1e-12);
        // Above: w(k, 2) = w(k, 1) * 2.
        let w_double = surface.total_variance(0.0, 2.0);
        assert!((w_double - s.total_variance(0.0) * 2.0).abs() < 1e-12);
        // Implied vol is constant across extrapolated maturities.
        let v05 = surface.implied_vol(0.0, 0.5).unwrap();
        let v20 = surface.implied_vol(0.0, 2.0).unwrap();
        assert!((v05 - v20).abs() < 1e-12);
    }

    #[test]
    fn implied_vol_rejects_bad_maturity() {
        let s = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(1.0, s)]).unwrap();
        assert!(surface.implied_vol(0.0, 0.0).is_err());
    }

    #[test]
    fn slice_surface_calendar_check() {
        let early = RawSvi::new(0.02, 0.3, -0.2, 0.0, 0.1).unwrap();
        let late = RawSvi::new(0.06, 0.3, -0.2, 0.0, 0.1).unwrap();
        let ok = Surface::from_slices(vec![(0.5, early), (1.5, late)]).unwrap();
        assert!(ok.is_calendar_free(-0.5, 0.5));
        // Reversed: the later slice has lower variance -> arbitrage.
        let bad = Surface::from_slices(vec![(0.5, late), (1.5, early)]).unwrap();
        assert!(!bad.is_calendar_free(-0.5, 0.5));
    }

    #[test]
    fn ssvi_surface_evaluates_and_interpolates() {
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        let surface =
            Surface::from_ssvi(ssvi, vec![(0.5, 0.02), (1.0, 0.04), (2.0, 0.08)]).unwrap();
        // At a knot maturity, theta is exact.
        let w_knot = surface.total_variance(0.1, 1.0);
        assert!((w_knot - ssvi.total_variance(0.1, 0.04)).abs() < 1e-12);
        // Interpolated maturity is between bracketing values.
        let w_mid = surface.total_variance(0.0, 1.5);
        assert!(w_mid > ssvi.total_variance(0.0, 0.04));
        assert!(w_mid < ssvi.total_variance(0.0, 0.08));
    }

    #[test]
    fn ssvi_surface_calendar_check() {
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        let surface =
            Surface::from_ssvi(ssvi, vec![(0.5, 0.02), (1.0, 0.04), (2.0, 0.08)]).unwrap();
        assert!(surface.is_calendar_free(-0.5, 0.5));
    }

    #[test]
    fn ssvi_surface_rejects_bad_term() {
        let ssvi = Ssvi::new(-0.3, Phi::heston(1.0).unwrap()).unwrap();
        assert!(Surface::from_ssvi(ssvi, vec![]).is_err());
        assert!(Surface::from_ssvi(ssvi, vec![(1.0, 0.0)]).is_err());
        assert!(Surface::from_ssvi(ssvi, vec![(0.0, 0.04)]).is_err());
    }

    #[test]
    fn is_empty_is_false_for_built_surface() {
        let s = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(1.0, s)]).unwrap();
        assert!(!surface.is_empty());
    }
}

// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Static-arbitrage checks: butterfly (`g(k) >= 0`) and calendar-spread.
//!
//! # Butterfly arbitrage and the g function
//!
//! A slice admits butterfly arbitrage when the risk-neutral density it
//! implies is negative somewhere — a butterfly spread with negative cost.
//! Define
//!
//! ```text
//! g(k) = ( 1 - k*w'(k) / (2*w(k)) )^2
//!      - ( w'(k) / 2 )^2 * ( 1/w(k) + 1/4 )
//!      + w''(k) / 2
//! ```
//!
//! A slice is free of butterfly arbitrage iff `g(k) >= 0` for all `k` and
//! call prices vanish at infinite strike. The latter is, for raw SVI,
//! equivalent to the wing bound `b*(1 + |rho|) <= 2` (Lee 2004): implied
//! total variance cannot grow faster than `2|k|`.
//!
//! `g(k)` is evaluated in closed form from `w, w', w''`. The check scans `g`
//! on a dense grid spanning the quoted range plus a wing margin, and refines
//! any sign change with a Brent root-find to report the arbitrage interval.
//!
//! # Calendar-spread arbitrage
//!
//! Two slices at maturities `t_1 < t_2` admit calendar-spread arbitrage when
//! their total-variance curves cross. Absence is the pointwise monotonicity
//! `w(k, t_1) <= w(k, t_2)` for all `k`. The difference
//! `D(k) = w(k, t_2) - w(k, t_1)` is scanned for negativity, with Brent
//! refinement of any crossing.
//!
//! For SSVI, the closed-form Theorem 4.1 / 4.2 conditions of
//! [`crate::ssvi`] are exact and are used instead of the grid scan.
//!
//! # References
//!
//! - Gatheral, J. & Jacquier, A., "Arbitrage-free SVI volatility surfaces",
//!   *Quantitative Finance* 14(1):59-71 (2014), Section 2.
//! - Roper, M., "Arbitrage free implied volatility surfaces", preprint,
//!   University of Sydney (2010).
//! - Lee, R. W., "The moment formula for implied volatility at extreme
//!   strikes", *Mathematical Finance* 14(3):469-480 (2004).

use crate::math::{brent_root, index_to_f64};
use crate::raw::RawSvi;

/// Number of grid points used by the butterfly and calendar scans.
const SCAN_POINTS: usize = 401;
/// Wing margin added on each side of the quoted range when scanning.
const SCAN_MARGIN: f64 = 1.0;
/// Brent tolerance for refining a reported arbitrage boundary.
const REFINE_TOL: f64 = 1e-10;
/// Brent iteration cap for boundary refinement.
const REFINE_MAX_ITER: usize = 200;

/// The butterfly function `g(k)` for a raw SVI slice (MATH.md §7).
///
/// `g` is the (normalised) risk-neutral density in log-strike space:
/// `g(k) >= 0` everywhere is exactly the no-butterfly-arbitrage condition.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::arbitrage::g;
///
/// // A gently curved slice has positive g near the money.
/// let svi = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
/// assert!(g(&svi, 0.0) > 0.0);
/// ```
#[must_use]
pub fn g(svi: &RawSvi, k: f64) -> f64 {
    let w = svi.total_variance(k);
    let wp = svi.w_prime(k);
    let wpp = svi.w_double_prime(k);

    // term1 = (1 - k*w'/(2*w))^2
    let t1 = {
        let inner = 1.0 - k * wp / (2.0 * w);
        inner * inner
    };
    // term2 = (w'/2)^2 * (1/w + 1/4)
    let t2 = {
        let half_wp = wp / 2.0;
        half_wp * half_wp * (1.0 / w + 0.25)
    };
    // term3 = w''/2
    let t3 = wpp / 2.0;

    t1 - t2 + t3
}

/// The result of a butterfly-arbitrage scan over a raw SVI slice.
///
/// When `is_free` is `false`, [`Self::worst_k`] reports the log-moneyness of
/// the deepest density violation and [`Self::min_g`] its value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButterflyReport {
    /// `true` if `g(k) >= 0` over the scanned domain and the wing bound holds.
    pub is_free: bool,
    /// The smallest value of `g` observed on the scan grid.
    pub min_g: f64,
    /// The log-moneyness at which [`Self::min_g`] was observed.
    pub worst_k: f64,
    /// `true` if the wing bound `b*(1 + |rho|) <= 2` is satisfied.
    pub wing_bound_ok: bool,
}

/// The result of a calendar-spread-arbitrage scan over two raw SVI slices.
///
/// When `is_free` is `false`, [`Self::worst_k`] reports the log-moneyness at
/// which the longer-dated total variance falls furthest below the
/// shorter-dated one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalendarReport {
    /// `true` if `w(k, t_2) >= w(k, t_1)` over the scanned domain.
    pub is_free: bool,
    /// The smallest value of `w(k, t_2) - w(k, t_1)` observed.
    pub min_difference: f64,
    /// The log-moneyness at which [`Self::min_difference`] was observed.
    pub worst_k: f64,
}

/// Tests the wing bound `b*(1 + |rho|) <= 2` for a raw SVI slice (Lee 2004).
///
/// The wing bound guarantees call prices vanish at infinite strike; it is one
/// of the two conditions for absence of butterfly arbitrage (MATH.md §7).
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::arbitrage::wing_bound_ok;
///
/// let ok = RawSvi::new(0.04, 0.5, -0.3, 0.0, 0.1).unwrap();
/// assert!(wing_bound_ok(&ok));
/// let steep = RawSvi::new(0.04, 3.0, -0.3, 0.0, 0.1).unwrap();
/// assert!(!wing_bound_ok(&steep));
/// ```
#[must_use]
#[inline]
pub fn wing_bound_ok(svi: &RawSvi) -> bool {
    svi.b * (1.0 + svi.rho.abs()) <= 2.0
}

/// Scans a raw SVI slice for butterfly arbitrage over `[k_lo, k_hi]` plus a
/// wing margin (MATH.md §7).
///
/// Evaluates `g` on a dense grid; any negative value flags arbitrage. The
/// reported [`ButterflyReport::worst_k`] is refined by a Brent root-find on
/// `g` around the deepest grid violation, so the boundary of the arbitrage
/// region is located to a tight tolerance.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::arbitrage::butterfly_scan;
///
/// // A well-behaved slice is free of butterfly arbitrage.
/// let svi = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
/// let report = butterfly_scan(&svi, -0.5, 0.5);
/// assert!(report.is_free);
/// ```
#[must_use]
pub fn butterfly_scan(svi: &RawSvi, k_lo: f64, k_hi: f64) -> ButterflyReport {
    let lo = k_lo.min(k_hi) - SCAN_MARGIN;
    let hi = k_lo.max(k_hi) + SCAN_MARGIN;
    let step = (hi - lo) / index_to_f64(SCAN_POINTS - 1);

    let mut min_g = f64::INFINITY;
    let mut worst_k = lo;
    let mut worst_idx = 0_usize;

    for i in 0..SCAN_POINTS {
        let k = step.mul_add(index_to_f64(i), lo);
        let gi = g(svi, k);
        if gi < min_g {
            min_g = gi;
            worst_k = k;
            worst_idx = i;
        }
    }

    let wing_ok = wing_bound_ok(svi);
    let is_free = min_g >= 0.0 && wing_ok;

    // Refine the worst-violation location with a Brent root-find on g, if a
    // sign change was bracketed near the deepest grid point.
    if !is_free && min_g < 0.0 {
        let lo_k = step.mul_add(index_to_f64(worst_idx.saturating_sub(1)), lo);
        let hi_k = step.mul_add(index_to_f64((worst_idx + 1).min(SCAN_POINTS - 1)), lo);
        if let Some(root) = brent_root(|x| g(svi, x), lo_k, hi_k, REFINE_TOL, REFINE_MAX_ITER) {
            worst_k = root;
        }
    }

    ButterflyReport {
        is_free,
        min_g,
        worst_k,
        wing_bound_ok: wing_ok,
    }
}

/// Tests whether a raw SVI slice is free of butterfly arbitrage over a
/// default symmetric domain `[-1, 1]` in log-moneyness.
///
/// A convenience wrapper over [`butterfly_scan`] for callers without a
/// quoted strike range.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::arbitrage::is_butterfly_free;
///
/// let svi = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
/// assert!(is_butterfly_free(&svi));
/// ```
#[must_use]
pub fn is_butterfly_free(svi: &RawSvi) -> bool {
    butterfly_scan(svi, -1.0, 1.0).is_free
}

/// Scans two raw SVI slices for calendar-spread arbitrage (MATH.md §8).
///
/// `early` is the shorter-dated slice and `late` the longer-dated one. The
/// difference `D(k) = w_late(k) - w_early(k)` is evaluated on a dense grid;
/// any negative value flags arbitrage. The reported worst location is refined
/// by a Brent root-find on `D` when a sign change is bracketed.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::arbitrage::calendar_scan;
///
/// // A higher-variance late slice dominates the early one everywhere.
/// let early = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
/// let late = RawSvi::new(0.08, 0.3, -0.2, 0.0, 0.1).unwrap();
/// let report = calendar_scan(&early, &late, -0.5, 0.5);
/// assert!(report.is_free);
/// ```
#[must_use]
pub fn calendar_scan(early: &RawSvi, late: &RawSvi, k_lo: f64, k_hi: f64) -> CalendarReport {
    let lo = k_lo.min(k_hi) - SCAN_MARGIN;
    let hi = k_lo.max(k_hi) + SCAN_MARGIN;
    let step = (hi - lo) / index_to_f64(SCAN_POINTS - 1);

    let diff = |k: f64| late.total_variance(k) - early.total_variance(k);

    let mut min_diff = f64::INFINITY;
    let mut worst_k = lo;
    let mut worst_idx = 0_usize;

    for i in 0..SCAN_POINTS {
        let k = step.mul_add(index_to_f64(i), lo);
        let d = diff(k);
        if d < min_diff {
            min_diff = d;
            worst_k = k;
            worst_idx = i;
        }
    }

    let is_free = min_diff >= 0.0;

    if !is_free {
        let lo_k = step.mul_add(index_to_f64(worst_idx.saturating_sub(1)), lo);
        let hi_k = step.mul_add(index_to_f64((worst_idx + 1).min(SCAN_POINTS - 1)), lo);
        if let Some(root) = brent_root(diff, lo_k, hi_k, REFINE_TOL, REFINE_MAX_ITER) {
            worst_k = root;
        }
    }

    CalendarReport {
        is_free,
        min_difference: min_diff,
        worst_k,
    }
}

/// Tests whether two raw SVI slices are free of calendar-spread arbitrage
/// over a default domain `[-1, 1]`.
///
/// `early` is the shorter-dated slice, `late` the longer-dated one.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::arbitrage::is_calendar_free;
///
/// let early = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
/// let late = RawSvi::new(0.08, 0.3, -0.2, 0.0, 0.1).unwrap();
/// assert!(is_calendar_free(&early, &late));
/// ```
#[must_use]
pub fn is_calendar_free(early: &RawSvi, late: &RawSvi) -> bool {
    calendar_scan(early, late, -1.0, 1.0).is_free
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssvi::{Phi, Ssvi};

    #[test]
    fn g_is_positive_for_benign_slice() {
        let svi = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
        for &k in &[-1.0, -0.3, 0.0, 0.3, 1.0] {
            assert!(g(&svi, k) > 0.0, "g({k}) should be positive");
        }
    }

    #[test]
    fn g_equals_density_factor_at_atm() {
        // For a flat slice (b = 0), w' = w'' = 0, so g(0) = 1.
        let flat = RawSvi::new(0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
        assert!((g(&flat, 0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn wing_bound_accepts_gentle_rejects_steep() {
        assert!(wing_bound_ok(
            &RawSvi::new(0.04, 0.5, -0.3, 0.0, 0.1).unwrap()
        ));
        assert!(!wing_bound_ok(
            &RawSvi::new(0.04, 3.0, -0.3, 0.0, 0.1).unwrap()
        ));
    }

    #[test]
    fn butterfly_scan_passes_benign_slice() {
        let svi = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
        let report = butterfly_scan(&svi, -0.5, 0.5);
        assert!(report.is_free);
        assert!(report.min_g > 0.0);
        assert!(report.wing_bound_ok);
    }

    #[test]
    fn butterfly_scan_flags_vogt_slice() {
        // The Axel Vogt slice from Gatheral & Jacquier (2014), Section 2.2 —
        // a raw SVI slice with documented butterfly arbitrage.
        // a = -0.0410, b = 0.1331, rho = 0.3060, m = 0.3586, sigma = 0.4153.
        let vogt = RawSvi::new(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153).unwrap();
        let report = butterfly_scan(&vogt, -1.5, 1.5);
        assert!(
            !report.is_free,
            "Vogt slice must be flagged as arbitrageable"
        );
        assert!(report.min_g < 0.0, "min_g = {}", report.min_g);
    }

    #[test]
    fn is_butterfly_free_convenience() {
        let svi = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
        assert!(is_butterfly_free(&svi));
        let vogt = RawSvi::new(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153).unwrap();
        assert!(!is_butterfly_free(&vogt));
    }

    #[test]
    fn calendar_scan_passes_ordered_slices() {
        let early = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
        let late = RawSvi::new(0.08, 0.3, -0.2, 0.0, 0.1).unwrap();
        let report = calendar_scan(&early, &late, -0.5, 0.5);
        assert!(report.is_free);
        assert!(report.min_difference > 0.0);
    }

    #[test]
    fn calendar_scan_flags_crossing_slices() {
        // The late slice has lower ATM variance -> the curves cross.
        let early = RawSvi::new(0.08, 0.3, -0.2, 0.0, 0.1).unwrap();
        let late = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
        let report = calendar_scan(&early, &late, -0.5, 0.5);
        assert!(!report.is_free);
        assert!(report.min_difference < 0.0);
    }

    #[test]
    fn is_calendar_free_convenience() {
        let early = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
        let late = RawSvi::new(0.08, 0.3, -0.2, 0.0, 0.1).unwrap();
        assert!(is_calendar_free(&early, &late));
    }

    #[test]
    fn ssvi_slice_passing_theorem_42_is_butterfly_free() {
        // An SSVI slice satisfying Theorem 4.2 should also pass the g-scan.
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        assert!(ssvi.is_butterfly_free_at(0.04));
        let raw = ssvi.slice_at(0.04).unwrap();
        let report = butterfly_scan(&raw, -1.0, 1.0);
        assert!(report.is_free, "min_g = {}", report.min_g);
    }
}

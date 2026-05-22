// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for regit-svi.
//!
//! Structure:
//!   - mod golden        -- regression anchors against hand-computed values
//!   - mod roundtrip     -- Raw <-> JW and SSVI -> Raw conversion identities
//!   - mod arbitrage     -- known-bad / known-good slice oracle
//!   - mod density       -- risk-neutral density mass and positivity
//!   - mod calibration   -- synthetic-data parameter recovery
//!   - mod surface       -- multi-slice assembly and interpolation
//!   - mod properties    -- proptest invariants

use approx::assert_abs_diff_eq;
use regit_svi::arbitrage::{butterfly_scan, calendar_scan, g, is_butterfly_free};
use regit_svi::calibration::{calibrate_slice, least_squares, quasi_explicit, ssvi as ssvi_cal};
use regit_svi::convert::{jw_to_raw, raw_to_jw, ssvi_to_raw};
use regit_svi::density::{density_report, risk_neutral_density};
use regit_svi::raw::RawSvi;
use regit_svi::ssvi::{Phi, Ssvi};
use regit_svi::surface::Surface;
use regit_svi::types::Quote;

// ─── Tolerance constants ─────────────────────────────────────────────────────

const TIGHT: f64 = 1e-10;
const STANDARD: f64 = 1e-6;
const LOOSE: f64 = 1e-3;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// A representative valid raw SVI slice.
fn reference_slice() -> RawSvi {
    RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.15).unwrap()
}

/// Generates noise-free synthetic quotes from a raw slice.
fn synthetic(svi: &RawSvi, ks: &[f64]) -> Vec<Quote> {
    ks.iter()
        .map(|&k| Quote::new(k, svi.total_variance(k), 1.0).unwrap())
        .collect()
}

// ─── Golden values ───────────────────────────────────────────────────────────

mod golden {
    use super::*;

    #[test]
    fn raw_total_variance_hand_computed() {
        // a=0.04, b=0.4, rho=0, m=0, sigma=0.1: w(0) = a + b*sigma.
        let svi = RawSvi::new(0.04, 0.4, 0.0, 0.0, 0.1).unwrap();
        assert_abs_diff_eq!(svi.total_variance(0.0), 0.08, epsilon = TIGHT);
        // Symmetric smile (rho = 0, m = 0).
        assert_abs_diff_eq!(
            svi.total_variance(0.3),
            svi.total_variance(-0.3),
            epsilon = TIGHT
        );
    }

    #[test]
    fn raw_w_min_attained_at_k_min() {
        let svi = reference_slice();
        let km = svi.k_min();
        assert_abs_diff_eq!(svi.w_prime(km), 0.0, epsilon = TIGHT);
        assert_abs_diff_eq!(svi.w_min(), svi.total_variance(km), epsilon = TIGHT);
    }

    #[test]
    fn ssvi_slice_map_matches_closed_form() {
        // SSVI -> Raw slice map reproduces the SSVI total variance exactly.
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        let raw = ssvi.slice_at(0.04).unwrap();
        for &k in &[-0.6, -0.2, 0.0, 0.2, 0.6] {
            assert_abs_diff_eq!(
                raw.total_variance(k),
                ssvi.total_variance(k, 0.04),
                epsilon = TIGHT
            );
        }
    }

    #[test]
    fn g_for_flat_slice_is_one_at_atm() {
        // For b = 0 the slice is flat, so g(0) = 1 exactly.
        let flat = RawSvi::new(0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
        assert_abs_diff_eq!(g(&flat, 0.0), 1.0, epsilon = TIGHT);
    }
}

// ─── Round-trip identities ───────────────────────────────────────────────────

mod roundtrip {
    use super::*;

    #[test]
    fn raw_jw_raw_is_identity() {
        for &t in &[0.25, 0.5, 1.0, 2.0, 5.0] {
            let raw = reference_slice();
            let back = jw_to_raw(&raw_to_jw(&raw, t).unwrap()).unwrap();
            assert_abs_diff_eq!(back.a, raw.a, epsilon = 1e-8);
            assert_abs_diff_eq!(back.b, raw.b, epsilon = 1e-8);
            assert_abs_diff_eq!(back.rho, raw.rho, epsilon = 1e-8);
            assert_abs_diff_eq!(back.m, raw.m, epsilon = 1e-8);
            assert_abs_diff_eq!(back.sigma, raw.sigma, epsilon = 1e-8);
        }
    }

    #[test]
    fn raw_jw_raw_preserves_total_variance() {
        let raw = RawSvi::new(0.045, 0.38, 0.25, -0.03, 0.14).unwrap();
        let back = jw_to_raw(&raw_to_jw(&raw, 1.5).unwrap()).unwrap();
        for &k in &[-1.0, -0.4, 0.0, 0.4, 1.0] {
            assert_abs_diff_eq!(
                back.total_variance(k),
                raw.total_variance(k),
                epsilon = 1e-8
            );
        }
    }

    #[test]
    fn ssvi_to_raw_matches_slice_at() {
        let ssvi = Ssvi::new(0.2, Phi::heston(2.0).unwrap()).unwrap();
        let via_convert = ssvi_to_raw(&ssvi, 0.05).unwrap();
        let via_method = ssvi.slice_at(0.05).unwrap();
        assert_abs_diff_eq!(via_convert.a, via_method.a, epsilon = TIGHT);
        assert_abs_diff_eq!(via_convert.sigma, via_method.sigma, epsilon = TIGHT);
    }

    #[test]
    fn jw_no_preimage_is_rejected() {
        // A JW tuple with an extreme skew has |beta| > 1.
        let jw = regit_svi::jw::SviJw::new_unchecked(0.04, 10.0, 0.3, 0.25, 0.035, 1.0);
        assert!(jw_to_raw(&jw).is_err());
    }
}

// ─── Arbitrage oracle ────────────────────────────────────────────────────────

mod arbitrage {
    use super::*;

    #[test]
    fn benign_slice_is_butterfly_free() {
        let svi = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
        let report = butterfly_scan(&svi, -1.0, 1.0);
        assert!(report.is_free);
        assert!(report.min_g > 0.0);
        assert!(report.wing_bound_ok);
    }

    #[test]
    fn vogt_slice_has_butterfly_arbitrage() {
        // The Axel Vogt slice from Gatheral & Jacquier (2014), Section 2.2,
        // is a documented example of a raw SVI slice with butterfly arbitrage.
        let vogt = RawSvi::new(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153).unwrap();
        let report = butterfly_scan(&vogt, -1.5, 1.5);
        assert!(!report.is_free, "Vogt slice must be flagged");
        assert!(report.min_g < 0.0);
        // The convenience predicate must agree.
        assert!(!is_butterfly_free(&vogt));
    }

    #[test]
    fn ordered_slices_are_calendar_free() {
        let early = RawSvi::new(0.03, 0.3, -0.2, 0.0, 0.1).unwrap();
        let late = RawSvi::new(0.07, 0.3, -0.2, 0.0, 0.1).unwrap();
        assert!(calendar_scan(&early, &late, -1.0, 1.0).is_free);
    }

    #[test]
    fn crossing_slices_have_calendar_arbitrage() {
        // The longer-dated slice has lower ATM variance -> crossing.
        let early = RawSvi::new(0.07, 0.3, -0.2, 0.0, 0.1).unwrap();
        let late = RawSvi::new(0.03, 0.3, -0.2, 0.0, 0.1).unwrap();
        let report = calendar_scan(&early, &late, -1.0, 1.0);
        assert!(!report.is_free);
        assert!(report.min_difference < 0.0);
    }

    #[test]
    fn ssvi_theorem_42_implies_butterfly_free() {
        // An SSVI surface satisfying Theorem 4.2 yields butterfly-free slices.
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        for &theta in &[0.01, 0.04, 0.09, 0.25] {
            assert!(ssvi.is_butterfly_free_at(theta));
            let raw = ssvi.slice_at(theta).unwrap();
            assert!(butterfly_scan(&raw, -1.5, 1.5).is_free, "theta = {theta}");
        }
    }
}

// ─── Risk-neutral density ────────────────────────────────────────────────────

mod density {
    use super::*;

    #[test]
    fn density_integrates_to_one_on_arbitrage_free_slice() {
        let svi = RawSvi::new(0.04, 0.05, -0.1, 0.0, 0.4).unwrap();
        let report = density_report(&svi, -8.0, 8.0, 4000);
        assert!(report.is_non_negative);
        assert_abs_diff_eq!(report.integral, 1.0, epsilon = LOOSE);
    }

    #[test]
    fn density_is_negative_on_vogt_slice() {
        let vogt = RawSvi::new(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153).unwrap();
        let report = density_report(&vogt, -2.0, 2.0, 3000);
        assert!(!report.is_non_negative);
        assert!(report.min_density < 0.0);
    }

    #[test]
    fn density_is_finite_everywhere() {
        let svi = reference_slice();
        for i in -50..=50 {
            let k = f64::from(i) * 0.1;
            assert!(risk_neutral_density(&svi, k).is_finite());
        }
    }
}

// ─── Calibration recovery ────────────────────────────────────────────────────

mod calibration {
    use super::*;

    #[test]
    fn quasi_explicit_recovers_synthetic_parameters() {
        let truth = reference_slice();
        let ks = [-0.4, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4];
        let quotes = synthetic(&truth, &ks);
        let fit = quasi_explicit::calibrate(&quotes).unwrap();
        assert!(fit.rmse < STANDARD, "rmse = {}", fit.rmse);
        for &k in &[-0.5, -0.2, 0.0, 0.2, 0.5] {
            assert_abs_diff_eq!(
                fit.slice.total_variance(k),
                truth.total_variance(k),
                epsilon = LOOSE
            );
        }
    }

    #[test]
    fn levenberg_marquardt_polishes_synthetic_fit() {
        let truth = RawSvi::new(0.03, 0.35, 0.2, -0.02, 0.16).unwrap();
        let ks = [-0.4, -0.2, -0.05, 0.05, 0.2, 0.4];
        let quotes = synthetic(&truth, &ks);
        let seed = RawSvi::new(0.04, 0.3, 0.1, 0.0, 0.2).unwrap();
        let fit = least_squares::refine(&quotes, &seed).unwrap();
        assert!(fit.rmse < STANDARD, "rmse = {}", fit.rmse);
    }

    #[test]
    fn default_pipeline_recovers_and_certifies() {
        let truth = reference_slice();
        let ks = [-0.4, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4];
        let quotes = synthetic(&truth, &ks);
        let fit = calibrate_slice(&quotes).unwrap();
        assert!(fit.rmse < STANDARD);
        assert!(fit.butterfly_free);
    }

    #[test]
    fn calibration_degrades_gracefully_with_noise() {
        let truth = reference_slice();
        let ks = [-0.4, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4];
        // Deterministic pseudo-noise via a small LCG.
        let mut state = 987_654_321_u64;
        let quotes: Vec<Quote> = ks
            .iter()
            .map(|&k| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let unit = f64::from(u32::try_from(state >> 40).unwrap_or(0)) / f64::from(u32::MAX);
                let noise = (unit - 0.5) * 2.0 * 5e-4;
                Quote::new(k, truth.total_variance(k) + noise, 1.0).unwrap()
            })
            .collect();
        let fit = calibrate_slice(&quotes).unwrap();
        assert!(fit.rmse < LOOSE, "rmse = {}", fit.rmse);
    }

    #[test]
    fn ssvi_surface_calibration_recovers_truth() {
        let truth = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        let ks = [-0.3, -0.15, 0.0, 0.15, 0.3];
        let mats: Vec<ssvi_cal::SsviMaturity> = [(0.5, 0.02), (1.0, 0.04), (2.0, 0.07)]
            .iter()
            .map(|&(t, theta)| ssvi_cal::SsviMaturity {
                t,
                theta,
                quotes: ks
                    .iter()
                    .map(|&k| Quote::new(k, truth.total_variance(k, theta), 1.0).unwrap())
                    .collect(),
            })
            .collect();
        let fit = ssvi_cal::calibrate(&mats, ssvi_cal::PhiFamily::PowerLaw).unwrap();
        assert!(fit.rmse < LOOSE, "rmse = {}", fit.rmse);
        assert!(fit.arbitrage_free);
        assert_abs_diff_eq!(fit.ssvi.rho, -0.3, epsilon = 0.1);
    }

    #[test]
    fn calibration_rejects_insufficient_data() {
        let q = Quote::new(0.0, 0.04, 1.0).unwrap();
        assert!(quasi_explicit::calibrate(&[q, q]).is_err());
        assert!(calibrate_slice(&[]).is_err());
    }
}

// ─── Surface assembly ────────────────────────────────────────────────────────

mod surface {
    use super::*;

    #[test]
    fn slice_surface_interpolates_linearly_in_w() {
        let s1 = RawSvi::new(0.02, 0.3, -0.2, 0.0, 0.1).unwrap();
        let s2 = RawSvi::new(0.06, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(1.0, s1), (3.0, s2)]).unwrap();
        for &k in &[-0.3, 0.0, 0.3] {
            let mid = surface.total_variance(k, 2.0);
            let expect = 0.5 * (s1.total_variance(k) + s2.total_variance(k));
            assert_abs_diff_eq!(mid, expect, epsilon = TIGHT);
        }
    }

    #[test]
    fn surface_recovers_knot_slices() {
        let s1 = RawSvi::new(0.02, 0.3, -0.2, 0.0, 0.1).unwrap();
        let s2 = RawSvi::new(0.06, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(1.0, s1), (3.0, s2)]).unwrap();
        assert_abs_diff_eq!(
            surface.total_variance(0.1, 1.0),
            s1.total_variance(0.1),
            epsilon = TIGHT
        );
        assert_abs_diff_eq!(
            surface.total_variance(0.1, 3.0),
            s2.total_variance(0.1),
            epsilon = TIGHT
        );
    }

    #[test]
    fn surface_extrapolation_is_flat_in_vol() {
        let s = RawSvi::new(0.04, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(1.0, s)]).unwrap();
        let v_short = surface.implied_vol(0.0, 0.25).unwrap();
        let v_long = surface.implied_vol(0.0, 4.0).unwrap();
        assert_abs_diff_eq!(v_short, v_long, epsilon = TIGHT);
    }

    #[test]
    fn ssvi_surface_is_calendar_free() {
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        let surface =
            Surface::from_ssvi(ssvi, vec![(0.5, 0.02), (1.0, 0.04), (2.0, 0.08)]).unwrap();
        assert!(surface.is_calendar_free(-0.5, 0.5));
    }

    #[test]
    fn ordered_slice_surface_passes_calendar_check() {
        let early = RawSvi::new(0.02, 0.3, -0.2, 0.0, 0.1).unwrap();
        let late = RawSvi::new(0.06, 0.3, -0.2, 0.0, 0.1).unwrap();
        let surface = Surface::from_slices(vec![(0.5, early), (1.5, late)]).unwrap();
        assert!(surface.is_calendar_free(-0.5, 0.5));
    }
}

// ─── proptest invariants ─────────────────────────────────────────────────────

mod properties {
    use super::*;
    use proptest::prelude::*;

    /// Strategy producing a valid raw SVI slice.
    fn raw_svi_strategy() -> impl Strategy<Value = RawSvi> {
        (
            0.001_f64..0.2,  // a
            0.01_f64..1.5,   // b
            -0.95_f64..0.95, // rho
            -0.5_f64..0.5,   // m
            0.02_f64..0.6,   // sigma
        )
            .prop_filter("valid raw SVI domain", |&(a, b, rho, m, sigma)| {
                RawSvi::new(a, b, rho, m, sigma).is_ok()
            })
            .prop_map(|(a, b, rho, m, sigma)| RawSvi::new(a, b, rho, m, sigma).unwrap())
    }

    proptest! {
        #[test]
        fn raw_total_variance_is_non_negative(svi in raw_svi_strategy(), k in -3.0_f64..3.0) {
            prop_assert!(svi.total_variance(k) >= 0.0);
        }

        #[test]
        fn raw_is_strictly_convex(svi in raw_svi_strategy(), k in -3.0_f64..3.0) {
            // w'' > 0 everywhere on the valid domain.
            prop_assert!(svi.w_double_prime(k) > 0.0);
        }

        #[test]
        fn raw_jw_roundtrip_preserves_total_variance(
            svi in raw_svi_strategy(),
            t in 0.1_f64..5.0,
            k in -1.0_f64..1.0,
        ) {
            if let Ok(jw) = raw_to_jw(&svi, t) {
                if let Ok(back) = jw_to_raw(&jw) {
                    prop_assert!(
                        (back.total_variance(k) - svi.total_variance(k)).abs() < 1e-6
                    );
                }
            }
        }

        #[test]
        fn ssvi_theorem_42_slices_pass_butterfly_scan(
            rho in -0.9_f64..0.9,
            eta in 0.05_f64..1.0,
            gamma in 0.1_f64..0.9,
            theta in 0.005_f64..0.3,
        ) {
            // An SSVI slice satisfying Theorem 4.2 must pass the g-scan.
            if let Ok(phi) = Phi::power_law(eta, gamma) {
                if let Ok(ssvi) = Ssvi::new(rho, phi) {
                    if ssvi.is_butterfly_free_at(theta) {
                        if let Ok(raw) = ssvi.slice_at(theta) {
                            prop_assert!(butterfly_scan(&raw, -2.0, 2.0).is_free);
                        }
                    }
                }
            }
        }

        #[test]
        fn raw_w_min_is_the_global_minimum(svi in raw_svi_strategy(), k in -3.0_f64..3.0) {
            // No point on the slice has total variance below w_min.
            prop_assert!(svi.total_variance(k) >= svi.w_min() - 1e-9);
        }
    }
}

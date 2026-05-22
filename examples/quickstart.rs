// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Quickstart example for regit-svi.
//!
//! Walks through the full slice and surface workflow: calibrate a raw SVI
//! slice to market quotes, convert between parametrisations, check static
//! arbitrage, inspect the risk-neutral density, fit an arbitrage-free SSVI
//! surface, and evaluate it.

use regit_svi::arbitrage::butterfly_scan;
use regit_svi::calibration::ssvi::{PhiFamily, SsviMaturity, calibrate as calibrate_ssvi};
use regit_svi::calibration::{calibrate_slice, quasi_explicit};
use regit_svi::convert::{jw_to_raw, raw_to_jw};
use regit_svi::density::density_report;
use regit_svi::ssvi::{Phi, Ssvi};
use regit_svi::surface::Surface;
use regit_svi::types::Quote;

fn main() {
    // ── 1. Calibrate a raw SVI slice ────────────────────────────────────
    // Market quotes for one maturity: (log-moneyness, total variance, weight).
    let quotes = [
        Quote::new(-0.20, 0.0512, 1.0).unwrap(),
        Quote::new(-0.10, 0.0432, 1.0).unwrap(),
        Quote::new(0.00, 0.0400, 1.0).unwrap(),
        Quote::new(0.10, 0.0420, 1.0).unwrap(),
        Quote::new(0.20, 0.0480, 1.0).unwrap(),
    ];

    // Default pipeline: quasi-explicit fit, then a Levenberg-Marquardt polish.
    let fit = calibrate_slice(&quotes).expect("calibration");
    let svi = fit.slice;
    println!("Calibrated raw SVI slice");
    println!("  a = {:.6}  b = {:.6}  rho = {:.6}", svi.a, svi.b, svi.rho);
    println!("  m = {:.6}  sigma = {:.6}", svi.m, svi.sigma);
    println!("  RMSE = {:.3e}", fit.rmse);
    println!("  butterfly-free = {}", fit.butterfly_free);

    // The quasi-explicit calibrator is also usable directly.
    let qe = quasi_explicit::calibrate(&quotes).expect("quasi-explicit");
    println!("  quasi-explicit RMSE = {:.3e}", qe.rmse);

    // ── 2. Evaluate the slice ───────────────────────────────────────────
    let w = svi.total_variance(0.05);
    let vol = svi.implied_vol(0.05, 1.0).expect("vol"); // 1-year expiry
    println!("\nAt k = 0.05:  w = {w:.6},  implied vol = {vol:.4}");
    println!("ATM variance = {:.6}", svi.atm_total_variance());
    println!("ATM skew     = {:.6}", svi.atm_skew());

    // ── 3. Convert between parametrisations ─────────────────────────────
    let jw = raw_to_jw(&svi, 1.0).expect("raw -> JW");
    println!("\nJump-Wings view (t = 1y)");
    println!("  v_t = {:.6}  psi_t = {:.6}", jw.v_t, jw.psi_t);
    println!("  p_t = {:.6}  c_t = {:.6}", jw.p_t, jw.c_t);
    let back = jw_to_raw(&jw).expect("JW -> raw");
    println!(
        "  round-trip max param error: {:.2e}",
        (back.a - svi.a).abs()
    );

    // ── 4. Static-arbitrage checks ──────────────────────────────────────
    let report = butterfly_scan(&svi, -0.5, 0.5);
    println!("\nButterfly scan");
    println!("  is_free = {}", report.is_free);
    println!(
        "  min g(k) = {:.6} at k = {:.4}",
        report.min_g, report.worst_k
    );

    // ── 5. Risk-neutral density ─────────────────────────────────────────
    let density = density_report(&svi, -6.0, 6.0, 2000);
    println!("\nRisk-neutral density");
    println!("  integral over [-6, 6] = {:.6}", density.integral);
    println!("  non-negative = {}", density.is_non_negative);

    // ── 6. SSVI surface — arbitrage-free by construction ────────────────
    let truth = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
    let ks = [-0.3, -0.15, 0.0, 0.15, 0.3];
    let maturities: Vec<SsviMaturity> = [(0.5, 0.02), (1.0, 0.04), (2.0, 0.07)]
        .iter()
        .map(|&(t, theta)| SsviMaturity {
            t,
            theta,
            quotes: ks
                .iter()
                .map(|&k| Quote::new(k, truth.total_variance(k, theta), 1.0).unwrap())
                .collect(),
        })
        .collect();

    let ssvi_fit = calibrate_ssvi(&maturities, PhiFamily::PowerLaw).expect("SSVI calibration");
    println!("\nSSVI surface calibration");
    println!("  rho = {:.6}", ssvi_fit.ssvi.rho);
    println!("  RMSE = {:.3e}", ssvi_fit.rmse);
    println!("  arbitrage-free = {}", ssvi_fit.arbitrage_free);

    // ── 7. Assemble and evaluate a surface ──────────────────────────────
    let surface = Surface::from_ssvi(
        ssvi_fit.ssvi,
        maturities.iter().map(|m| (m.t, m.theta)).collect(),
    )
    .expect("surface");
    let w_interp = surface.total_variance(0.05, 1.5);
    let vol_interp = surface.implied_vol(0.05, 1.5).expect("surface vol");
    println!("\nSurface at (k = 0.05, T = 1.5y)");
    println!("  w = {w_interp:.6},  implied vol = {vol_interp:.4}");
    println!("  calendar-free = {}", surface.is_calendar_free(-0.4, 0.4));
}

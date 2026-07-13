// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for regit-svi.
//!
//! Performance targets (indicative, native release on commodity hardware):
//!
//! | Operation                       | Target   |
//! |---------------------------------|----------|
//! | Raw SVI total variance (f64)    | < 10 ns  |
//! | Raw SVI w' + w'' (f64)          | < 25 ns  |
//! | Butterfly g(k) (f64)            | < 30 ns  |
//! | Raw -> JW -> Raw round-trip     | < 200 ns |
//! | Quasi-explicit slice calibrate  | < 2 ms   |
//! | Levenberg-Marquardt polish      | < 500 us |

use criterion::{Criterion, criterion_group, criterion_main};
use regit_svi::arbitrage::{butterfly_scan, g};
use regit_svi::calibration::{least_squares, quasi_explicit};
use regit_svi::convert::{jw_to_raw, raw_to_jw};
use regit_svi::raw::RawSvi;
use regit_svi::ssvi::{Phi, Ssvi};
use regit_svi::types::Quote;
use std::hint::black_box;

// ─── Fixtures ────────────────────────────────────────────────────────────────

fn slice() -> RawSvi {
    RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.15).unwrap()
}

fn synthetic_quotes() -> Vec<Quote> {
    let truth = slice();
    [-0.4, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4]
        .iter()
        .map(|&k| Quote::new(k, truth.total_variance(k), 1.0).unwrap())
        .collect()
}

// ─── Raw SVI evaluation ──────────────────────────────────────────────────────

fn bench_raw(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw");
    let svi = slice();

    group.bench_function("total_variance", |b| {
        b.iter(|| svi.total_variance(black_box(0.12)));
    });
    group.bench_function("w_prime", |b| {
        b.iter(|| svi.w_prime(black_box(0.12)));
    });
    group.bench_function("w_double_prime", |b| {
        b.iter(|| svi.w_double_prime(black_box(0.12)));
    });

    group.finish();
}

// ─── SSVI evaluation ─────────────────────────────────────────────────────────

fn bench_ssvi(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssvi");
    let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();

    group.bench_function("total_variance", |b| {
        b.iter(|| ssvi.total_variance(black_box(0.12), black_box(0.04)));
    });
    group.bench_function("slice_at", |b| {
        b.iter(|| ssvi.slice_at(black_box(0.04)));
    });

    group.finish();
}

// ─── Arbitrage checks ────────────────────────────────────────────────────────

fn bench_arbitrage(c: &mut Criterion) {
    let mut group = c.benchmark_group("arbitrage");
    let svi = slice();

    group.bench_function("g", |b| {
        b.iter(|| g(black_box(&svi), black_box(0.12)));
    });
    group.bench_function("butterfly_scan", |b| {
        b.iter(|| butterfly_scan(black_box(&svi), -0.5, 0.5));
    });

    group.finish();
}

// ─── Conversions ─────────────────────────────────────────────────────────────

fn bench_convert(c: &mut Criterion) {
    let mut group = c.benchmark_group("convert");
    let svi = slice();
    let jw = raw_to_jw(&svi, 1.0).unwrap();

    group.bench_function("raw_to_jw", |b| {
        b.iter(|| raw_to_jw(black_box(&svi), 1.0));
    });
    group.bench_function("jw_to_raw", |b| {
        b.iter(|| jw_to_raw(black_box(&jw)));
    });

    group.finish();
}

// ─── Calibration ─────────────────────────────────────────────────────────────

fn bench_calibration(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration");
    let quotes = synthetic_quotes();
    let seed = slice();

    group.bench_function("quasi_explicit", |b| {
        b.iter(|| quasi_explicit::calibrate(black_box(&quotes)));
    });
    group.bench_function("levenberg_marquardt_refine", |b| {
        b.iter(|| least_squares::refine(black_box(&quotes), black_box(&seed)));
    });

    group.finish();
}

// ─── Harness ─────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_raw,
    bench_ssvi,
    bench_arbitrage,
    bench_convert,
    bench_calibration,
);
criterion_main!(benches);

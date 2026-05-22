<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x     | Yes (pre-release) |

The supported-version table is updated to `1.x` on the first stable release.

## Reporting a Vulnerability

If you discover a security vulnerability in `regit-svi`, please report it
responsibly:

1. **Do not** open a public GitHub issue
2. Email **nicolas.koenig@regit.io** with a description of the vulnerability
3. Include steps to reproduce if possible
4. We will acknowledge receipt within 48 hours and provide a timeline for a fix

## Scope

This crate performs mathematical computation only — it does not handle network
I/O, file I/O, user authentication, or any form of external communication. It
has zero runtime dependencies.

The primary security concern is **numerical correctness**: an error in
calibration, in a parametrisation conversion, or in an arbitrage check could
lead to a volatility surface that misprices options or that carries undetected
static arbitrage into a downstream pricing or risk system.

In particular, a **false negative** from the butterfly (`g(k) >= 0`) or
calendar-spread check — reporting a surface as arbitrage-free when it is not —
is treated as a correctness defect of the highest severity.

If you find a numerical accuracy issue that falls outside the documented
tolerance bounds, or any case where an arbitrage check returns an incorrect
verdict, please report it using the process above.

## Dependencies

The crate has no runtime dependencies. License and supply-chain concerns for
development dependencies are policed via `cargo-deny` (`deny.toml` in the
repository root), checked in CI on every push. Dependency changes that
introduce a non-allowed licence or an active advisory are rejected at the gate.

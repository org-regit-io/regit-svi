// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Typed error enums for parametrisation, conversion, and calibration.
//!
//! All failure paths return a typed `Result` — no `panic!()`, no `unwrap()`,
//! no string errors. Each variant carries enough context for the caller to
//! decide how to recover.
//!
//! Three enums separate the three failure domains:
//!
//! - [`ParamError`] — invalid SVI / SSVI parameters or out-of-domain quotes.
//! - [`ConvertError`] — a parametrisation conversion has no valid pre-image.
//! - [`CalibrationError`] — a calibrator could not produce a usable fit.

use core::fmt;

// ─── Parametrisation errors ──────────────────────────────────────────────────

/// Error returned when SVI / SSVI parameters or market quotes fail validation.
///
/// Every SVI parametrisation has a validity domain (raw SVI: `b >= 0`,
/// `|rho| < 1`, `sigma > 0`, `a + b*sigma*sqrt(1-rho^2) >= 0`). A constructor
/// or `validate` method returns one of these variants when an input lies
/// outside that domain.
///
/// # Examples
///
/// ```
/// use regit_svi::errors::ParamError;
///
/// let err = ParamError::NegativeSlope { b: -0.1 };
/// assert_eq!(format!("{err}"), "raw SVI slope b must be non-negative, got -0.1");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamError {
    /// Raw SVI slope `b` is negative.
    NegativeSlope {
        /// The offending value of `b`.
        b: f64,
    },
    /// Raw SVI / SSVI correlation `rho` is outside `(-1, 1)`.
    CorrelationOutOfRange {
        /// The offending value of `rho`.
        rho: f64,
    },
    /// Raw SVI curvature `sigma` is not strictly positive.
    NonPositiveSigma {
        /// The offending value of `sigma`.
        sigma: f64,
    },
    /// The minimum total variance `a + b*sigma*sqrt(1-rho^2)` is negative,
    /// so the slice produces negative variance somewhere.
    NegativeMinVariance {
        /// The minimum value of `w` over the slice.
        w_min: f64,
    },
    /// A maturity `t` is not strictly positive.
    NonPositiveMaturity {
        /// The offending value of `t`.
        t: f64,
    },
    /// A fitting weight on a market quote is negative.
    NegativeWeight {
        /// The offending weight.
        weight: f64,
    },
    /// An observed total variance on a market quote is negative.
    NegativeTotalVariance {
        /// The offending total variance.
        w: f64,
    },
    /// An SSVI smoothing-function parameter is outside its valid domain
    /// (`lambda > 0`, `eta > 0`, `gamma in (0, 1)`).
    InvalidPhiParameter {
        /// Human-readable name of the offending parameter.
        name: &'static str,
        /// The offending value.
        value: f64,
    },
    /// An SSVI ATM total variance `theta` is not strictly positive.
    NonPositiveTheta {
        /// The offending value of `theta`.
        theta: f64,
    },
    /// A non-finite (`NaN` or infinite) value was supplied where a finite
    /// number is required.
    NonFinite {
        /// Human-readable name of the offending input.
        name: &'static str,
    },
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeSlope { b } => {
                write!(f, "raw SVI slope b must be non-negative, got {b}")
            }
            Self::CorrelationOutOfRange { rho } => {
                write!(f, "correlation rho must lie in (-1, 1), got {rho}")
            }
            Self::NonPositiveSigma { sigma } => {
                write!(f, "raw SVI curvature sigma must be positive, got {sigma}")
            }
            Self::NegativeMinVariance { w_min } => {
                write!(
                    f,
                    "minimum total variance must be non-negative, got w_min = {w_min}"
                )
            }
            Self::NonPositiveMaturity { t } => {
                write!(f, "maturity t must be positive, got {t}")
            }
            Self::NegativeWeight { weight } => {
                write!(f, "quote weight must be non-negative, got {weight}")
            }
            Self::NegativeTotalVariance { w } => {
                write!(f, "quoted total variance must be non-negative, got {w}")
            }
            Self::InvalidPhiParameter { name, value } => {
                write!(f, "SSVI phi parameter {name} is out of range: {value}")
            }
            Self::NonPositiveTheta { theta } => {
                write!(f, "SSVI ATM variance theta must be positive, got {theta}")
            }
            Self::NonFinite { name } => {
                write!(f, "input {name} must be a finite number")
            }
        }
    }
}

impl std::error::Error for ParamError {}

// ─── Conversion errors ───────────────────────────────────────────────────────

/// Error returned when a parametrisation conversion has no valid pre-image.
///
/// The Raw <-> Jump-Wings map is bijective only on a subset of JW space: a
/// JW tuple with `|beta| > 1` (where `beta = rho - 2*psi*sqrt(w)/b`) does not
/// correspond to any raw SVI slice — see MATH.md §4.
///
/// # Examples
///
/// ```
/// use regit_svi::errors::ConvertError;
///
/// let err = ConvertError::JwHasNoRawPreimage { beta: 1.4 };
/// let msg = format!("{err}");
/// assert!(msg.contains("1.4"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvertError {
    /// The Jump-Wings tuple yields `|beta| > 1`, so no raw SVI slice exists.
    JwHasNoRawPreimage {
        /// The computed value of `beta`.
        beta: f64,
    },
    /// A wing slope `p_t` or `c_t` is negative, which has no raw pre-image.
    NegativeWingSlope {
        /// Human-readable name of the offending wing slope.
        name: &'static str,
        /// The offending value.
        value: f64,
    },
    /// The Jump-Wings ATM total variance `v_t * t` is not strictly positive.
    NonPositiveAtmVariance {
        /// The computed ATM total variance.
        w: f64,
    },
    /// A degenerate intermediate (`b = 0` or `c_t + p_t = 0`) makes the
    /// inverse map indeterminate.
    DegenerateJw,
    /// A parameter error surfaced while constructing the converted slice.
    Param(ParamError),
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JwHasNoRawPreimage { beta } => {
                write!(
                    f,
                    "Jump-Wings tuple has no raw SVI pre-image: |beta| > 1, beta = {beta}"
                )
            }
            Self::NegativeWingSlope { name, value } => {
                write!(
                    f,
                    "Jump-Wings wing slope {name} must be non-negative, got {value}"
                )
            }
            Self::NonPositiveAtmVariance { w } => {
                write!(f, "Jump-Wings ATM total variance must be positive, got {w}")
            }
            Self::DegenerateJw => {
                write!(
                    f,
                    "Jump-Wings tuple is degenerate: inverse map is indeterminate"
                )
            }
            Self::Param(e) => write!(f, "converted slice is invalid: {e}"),
        }
    }
}

impl std::error::Error for ConvertError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Param(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ParamError> for ConvertError {
    fn from(e: ParamError) -> Self {
        Self::Param(e)
    }
}

// ─── Calibration errors ──────────────────────────────────────────────────────

/// Error returned when a calibrator cannot produce a usable fit.
///
/// Covers insufficient data, non-convergence of the outer optimiser, and any
/// parameter error surfaced while assembling the calibrated slice.
///
/// # Examples
///
/// ```
/// use regit_svi::errors::CalibrationError;
///
/// let err = CalibrationError::TooFewQuotes { got: 2, need: 5 };
/// let msg = format!("{err}");
/// assert!(msg.contains("2"));
/// assert!(msg.contains("5"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationError {
    /// Fewer quotes were supplied than the model has free parameters.
    TooFewQuotes {
        /// Number of quotes supplied.
        got: usize,
        /// Minimum number of quotes required.
        need: usize,
    },
    /// The supplied quote set is empty.
    EmptyQuotes,
    /// The outer optimiser reached its iteration cap without converging.
    DidNotConverge {
        /// Number of iterations performed.
        iterations: usize,
        /// The final residual norm.
        residual: f64,
    },
    /// All fitting weights are zero, so the objective is identically zero.
    AllWeightsZero,
    /// A parameter error surfaced while assembling the calibrated slice.
    Param(ParamError),
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewQuotes { got, need } => {
                write!(
                    f,
                    "too few quotes for calibration: got {got}, need at least {need}"
                )
            }
            Self::EmptyQuotes => write!(f, "quote set is empty"),
            Self::DidNotConverge {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "calibration did not converge after {iterations} iterations, residual = {residual}"
                )
            }
            Self::AllWeightsZero => write!(f, "all fitting weights are zero"),
            Self::Param(e) => write!(f, "calibrated slice is invalid: {e}"),
        }
    }
}

impl std::error::Error for CalibrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Param(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ParamError> for CalibrationError {
    fn from(e: ParamError) -> Self {
        Self::Param(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_error_display_negative_slope() {
        let err = ParamError::NegativeSlope { b: -0.1 };
        assert_eq!(
            format!("{err}"),
            "raw SVI slope b must be non-negative, got -0.1"
        );
    }

    #[test]
    fn param_error_display_correlation() {
        let err = ParamError::CorrelationOutOfRange { rho: 1.5 };
        assert!(format!("{err}").contains("1.5"));
    }

    #[test]
    fn param_error_display_non_positive_sigma() {
        let err = ParamError::NonPositiveSigma { sigma: 0.0 };
        assert!(format!("{err}").contains("sigma"));
    }

    #[test]
    fn param_error_display_negative_min_variance() {
        let err = ParamError::NegativeMinVariance { w_min: -0.01 };
        assert!(format!("{err}").contains("w_min"));
    }

    #[test]
    fn param_error_display_remaining_variants() {
        assert!(format!("{}", ParamError::NonPositiveMaturity { t: 0.0 }).contains("maturity"));
        assert!(format!("{}", ParamError::NegativeWeight { weight: -1.0 }).contains("weight"));
        assert!(
            format!("{}", ParamError::NegativeTotalVariance { w: -0.1 }).contains("total variance")
        );
        assert!(
            format!(
                "{}",
                ParamError::InvalidPhiParameter {
                    name: "eta",
                    value: -1.0
                }
            )
            .contains("eta")
        );
        assert!(format!("{}", ParamError::NonPositiveTheta { theta: 0.0 }).contains("theta"));
        assert!(format!("{}", ParamError::NonFinite { name: "k" }).contains("finite"));
    }

    #[test]
    fn param_error_is_error_trait() {
        let err: &dyn std::error::Error = &ParamError::NegativeSlope { b: -1.0 };
        assert!(err.source().is_none());
    }

    #[test]
    fn param_error_copy_eq() {
        let err = ParamError::NonFinite { name: "x" };
        let copy = err;
        assert_eq!(err, copy);
    }

    #[test]
    fn convert_error_display() {
        let err = ConvertError::JwHasNoRawPreimage { beta: 1.4 };
        assert!(format!("{err}").contains("1.4"));
        let err = ConvertError::NegativeWingSlope {
            name: "p_t",
            value: -1.0,
        };
        assert!(format!("{err}").contains("p_t"));
        assert!(format!("{}", ConvertError::DegenerateJw).contains("degenerate"));
        assert!(
            format!("{}", ConvertError::NonPositiveAtmVariance { w: -0.1 }).contains("positive")
        );
    }

    #[test]
    fn convert_error_from_param_and_source() {
        let pe = ParamError::NegativeSlope { b: -1.0 };
        let ce: ConvertError = pe.into();
        assert!(matches!(ce, ConvertError::Param(_)));
        let dyn_err: &dyn std::error::Error = &ce;
        assert!(dyn_err.source().is_some());
    }

    #[test]
    fn calibration_error_display() {
        let err = CalibrationError::TooFewQuotes { got: 2, need: 5 };
        let msg = format!("{err}");
        assert!(msg.contains('2') && msg.contains('5'));
        assert!(format!("{}", CalibrationError::EmptyQuotes).contains("empty"));
        assert!(
            format!(
                "{}",
                CalibrationError::DidNotConverge {
                    iterations: 100,
                    residual: 1e-3
                }
            )
            .contains("converge")
        );
        assert!(format!("{}", CalibrationError::AllWeightsZero).contains("weights"));
    }

    #[test]
    fn calibration_error_from_param_and_source() {
        let pe = ParamError::NonPositiveSigma { sigma: 0.0 };
        let ce: CalibrationError = pe.into();
        assert!(matches!(ce, CalibrationError::Param(_)));
        let dyn_err: &dyn std::error::Error = &ce;
        assert!(dyn_err.source().is_some());
    }

    #[test]
    fn errors_debug() {
        assert!(format!("{:?}", ParamError::NonFinite { name: "k" }).contains("NonFinite"));
        assert!(format!("{:?}", ConvertError::DegenerateJw).contains("Degenerate"));
        assert!(format!("{:?}", CalibrationError::EmptyQuotes).contains("Empty"));
    }
}

// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Surface SVI (SSVI) — the whole-surface arbitrage-free parametrisation.
//!
//! SSVI parametrises the entire surface at once, as a function of
//! log-moneyness `k` and the ATM total variance `theta = w(0, t)`:
//!
//! ```text
//! w(k, theta) = (theta / 2)
//!             * ( 1 + rho*phi(theta)*k
//!                   + sqrt( (phi(theta)*k + rho)^2 + (1 - rho^2) ) )
//! ```
//!
//! The free objects are a single global correlation `rho in (-1, 1)`, the ATM
//! total-variance term structure `theta_t`, and a smoothing function
//! `phi : R+ -> R+`. Two standard choices of `phi` are supported (MATH.md §6):
//!
//! ```text
//! Heston-like:  phi(theta) = (1/(lambda*theta))
//!                          * (1 - (1 - exp(-lambda*theta))/(lambda*theta))
//! Power-law:    phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))
//! ```
//!
//! For fixed `theta`, an SSVI slice equals a raw SVI slice — the closed-form
//! map is [`Ssvi::slice_at`].
//!
//! SSVI's value is that static arbitrage reduces to closed-form inequalities
//! (Gatheral & Jacquier 2014, Theorems 4.1 and 4.2), checked here by
//! [`Ssvi::is_butterfly_free`] and [`Ssvi::is_calendar_free`].
//!
//! # References
//!
//! - Gatheral, J. & Jacquier, A., "Arbitrage-free SVI volatility surfaces",
//!   *Quantitative Finance* 14(1):59-71 (2014), Section 4.

use crate::errors::ParamError;
use crate::raw::RawSvi;

/// SSVI smoothing function `phi : R+ -> R+`.
///
/// `phi` controls how the smile curvature evolves with maturity. Both
/// supported forms are positive and decreasing in `theta`, as required for an
/// arbitrage-free surface.
///
/// # Examples
///
/// ```
/// use regit_svi::ssvi::Phi;
///
/// let heston = Phi::heston(1.0).unwrap();
/// let power = Phi::power_law(0.5, 0.5).unwrap();
/// assert!(heston.eval(0.04) > 0.0);
/// assert!(power.eval(0.04) > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Phi {
    /// Heston-like form, parametrised by `lambda > 0`.
    Heston {
        /// Mean-reversion-like speed `lambda > 0`.
        lambda: f64,
    },
    /// Power-law form, parametrised by `eta > 0` and `gamma in (0, 1)`.
    PowerLaw {
        /// Overall scale `eta > 0`.
        eta: f64,
        /// Decay exponent `gamma in (0, 1)`.
        gamma: f64,
    },
}

impl Phi {
    /// Creates a validated Heston-like smoothing function.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::InvalidPhiParameter`] if `lambda <= 0`, or
    /// [`ParamError::NonFinite`] if `lambda` is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::Phi;
    ///
    /// assert!(Phi::heston(1.5).is_ok());
    /// assert!(Phi::heston(0.0).is_err());
    /// ```
    pub fn heston(lambda: f64) -> Result<Self, ParamError> {
        if !lambda.is_finite() {
            return Err(ParamError::NonFinite { name: "lambda" });
        }
        if lambda <= 0.0 {
            return Err(ParamError::InvalidPhiParameter {
                name: "lambda",
                value: lambda,
            });
        }
        Ok(Self::Heston { lambda })
    }

    /// Creates a validated power-law smoothing function.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::InvalidPhiParameter`] if `eta <= 0` or
    /// `gamma` is outside `(0, 1)`, or [`ParamError::NonFinite`] if either
    /// parameter is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::Phi;
    ///
    /// assert!(Phi::power_law(0.5, 0.5).is_ok());
    /// assert!(Phi::power_law(0.5, 1.0).is_err());
    /// ```
    pub fn power_law(eta: f64, gamma: f64) -> Result<Self, ParamError> {
        if !eta.is_finite() {
            return Err(ParamError::NonFinite { name: "eta" });
        }
        if !gamma.is_finite() {
            return Err(ParamError::NonFinite { name: "gamma" });
        }
        if eta <= 0.0 {
            return Err(ParamError::InvalidPhiParameter {
                name: "eta",
                value: eta,
            });
        }
        if gamma <= 0.0 || gamma >= 1.0 {
            return Err(ParamError::InvalidPhiParameter {
                name: "gamma",
                value: gamma,
            });
        }
        Ok(Self::PowerLaw { eta, gamma })
    }

    /// Evaluates `phi(theta)` for `theta > 0`.
    ///
    /// The Heston-like form uses `x = lambda*theta` and
    /// `phi = (1/x)*(1 - (1 - exp(-x))/x)`; the power-law form uses
    /// `phi = eta / (theta^gamma * (1 + theta)^(1-gamma))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::Phi;
    ///
    /// // Power-law at theta = 1: phi = eta / (1 * 2^(1-gamma)).
    /// let p = Phi::power_law(1.0, 0.5).unwrap();
    /// assert!((p.eval(1.0) - 1.0 / 2.0_f64.powf(0.5)).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn eval(&self, theta: f64) -> f64 {
        match *self {
            Self::Heston { lambda } => {
                let x = lambda * theta;
                // phi(theta) = (1/x) * (1 - (1 - exp(-x))/x)
                (1.0 / x) * (1.0 - (1.0 - (-x).exp()) / x)
            }
            Self::PowerLaw { eta, gamma } => {
                eta / (theta.powf(gamma) * (1.0 + theta).powf(1.0 - gamma))
            }
        }
    }
}

/// A Surface SVI parametrisation: a global correlation plus a smoothing
/// function.
///
/// The ATM total-variance term structure `theta_t` is supplied per evaluation
/// (it is interpolated from market quotes by the surface layer), so an
/// [`Ssvi`] value is the maturity-independent part of the surface.
///
/// # Examples
///
/// ```
/// use regit_svi::ssvi::{Phi, Ssvi};
///
/// let ssvi = Ssvi::new(-0.4, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
/// let w = ssvi.total_variance(0.1, 0.04);
/// assert!(w > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ssvi {
    /// Global correlation `rho in (-1, 1)`.
    pub rho: f64,
    /// The smoothing function `phi`.
    pub phi: Phi,
}

impl Ssvi {
    /// Creates a validated SSVI surface.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if `rho` is not finite.
    /// - [`ParamError::CorrelationOutOfRange`] if `|rho| >= 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::{Phi, Ssvi};
    ///
    /// assert!(Ssvi::new(-0.4, Phi::heston(1.0).unwrap()).is_ok());
    /// assert!(Ssvi::new(1.0, Phi::heston(1.0).unwrap()).is_err());
    /// ```
    pub fn new(rho: f64, phi: Phi) -> Result<Self, ParamError> {
        if !rho.is_finite() {
            return Err(ParamError::NonFinite { name: "rho" });
        }
        if rho.abs() >= 1.0 {
            return Err(ParamError::CorrelationOutOfRange { rho });
        }
        Ok(Self { rho, phi })
    }

    /// Total implied variance `w(k, theta)` from the SSVI form.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::{Phi, Ssvi};
    ///
    /// let ssvi = Ssvi::new(0.0, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
    /// // With rho = 0 and k = 0, w = theta/2 * (1 + sqrt(1)) = theta.
    /// assert!((ssvi.total_variance(0.0, 0.04) - 0.04).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn total_variance(&self, k: f64, theta: f64) -> f64 {
        let phi = self.phi.eval(theta);
        let pk = phi * k;
        let inner = (pk + self.rho).mul_add(pk + self.rho, 1.0 - self.rho * self.rho);
        (theta / 2.0) * (1.0 + self.rho * pk + inner.sqrt())
    }

    /// The raw SVI slice equal to this SSVI surface at fixed `theta`.
    ///
    /// Closed-form map (MATH.md §6):
    ///
    /// ```text
    /// a     = (theta/2) * (1 - rho^2)
    /// b     = theta * phi / 2
    /// rho   = rho
    /// m     = -rho / phi
    /// sigma = sqrt(1 - rho^2) / phi
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::NonPositiveTheta`] if `theta <= 0`, or any
    /// [`ParamError`] surfaced by [`RawSvi::new`] (which should not occur for
    /// a valid SSVI surface).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::{Phi, Ssvi};
    ///
    /// let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
    /// let raw = ssvi.slice_at(0.04).unwrap();
    /// // The raw slice reproduces the SSVI total variance.
    /// let direct = ssvi.total_variance(0.1, 0.04);
    /// assert!((raw.total_variance(0.1) - direct).abs() < 1e-12);
    /// ```
    pub fn slice_at(&self, theta: f64) -> Result<RawSvi, ParamError> {
        if theta <= 0.0 || !theta.is_finite() {
            return Err(ParamError::NonPositiveTheta { theta });
        }
        let phi = self.phi.eval(theta);
        if phi <= 0.0 || !phi.is_finite() {
            return Err(ParamError::InvalidPhiParameter {
                name: "phi(theta)",
                value: phi,
            });
        }
        let one_minus_rho2 = 1.0 - self.rho * self.rho;
        let a = (theta / 2.0) * one_minus_rho2;
        let b = theta * phi / 2.0;
        let m = -self.rho / phi;
        let sigma = one_minus_rho2.sqrt() / phi;
        RawSvi::new(a, b, self.rho, m, sigma)
    }

    /// Tests the SSVI sufficient no-butterfly-arbitrage condition at one
    /// `theta` (Gatheral & Jacquier 2014, Theorem 4.2).
    ///
    /// The slice at `theta` is free of butterfly arbitrage if both hold:
    ///
    /// ```text
    /// theta * phi      * (1 + |rho|) < 4
    /// theta * phi^2    * (1 + |rho|) <= 4
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::{Phi, Ssvi};
    ///
    /// let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
    /// assert!(ssvi.is_butterfly_free_at(0.04));
    /// ```
    #[must_use]
    pub fn is_butterfly_free_at(&self, theta: f64) -> bool {
        if theta <= 0.0 || !theta.is_finite() {
            return false;
        }
        let phi = self.phi.eval(theta);
        if phi <= 0.0 || !phi.is_finite() {
            return false;
        }
        let factor = 1.0 + self.rho.abs();
        let tp = theta * phi;
        (tp * factor < 4.0) && (tp * phi * factor <= 4.0)
    }

    /// Tests the SSVI sufficient no-butterfly condition across a set of
    /// `theta` values (Gatheral & Jacquier 2014, Theorem 4.2).
    ///
    /// Both `theta * phi` and `theta * phi^2` are increasing in `theta` for
    /// the supported smoothing functions, so checking the largest `theta`
    /// supplied is, in practice, sufficient; for robustness every value is
    /// tested.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::{Phi, Ssvi};
    ///
    /// let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
    /// assert!(ssvi.is_butterfly_free(&[0.01, 0.04, 0.09]));
    /// ```
    #[must_use]
    pub fn is_butterfly_free(&self, thetas: &[f64]) -> bool {
        thetas.iter().all(|&theta| self.is_butterfly_free_at(theta))
    }

    /// Tests the SSVI no-calendar-spread-arbitrage condition at one `theta`
    /// (Gatheral & Jacquier 2014, Theorem 4.1, condition (ii)).
    ///
    /// Given a non-decreasing `theta_t` curve, the surface is calendar-free
    /// if, for every `theta`,
    ///
    /// ```text
    /// 0 <= d(theta * phi)/d(theta)
    ///        <= (1/rho^2) * (1 + sqrt(1 - rho^2)) * phi
    /// ```
    ///
    /// The upper bound is `+infinity` when `rho = 0`. The derivative
    /// `d(theta*phi)/d(theta)` is evaluated by a central finite difference.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::{Phi, Ssvi};
    ///
    /// let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
    /// assert!(ssvi.is_calendar_free_at(0.04));
    /// ```
    #[must_use]
    pub fn is_calendar_free_at(&self, theta: f64) -> bool {
        if theta <= 0.0 || !theta.is_finite() {
            return false;
        }
        let h = (theta * 1e-6).max(1e-9);
        let theta_phi = |x: f64| x * self.phi.eval(x);
        let deriv = (theta_phi(theta + h) - theta_phi(theta - h)) / (2.0 * h);
        if deriv < -1e-12 {
            return false;
        }
        let rho2 = self.rho * self.rho;
        if rho2 < 1e-300 {
            // rho = 0: the upper bound is +infinity, only the lower bound binds.
            return true;
        }
        let phi = self.phi.eval(theta);
        let upper = (1.0 / rho2) * (1.0 + (1.0 - rho2).sqrt()) * phi;
        deriv <= upper + 1e-12
    }

    /// Tests the SSVI no-calendar condition across a set of `theta` values
    /// (Gatheral & Jacquier 2014, Theorem 4.1).
    ///
    /// Verifies condition (i) — `theta_t` non-decreasing — on the supplied
    /// sequence, then condition (ii) at each `theta`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::ssvi::{Phi, Ssvi};
    ///
    /// let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
    /// assert!(ssvi.is_calendar_free(&[0.01, 0.04, 0.09]));
    /// ```
    #[must_use]
    pub fn is_calendar_free(&self, thetas: &[f64]) -> bool {
        // Condition (i): theta_t non-decreasing in maturity order.
        for pair in thetas.windows(2) {
            if pair[1] < pair[0] - 1e-12 {
                return false;
            }
        }
        // Condition (ii): the slope bound at every theta.
        thetas.iter().all(|&theta| self.is_calendar_free_at(theta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_heston_validation() {
        assert!(Phi::heston(1.0).is_ok());
        assert!(Phi::heston(0.0).is_err());
        assert!(Phi::heston(-1.0).is_err());
        assert!(Phi::heston(f64::NAN).is_err());
    }

    #[test]
    fn phi_power_law_validation() {
        assert!(Phi::power_law(0.5, 0.5).is_ok());
        assert!(Phi::power_law(0.0, 0.5).is_err());
        assert!(Phi::power_law(0.5, 0.0).is_err());
        assert!(Phi::power_law(0.5, 1.0).is_err());
        assert!(Phi::power_law(f64::INFINITY, 0.5).is_err());
    }

    #[test]
    fn phi_eval_positive_and_decreasing() {
        for phi in [Phi::heston(1.0).unwrap(), Phi::power_law(0.5, 0.5).unwrap()] {
            let a = phi.eval(0.01);
            let b = phi.eval(0.04);
            let c = phi.eval(0.16);
            assert!(a > 0.0 && b > 0.0 && c > 0.0);
            assert!(a > b && b > c, "phi should be decreasing in theta");
        }
    }

    #[test]
    fn phi_power_law_golden() {
        // phi(1) = eta / (1^gamma * 2^(1-gamma)).
        let p = Phi::power_law(1.0, 0.5).unwrap();
        assert!((p.eval(1.0) - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn ssvi_new_validation() {
        let phi = Phi::heston(1.0).unwrap();
        assert!(Ssvi::new(-0.4, phi).is_ok());
        assert!(Ssvi::new(1.0, phi).is_err());
        assert!(Ssvi::new(f64::NAN, phi).is_err());
    }

    #[test]
    fn ssvi_total_variance_atm_zero_rho() {
        // rho = 0, k = 0: w = theta/2 * (1 + 1) = theta.
        let ssvi = Ssvi::new(0.0, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        assert!((ssvi.total_variance(0.0, 0.04) - 0.04).abs() < 1e-15);
    }

    #[test]
    fn slice_at_reproduces_total_variance() {
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        let raw = ssvi.slice_at(0.04).unwrap();
        for &k in &[-0.5, -0.1, 0.0, 0.1, 0.5] {
            let direct = ssvi.total_variance(k, 0.04);
            assert!((raw.total_variance(k) - direct).abs() < 1e-12, "k = {k}");
        }
    }

    #[test]
    fn slice_at_rejects_non_positive_theta() {
        let ssvi = Ssvi::new(-0.3, Phi::heston(1.0).unwrap()).unwrap();
        assert!(matches!(
            ssvi.slice_at(0.0),
            Err(ParamError::NonPositiveTheta { .. })
        ));
    }

    #[test]
    fn slice_at_heston_reproduces_total_variance() {
        let ssvi = Ssvi::new(0.2, Phi::heston(2.0).unwrap()).unwrap();
        let raw = ssvi.slice_at(0.09).unwrap();
        for &k in &[-0.4, 0.0, 0.4] {
            let direct = ssvi.total_variance(k, 0.09);
            assert!((raw.total_variance(k) - direct).abs() < 1e-12, "k = {k}");
        }
    }

    #[test]
    fn butterfly_free_holds_for_small_eta() {
        // Power-law corollary: eta*(1+|rho|) <= 2 implies butterfly-free.
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        assert!(ssvi.is_butterfly_free(&[0.01, 0.04, 0.09, 0.25]));
    }

    #[test]
    fn butterfly_violation_for_large_phi() {
        // A power-law phi with large eta gives huge phi near theta -> 0,
        // which violates Theorem 4.2 (theta*phi^2*(1+|rho|) > 4).
        let ssvi = Ssvi::new(0.5, Phi::power_law(20.0, 0.9).unwrap()).unwrap();
        assert!(!ssvi.is_butterfly_free_at(1e-3));
    }

    #[test]
    fn butterfly_free_rejects_bad_theta() {
        let ssvi = Ssvi::new(0.0, Phi::heston(1.0).unwrap()).unwrap();
        assert!(!ssvi.is_butterfly_free_at(0.0));
        assert!(!ssvi.is_butterfly_free_at(-1.0));
    }

    #[test]
    fn calendar_free_for_monotone_thetas() {
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        assert!(ssvi.is_calendar_free(&[0.01, 0.04, 0.09]));
    }

    #[test]
    fn calendar_arbitrage_for_decreasing_thetas() {
        let ssvi = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        assert!(!ssvi.is_calendar_free(&[0.09, 0.04, 0.01]));
    }

    #[test]
    fn calendar_free_at_zero_rho() {
        let ssvi = Ssvi::new(0.0, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        assert!(ssvi.is_calendar_free_at(0.04));
    }
}

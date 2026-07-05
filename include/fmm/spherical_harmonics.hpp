#pragma once
#include <complex>
#include <vector>

namespace fmm {

// Associated Legendre polynomials P_n^m(x) for 0 <= m <= n, evaluated at a
// single x via the standard three-term upward recurrence. Returns a flat
// vector indexed as P[n * (n+1) / 2 + m] for n = 0..n_max, m = 0..n
// (i.e. only m >= 0 is stored; negative-m values are recovered via the
// well-known symmetry relation when needed by the caller).
std::vector<double> associated_legendre(int n_max, double x);

// Index helper matching the packing used by associated_legendre.
inline int legendre_index(int n, int m) { return n * (n + 1) / 2 + m; }

// Complex spherical harmonic Y_n^m(theta, phi), fully normalized
// (orthonormal on the unit sphere):
//   Y_n^m(theta, phi) = sqrt((2n+1)/(4*pi) * (n-m)!/(n+m)!)
//                        * P_n^m(cos theta) * e^{i m phi}
// for -n <= m <= n. theta is the polar angle (from +z axis), phi the
// azimuthal angle.
std::complex<double> spherical_harmonic(int n, int m, double theta, double phi);

// All Y_n^m for n = 0..n_max, all valid m, evaluated at one (theta, phi).
// Returned as a flat vector; use `sh_index` to look up a given (n, m).
std::vector<std::complex<double>> spherical_harmonics_all(int n_max, double theta, double phi);

// Index into the flat array returned by spherical_harmonics_all: valid m
// ranges over [-n, n], so each degree n has (2n+1) orders, offset by n^2
// from the start of that degree's block.
inline int sh_index(int n, int m) { return n * n + (m + n); }

// --- Greengard-Rokhlin convention (RR-1115 / Acta Numerica 1997) ---
//
// Their eq. (34) defines a differently-normalized spherical harmonic:
//
//   Y_n^m(theta, phi) = sqrt((n-|m|)!/(n+|m|)!) * P_n^{|m|}(cos theta) * e^{i m phi}
//
// i.e. WITHOUT the sqrt((2n+1)/4pi) orthonormalization factor, and with the
// phase e^{i m phi} taken with SIGNED m (so negative orders differ from the
// orthonormal convention above by a factor of (-1)^m as well). The
// translation theorems (M2M and later M2L/L2L) are stated in this basis,
// so the multipole module uses these functions exclusively -- do not mix
// the two conventions within one expansion.
std::complex<double> spherical_harmonic_gr(int n, int m, double theta, double phi);

// All GR-convention Y_n^m for n = 0..n_max, indexed via sh_index.
std::vector<std::complex<double>> spherical_harmonics_gr_all(int n_max, double theta, double phi);

}  // namespace fmm
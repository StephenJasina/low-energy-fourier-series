#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <utility>

#include "fourier.h"

using namespace std;

const long double PI = 3.14159265358979323846L;

// Do one iteration of stochastic gradient descent. Note that the unordered_map
// coefficients is changed. eta_divisor controls how large of a step to take (a
// larger value means a smaller step).
void sgd_step(FourierSeries &series, vector<pair<int, int> > &sum_keys,
              const vector<pair<int, int> > &half_keys, long double h,
              long double p, long double eta_divisor = 10) {
  long double eta;

  // All of the derivatives must be stored at once to make the best gradient
  // approximation
  unordered_map<pair<int, int>, long double[2], FourierSeries::pair_hash>
      derivatives;

  // This quantity will be used to figure out how large of a step to take.
  long double max_derivative = 0;

  for (const auto &k : sum_keys) {
    if (k.first == 0 && k.second == 0) {
      continue;
    }

    complex<long double> scaled_hess_det_coef =
        series.hessian_determinant_coefficient(k) /
        (long double)((k.first * k.first + k.second * k.second) *
                      (k.first * k.first + k.second * k.second));

    // This loop calculates the derivatives of our objective function with
    // respect to p_z, q_z, r_z, and s_z for every valid choice of z.
    for (const auto &z : half_keys) {
      // Contribution to the derivatives due to each E.
      long double ddr_stretch = 0, dds_stretch = 0;

      // Auxillary variables to make notation better.
      complex<long double>
          at_k_minus_z = series.coefficients.count(pair<int, int>(
                             k.first - z.first, k.second - z.second))
                             ? series.coefficients[pair<int, int>(
                                   k.first - z.first, k.second - z.second)]
                             : 0,
          at_k_plus_z = series.coefficients.count(pair<int, int>(
                            k.first + z.first, k.second + z.second))
                            ? series.coefficients[pair<int, int>(
                                  k.first + z.first, k.second + z.second)]
                            : 0;

      complex<long double> stretch_multiplier =
          (long double)(k.first * z.second - k.second * z.first) *
          (k.first * z.second - k.second * z.first) * scaled_hess_det_coef;

      // Calculate the E_stretch derivatives.
      ddr_stretch =
          (stretch_multiplier * conj(at_k_minus_z + at_k_plus_z)).real();
      dds_stretch =
          (stretch_multiplier * conj(at_k_minus_z - at_k_plus_z)).imag();

      // Calculate the actual derivatives
      derivatives[z][0] += ddr_stretch;
      derivatives[z][1] += dds_stretch;
    }
  }

  // Each mat_multiplier# is to contain the value of a summation that appears
  // in every derivative of the outer product norm. Essentially, they are
  // calculated here to avoid having to recalculate the same value for every z
  // and for every derivative.
  long double mat_multiplier1 = 0, mat_multiplier2 = 0, mat_multiplier3 = 0;
  for (auto it = series.coefficients.cbegin(); it != series.coefficients.end();
       ++it) {
    mat_multiplier1 += it->first.first * it->first.first * norm(it->second);
    mat_multiplier2 += it->first.first * it->first.second * norm(it->second);
    mat_multiplier3 += it->first.second * it->first.second * norm(it->second);
  }
  mat_multiplier1 = 8 * PI * PI * (2 * PI * PI * mat_multiplier1 - 1);
  mat_multiplier2 = 32 * PI * PI * PI * PI * mat_multiplier2;
  mat_multiplier3 = 8 * PI * PI * (2 * PI * PI * mat_multiplier3 - 1);

  // This loop calculates the derivatives of our objective function with
  // respect to p_z, q_z, r_z, and s_z for every valid choice of z.
  for (const auto &z : half_keys) {
    // Contribution to the derivatives due to each E.
    long double ddr_mat = 0, dds_mat = 0;
    long double ddr_bend = 0, dds_bend = 0;
    long double ddr_height = 0, dds_height = 0;

    complex<long double> mat_multiplier =
                             (mat_multiplier1 * z.first * z.first +
                              mat_multiplier2 * z.first * z.second +
                              mat_multiplier3 * z.second * z.second) *
                             series.coefficients[z],
                         bend_multiplier =
                             32 * PI * PI * PI * PI * h * h *
                             (z.first * z.first + z.second * z.second) *
                             (z.first * z.first + z.second * z.second) *
                             series.coefficients[z],
                         height_multiplier =
                             2 * pow(h, -p) * series.coefficients[z];

    // Calculate the E_mat derivatives.
    ddr_mat += mat_multiplier.real();
    dds_mat += mat_multiplier.imag();

    // Calculate E_bend derivatives.
    ddr_bend += bend_multiplier.real();
    dds_bend += bend_multiplier.imag();

    // Calculate E_height derivatives.
    ddr_height +=
        (z.first != 0 || z.second != 0 ? 1 : 0.5) * height_multiplier.real();
    dds_height += height_multiplier.imag();

    // Calculate the actual derivatives
    derivatives[z][0] += ddr_mat + ddr_bend + ddr_height;
    derivatives[z][1] += dds_mat + dds_bend + dds_height;
  }

  // Find the largest magnitude of the partial derivatives, or 1 if all
  // derivatives are smaller.
  for (auto it = derivatives.cbegin(); it != derivatives.cend(); ++it) {
    for (size_t i = 0; i != 2; ++i) {
      if (abs(it->second[i]) > max_derivative) {
        max_derivative = abs(it->second[i]);
      }
    }
  }

  eta = 1 / max_derivative / eta_divisor;

  // Make the updates to the necessary coefficients.
  for (const auto &z : half_keys) {
    pair<int, int> neg_z = pair<int, int>(-z.first, -z.second);

    // These variables are notationally convenient.
    long double &ddr = derivatives[z][0], &dds = derivatives[z][1];

    series.coefficients[z] -=
        eta * complex<long double>(ddr, dds) / (long double)(2);

    if (z.first != 0 || z.second != 0) {
      series.coefficients[neg_z] -=
          eta * complex<long double>(ddr, -dds) / (long double)(2);
    }
  }
}

// Do many iterations of stochastic gradient descent. Repeat until either the
// objective function is less than max_e or until our learning rate is very
// small.
// The variable initial is the beginning value of eta_divisor. The variable
// initial is how much to multiply (divide) eta_divisor by when the current
// value is doing poorly (well).
void sgd(FourierSeries &series, long double h, long double p,
         bool verbose = false, long double max_e = 2, long double initial = 10,
         long double multiplier = 1.1) {
  auto best_coefficients = series.coefficients;

  // These key lists are made here so that we don't have to recreate them every
  // time we run an iteration.
  auto sum_keys = series.sum_keys();
  auto half_keys = series.half_keys();

  long double eta_divisor = initial,
              e = series.e_stretch() + series.e_mat() +
                  h * h * series.e_bend() + pow(h, -p) * series.e_height(),
              best_e = e;

  unsigned consecutive_correct = 0;

  if (verbose) {
    cout << "Initial e: " << e << '\n' << endl;
  }

  while (e > max_e && eta_divisor < 1000000) {
    if (verbose) {
      cout << "eta_divisor = " << eta_divisor << '\n';
    }

    sgd_step(series, sum_keys, half_keys, h, p, eta_divisor);

    long double e_stretch = series.e_stretch(), e_mat = series.e_mat(),
                e_bend = series.e_bend(), e_height = series.e_height();
    e = e_stretch + e_mat + h * h * e_bend + pow(h, -p) * e_height;

    if (verbose) {
      cout << "\tE_stretch = " << e_stretch << "\n\tE_mat = " << e_mat
           << "\n\tE_bend = " << e_bend << "\n\tE_height = " << e_height
           << "\n\tsum = " << e << endl;
    }

    if (best_e > e) {
      best_coefficients = series.coefficients;
      best_e = e;
      ++consecutive_correct;

      // If we get many correct steps in a row, our step size might be too
      // large, meaning we should decrease eta_divisor.
      if (consecutive_correct == 4) {
        eta_divisor /= multiplier;
        consecutive_correct = 0;
      }
    } else {
      eta_divisor *= multiplier;
      consecutive_correct = 0;
    }

    series.coefficients = best_coefficients;
  }
}

int main(int argc, char **argv) {
  unordered_map<pair<int, int>, complex<long double>, FourierSeries::pair_hash>
      coefficients;
  long double max_e, rho1, rho2, h, p;

  if (argc < 5 || argc > 6) {
    cout << "Usage: " << endl;
    exit(1);
  }

  if (argc == 6) {
    rho1 = stold(argv[1]);
    rho2 = stold(argv[2]);
    h = stold(argv[3]);
    p = stold(argv[4]);
    max_e = stold(argv[5]);
  } else {
    rho1 = stold(argv[1]);
    rho2 = stold(argv[2]);
    h = stold(argv[3]);
    p = stold(argv[4]);
    max_e = 1;
  }

  coefficients = FourierSeries::random_coefficients(rho1, rho2);
  auto series = FourierSeries(coefficients);
  ofstream ofs;
  ofs.precision(15);

  sgd(series, h, p, true, max_e, 1, 1.1);

  for (size_t i = 0; i != 80; ++i) cout << '=';
  cout << endl;

  cout.precision(15);
  long double e_stretch = series.e_stretch(), e_mat = series.e_mat(),
              e_bend = series.e_bend(), e_height = series.e_height(),
              e = e_stretch + e_mat + h * h * e_bend + pow(h, -p) * e_height;
  cout << "Final results:" << endl;
  cout << "\tE_stretch = " << e_stretch << "\n\tE_mat = " << e_mat
       << "\n\tE_bend = " << e_bend << "\n\tE_height = " << e_height
       << "\n\tsum = " << e << endl
       << endl;

  ofs.open("coefficients_" + to_string(rho1) + "_" + to_string(rho2) + ".txt");
  ofs << series << endl;
  ofs.close();

  return 0;
}

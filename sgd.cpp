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
void sgd_step(FourierSeries &series, vector<pair<int, int> > &keys,
              const vector<pair<int, int> > &non_neg_keys, long double h,
              long double p, long double eta_divisor = 10) {
  long double eta;

  random_shuffle(keys.begin(), keys.end());

  for (const auto &k : keys) {
    if (k.first == 0 && k.second == 0) {
      continue;
    }

    complex<long double> scaled_hess_det_coef =
        series.hessian_determinant_coefficient(k) *
        (long double)(series.coefficients.size() /
                      (2 * (k.first * k.first + k.second * k.second) *
                       (k.first * k.first + k.second * k.second)));

    // All of the derivatives must be stored at once to make the best gradient
    // approximation
    unordered_map<pair<int, int>, long double[4], FourierSeries::pair_hash>
        derivatives;

    // This quantity will be used to figure out how large of a step to take.
    long double max_derivative;

    // Each mat_multiplier# is to contain the value of a summation that appears
    // in every derivative of the outer product norm. Essentially, they are
    // calculated here to avoid having to recalculate the same value for every z
    // and for every derivative.
    long double mat_multiplier1 = 0, mat_multiplier2 = 0, mat_multiplier3 = 0;
    for (auto it = series.coefficients.cbegin();
         it != series.coefficients.end(); ++it) {
      mat_multiplier1 += it->first.first * it->first.first * norm(it->second);
      mat_multiplier2 += it->first.first * it->first.second * norm(it->second);
      mat_multiplier3 += it->first.second * it->first.second * norm(it->second);
    }
    mat_multiplier1 = PI * PI * (2 * PI * PI * mat_multiplier1 - 1) / 2;
    mat_multiplier2 = 2 * PI * PI * PI * PI * mat_multiplier2;
    mat_multiplier3 = PI * PI * (2 * PI * PI * mat_multiplier3 - 1) / 2;

    // This loop calculates the derivatives of our objective function with
    // respect to p_z, q_z, r_z, and s_z for every valid choice of z.
    for (const auto &z : non_neg_keys) {
      // Contribution to the derivatives due to E_stretch.
      long double ddp_stretch = 0, ddq_stretch = 0, ddr_stretch = 0,
                  dds_stretch = 0;

      // Contribution to the derivatives due to norms.
      long double ddp_mat = 0, ddq_mat = 0, ddr_mat = 0, dds_mat = 0;
      long double ddp_bend = 0, ddq_bend = 0, ddr_bend = 0, dds_bend = 0;
      long double ddp_height = 0, ddq_height = 0, ddr_height = 0,
                  dds_height = 0;

      // Auxillary variables to make notation better. The last two letters refer
      // to (p)lus and (m)inus and are read top coordinate then bottom
      // coordinate.
      pair<int, int> zpp = z, zpm = pair<int, int>(z.first, -z.second),
                     zmp = pair<int, int>(-z.first, z.second),
                     zmm = pair<int, int>(-z.first, -z.second);

      // Represents the set (i.e. no repeats) of {zpp, zpm, zmp, zmm}.
      vector<pair<int, int> > zs;
      if (z.first == 0) {
        if (z.second == 0) {
          zs = {zpp};
        } else {
          zs = {zpp, zpm};
        }
      } else {
        if (z.second == 0) {
          zs = {zpp, zmp};
        } else {
          zs = {zpp, zpm, zmp, zmm};
        }
      }

      for (const auto &j : zs) {
        complex<long double> stretch_multiplier = 0,
                             mat_multiplier =
                                 (mat_multiplier1 * j.first * j.first +
                                  mat_multiplier2 * j.first * j.second +
                                  mat_multiplier3 * j.second * j.second) *
                                 series.coefficients[j],
                             bend_multiplier =
                                 2 * PI * PI * PI * PI * h * h *
                                 (j.first * j.first + j.second * j.second) *
                                 (j.first * j.first + j.second * j.second) *
                                 series.coefficients[j],
                             height_multiplier =
                                 pow(h, -p) * series.coefficients[j];

        // This check is necessary since we don't want to accidentally increase
        // the size of the unordered_map coefficients.
        if (series.coefficients.count(
                pair<int, int>(k.first - j.first, k.second - j.second))) {
          stretch_multiplier =
              (long double)(k.first * j.second - k.second * j.first) *
              (k.first * j.second - k.second * j.first) * scaled_hess_det_coef *
              conj(series.coefficients[pair<int, int>(k.first - j.first,
                                                      k.second - j.second)]);
        }

        // Calculate the E_stretch derivatives.
        ddp_stretch += stretch_multiplier.real();
        ddq_stretch +=
            stretch_multiplier.imag() * ((j.second <= 0) - (j.second >= 0));
        ddr_stretch +=
            stretch_multiplier.imag() * ((j.first <= 0) - (j.first >= 0));
        dds_stretch += stretch_multiplier.real() *
                       ((j.first * j.second <= 0) - (j.first * j.second >= 0));

        // Calculate the E_mat derivatives.
        ddp_mat += mat_multiplier.real();
        ddq_mat += mat_multiplier.imag() * ((j.second <= 0) - (j.second >= 0));
        ddr_mat += mat_multiplier.imag() * ((j.first <= 0) - (j.first >= 0));
        dds_mat += mat_multiplier.real() *
                   ((j.first * j.second <= 0) - (j.first * j.second >= 0));

        // Calculate E_bend derivatives.
        ddp_bend += bend_multiplier.real();
        ddq_bend +=
            bend_multiplier.imag() * ((j.second <= 0) - (j.second >= 0));
        ddr_bend += bend_multiplier.imag() * ((j.first <= 0) - (j.first >= 0));
        dds_bend += bend_multiplier.real() *
                    ((j.first * j.second <= 0) - (j.first * j.second >= 0));

        // Calculate E_height derivatives.
        ddp_height += height_multiplier.real();
        ddq_height +=
            height_multiplier.imag() * ((j.second <= 0) - (j.second >= 0));
        ddr_height +=
            height_multiplier.imag() * ((j.first <= 0) - (j.first >= 0));
        dds_height += height_multiplier.real() *
                      ((j.first * j.second <= 0) - (j.first * j.second >= 0));
      }

      // Calculate the actual derivatives
      derivatives[z][0] = ddp_stretch + ddp_mat + ddp_bend + ddp_height;
      derivatives[z][1] = ddq_stretch + ddq_mat + ddq_bend + ddq_height;
      derivatives[z][2] = ddr_stretch + ddr_mat + ddr_bend + ddr_height;
      derivatives[z][3] = dds_stretch + dds_mat + dds_bend + dds_height;
    }

    // Find the largest magnitude of the partial derivatives, or 1 if all
    // derivatives are smaller.
    max_derivative = 0;
    for (auto it = derivatives.cbegin(); it != derivatives.cend(); ++it) {
      for (size_t i = 0; i != 4; ++i) {
        if (abs(it->second[i]) > max_derivative) {
          max_derivative = abs(it->second[i]);
        }
      }
    }

    eta = 1 / max_derivative / eta_divisor;

    // Make the updates to the necessary coefficients.
    for (const auto &z : non_neg_keys) {
      pair<int, int> zpp = z, zpm = pair<int, int>(z.first, -z.second),
                     zmp = pair<int, int>(-z.first, z.second),
                     zmm = pair<int, int>(-z.first, -z.second);

      // These variables are notationally convenient.
      long double &ddp = derivatives[z][0], &ddq = derivatives[z][1],
                  &ddr = derivatives[z][2], &dds = derivatives[z][3];

      series.coefficients[zpp] -=
          eta * complex<long double>(ddp - dds, -ddq - ddr) / (long double)(4);

      if (z.second != 0) {
        series.coefficients[zpm] -=
            eta * complex<long double>(ddp + dds, ddq - ddr) / (long double)(4);
      }

      if (z.first != 0) {
        series.coefficients[zmp] -=
            eta * complex<long double>(ddp + dds, -ddq + ddr) /
            (long double)(4);
      }

      if (z.first != 0 && z.second != 0) {
        series.coefficients[zmm] -=
            eta * complex<long double>(ddp - dds, ddq + ddr) / (long double)(4);
      }
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
  auto keys = series.keys();
  auto non_neg_keys = series.non_neg_keys();

  long double eta_divisor = initial, e = series.e_stretch() + series.e_mat(),
              best_e = e;

  unsigned consecutive_correct = 0;

  if (verbose) {
    cout << "Initial e: " << e << '\n' << endl;
  }

  while (e > max_e && eta_divisor < 1000000) {
    if (verbose) {
      cout << "eta_divisor = " << eta_divisor << '\n';
    }

    sgd_step(series, keys, non_neg_keys, h, p, eta_divisor);

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
      if (consecutive_correct == 1) {
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
  long double e_max, rho1, rho2, h, p;

  if (argc < 5 || argc > 6) {
    cout << "Usage: " << endl;
    exit(1);
  }

  if (argc == 6) {
    rho1 = stold(argv[1]);
    rho2 = stold(argv[2]);
    h = stold(argv[3]);
    p = stold(argv[4]);
    e_max = stold(argv[5]);
  } else {
    rho1 = stold(argv[1]);
    rho2 = stold(argv[2]);
    h = stoul(argv[3]);
    p = stoul(argv[4]);
    e_max = 1;
  }

  coefficients = FourierSeries::random_coefficients(rho1, rho2);
  auto series = FourierSeries(coefficients);
  ofstream ofs;
  ofs.precision(15);

  sgd(series, h, p, true, e_max, 10, 1.1);

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

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "fourier.h"

using namespace std;

const long double PI = 3.14159265358979323846L;

// Do one iteration of stochastic gradient descent. Note that the unordered_map
// coefficients is changed. eta_divisor controls how large of a step to take (a
// larger value means a smaller step).
void sgd_step(FourierSeries &series, vector<pair<int, int> > &keys,
              const vector<pair<int, int> > &non_neg_keys,
              long double eta_divisor = 10) {
  long double eta;

  random_shuffle(keys.begin(), keys.end());

  for (const auto &k : keys) {
    complex<long double> hess_det_coef =
        series.hessian_determinant_coefficient(k);

    // All of the derivatives must be stored at once to make the best gradient
    // approximation
    unordered_map<pair<int, int>, long double[4], FourierSeries::pair_hash>
        derivatives;

    // This quantity will be used to figure out how large of a step to take.
    long double max_derivative;

    // Each op_multiplier# is to contain the value of a summation that appears
    // in every derivative of the outer product norm. Essentially, they are
    // calculated here to avoid having to recalculate the same value for every z
    // and for every derivative.
    long double op_multiplier1 = 0, op_multiplier2 = 0, op_multiplier3 = 0;
    for (auto it = series.coefficients.cbegin();
         it != series.coefficients.end(); ++it) {
      op_multiplier1 += it->first.first * it->first.first * norm(it->second);
      op_multiplier2 += it->first.first * it->first.second * norm(it->second);
      op_multiplier3 += it->first.second * it->first.second * norm(it->second);
    }
    op_multiplier1 = PI * PI * (2 * PI * PI * op_multiplier1 - 1) / 2;
    op_multiplier2 = 4 * PI * PI * PI * PI * op_multiplier2;
    op_multiplier3 = PI * PI * (2 * PI * PI * op_multiplier3 - 1) / 2;

    // This loop calculates the derivatives of our objective function with
    // respect to p_z, q_z, r_z, and s_z for every valid choice of z.
    for (const auto &z : non_neg_keys) {
      // Contribution to the derivatives due to the outer product norm.
      long double ddp_op = 0, ddq_op = 0, ddr_op = 0, dds_op = 0;

      // Contribution to the derivatives due to E_stretch.
      long double ddp_stretch = 0, ddq_stretch = 0, ddr_stretch = 0,
                  dds_stretch = 0;

      // Auxillary variables to make notation better. The last two letters refer
      // to (p)lus and (m)inus, and are read top coordinate then bottom
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
        complex<long double> op_multiplier =
                                 (op_multiplier1 * j.first * j.first +
                                  op_multiplier2 * j.first * j.second +
                                  op_multiplier3 * j.second * j.second) *
                                 series.coefficients[j],
                             stretch_multiplier = 0;

        // This check is necessary since we don't want to accidentally increase
        // the size of the unordered_map coefficients.
        if (series.coefficients.count(
                pair<int, int>(k.first - j.first, k.second - j.second))) {
          stretch_multiplier =
              (long double)(k.first * j.second - k.second * j.first) *
              (k.first * j.second - k.second * j.first) / 2 /
              (k.first * k.first + k.second * k.second) *
              (k.first * k.first + k.second * k.second) * hess_det_coef *
              conj(series.coefficients[pair<int, int>(k.first - j.first,
                                                      k.second - j.second)]);
        }

        // Calculate the outer product derivatives.
        ddp_op += op_multiplier.real();
        ddq_op += op_multiplier.imag() * ((j.second <= 0) - (j.second >= 0));
        ddr_op += op_multiplier.imag() * ((j.first <= 0) - (j.first >= 0));
        dds_op += op_multiplier.imag() *
                  ((j.first * j.second <= 0) - (j.first * j.second >= 0));

        // Calculate the E_stretch derivatives.
        ddp_stretch += stretch_multiplier.real();
        ddq_stretch += (stretch_multiplier *
                        (long double)((j.second <= 0) - (j.second >= 0)))
                           .imag();
        ddr_stretch += (stretch_multiplier *
                        (long double)((j.first <= 0) - (j.first >= 0)))
                           .imag();
        dds_stretch +=
            (stretch_multiplier * (long double)((j.first * j.second <= 0) -
                                                (j.first * j.second >= 0)))
                .real();
      }

      // Calculate the actual derivatives
      derivatives[z][0] = ddp_stretch * series.coefficients.size() + ddp_op;
      derivatives[z][1] = ddq_stretch * series.coefficients.size() + ddq_op;
      derivatives[z][2] = ddr_stretch * series.coefficients.size() + ddr_op;
      derivatives[z][3] = dds_stretch * series.coefficients.size() + dds_op;
    }

    // Find the largest magnitude of the partial derivatives, or 1 if all
    // derivatives are smaller.
    max_derivative = 1;
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
      };
    }
  }
}

// Do many iterations of stochastic gradient descent. Repeat until either the
// objective function is less than max_e or until our learning rate is very
// small.
// The variable initial is the beginning value of eta_divisor. The variable
// initial is how much to multiply (divide) eta_divisor by when the current
// value is doing poorly (well).
void sgd(FourierSeries &series, bool verbose = false, long double max_e = 1,
         long double initial = 10, long double multiplier = 1.1) {
  auto best_coefficients = series.coefficients;

  // These key lists are made here so that we don't have to recreate them every
  // time we run an iteration.
  auto keys = series.keys();
  auto non_neg_keys = series.non_neg_keys();

  long double eta_divisor = initial,
              e = series.e_stretch() + series.matrix_norm(), best_e = e;

  unsigned consecutive_correct = 0;

  if (verbose) {
    cout << "Initial e: " << e << '\n' << endl;
  }

  while (e > max_e && eta_divisor < 1000000) {
    if (verbose) {
      cout << "eta_divisor = " << eta_divisor << '\n';
    }

    sgd_step(series, keys, non_neg_keys, eta_divisor);

    long double e_stretch = series.e_stretch(),
                matrix_norm = series.matrix_norm();
    e = e_stretch + matrix_norm;

    if (verbose) {
      cout << "E_stretch = " << e_stretch << "\nMatrix norm = " << matrix_norm
           << "\nSum = " << e << '\n'
           << endl;
    }

    if (best_e > e) {
      best_coefficients = series.coefficients;
      best_e = e;
      ++consecutive_correct;

      // If we get many correct steps in a row, our step size might be too
      // large, meaning we should decrease eta_divisor.
      if (consecutive_correct == 8) {
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
  long double e_max, rho1, rho2;

  if (argc == 0) {
    rho1 = 10;
    rho2 = 20;
    e_max = 2;
  } else if (argc == 4) {
    rho1 = stold(argv[1]);
    rho2 = stold(argv[2]);
    e_max = stold(argv[3]);
  } else if (argc == 3) {
    rho1 = stold(argv[1]);
    rho2 = stold(argv[2]);
    e_max = 1;
  } else {
    cout << "Usage: " << endl;
    exit(1);
  }

  coefficients = FourierSeries::random_coefficients(rho1, rho2);
  auto series = FourierSeries(coefficients);
  ofstream ofs;
  ofs.precision(15);

  sgd(series, true, e_max, 10, 1.1);

  for (size_t i = 0; i != 80; ++i) cout << '=';
  cout << endl;

  cout << "Final results:" << endl;
  cout << "\tE_stretch = " << series.e_stretch()
       << "\n\tMatrix norm = " << series.matrix_norm() << endl
       << endl;
  cout << series << endl;

  ofs.open("coefficients_" + to_string(rho1) + "_" + to_string(rho2) + ".txt");
  ofs << series << endl;
  ofs.close();

  return 0;
}

#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <utility>

#include "fourier.h"

using namespace std;

const long double PI = 3.14159265358979323846L;
const long double ID[2][2] = {{1, 0}, {0, 1}};

void d_e_stretch(const FourierSeries &series,
                 const vector<pair<int, int> > &sum_keys,
                 const vector<pair<int, int> > &half_keys,
                 unordered_map<pair<int, int>, long double[2],
                               FourierSeries::pair_hash> &derivatives) {
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
      // Auxillary variables to make notation better.
      complex<long double> at_k_minus_z =
                               series.coefficients.count(pair<int, int>(
                                   k.first - z.first, k.second - z.second))
                                   ? series.coefficients.at(
                                         pair<int, int>(k.first - z.first,
                                                        k.second - z.second))
                                   : 0,
                           at_k_plus_z =
                               series.coefficients.count(pair<int, int>(
                                   k.first + z.first, k.second + z.second))
                                   ? series.coefficients.at(
                                         pair<int, int>(k.first + z.first,
                                                        k.second + z.second))
                                   : 0;

      complex<long double> stretch_multiplier =
          (long double)(k.first * z.second - k.second * z.first) *
          (k.first * z.second - k.second * z.first) * scaled_hess_det_coef;

      // Calculate the E_stretch derivatives.
      derivatives[z][0] +=
          (stretch_multiplier * conj(at_k_minus_z + at_k_plus_z)).real();
      derivatives[z][1] +=
          (stretch_multiplier * conj(at_k_minus_z - at_k_plus_z)).imag();
    }
  }
}

void d_e_mat(const FourierSeries &series,
             const vector<pair<int, int> > &half_keys,
             unordered_map<pair<int, int>, long double[2],
                           FourierSeries::pair_hash> &derivatives,
             const long double M[2][2]) {
  // Each mat_multiplier# is to contain the value of a summation that appears in
  // every derivative of the outer product norm. Essentially, they are
  // calculated here to avoid having to recalculate the same value for every z
  // and for every derivative.
  long double mat_multiplier1 = 0, mat_multiplier2 = 0, mat_multiplier3 = 0;
  for (auto it = series.coefficients.cbegin(); it != series.coefficients.end();
       ++it) {
    mat_multiplier1 += it->first.first * it->first.first * norm(it->second);
    mat_multiplier2 += it->first.first * it->first.second * norm(it->second);
    mat_multiplier3 += it->first.second * it->first.second * norm(it->second);
  }
  mat_multiplier1 = 8 * PI * PI * (2 * PI * PI * mat_multiplier1 - M[0][0]);
  mat_multiplier2 = 16 * PI * PI * (2 * PI * PI * mat_multiplier2 - M[0][1]);
  mat_multiplier3 = 8 * PI * PI * (2 * PI * PI * mat_multiplier3 - M[1][1]);

  for (const auto &z : half_keys) {
    complex<long double> mat_multiplier =
        (mat_multiplier1 * z.first * z.first +
         mat_multiplier2 * z.first * z.second +
         mat_multiplier3 * z.second * z.second) *
        series.coefficients.at(z);

    derivatives[z][0] += mat_multiplier.real();
    derivatives[z][1] += mat_multiplier.imag();
  }
}

void d_e_bend(const FourierSeries &series,
              const vector<pair<int, int> > &half_keys, long double h,
              unordered_map<pair<int, int>, long double[2],
                            FourierSeries::pair_hash> &derivatives) {
  for (const auto &z : half_keys) {
    complex<long double> bend_multiplier =
        32 * PI * PI * PI * PI * h * h *
        (z.first * z.first + z.second * z.second) *
        (z.first * z.first + z.second * z.second) * series.coefficients.at(z);

    derivatives[z][0] += bend_multiplier.real();
    derivatives[z][1] += bend_multiplier.imag();
  }
}

void d_e_height(const FourierSeries &series,
                const vector<pair<int, int> > &half_keys, long double h,
                long double p,
                unordered_map<pair<int, int>, long double[2],
                              FourierSeries::pair_hash> &derivatives) {
  for (const auto &z : half_keys) {
    complex<long double> height_multiplier =
        2 * pow(h, -p) * series.coefficients.at(z);

    derivatives[z][0] +=
        (z.first != 0 || z.second != 0 ? 1 : 0.5) * height_multiplier.real();
    derivatives[z][1] += height_multiplier.imag();
  }
}

void take_step(FourierSeries &series,
               unordered_map<pair<int, int>, long double[2],
                             FourierSeries::pair_hash> &derivatives,
               long double eta_divisor = 1) {
  long double eta, max_derivative = 0;

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
  for (auto it = derivatives.cbegin(); it != derivatives.cend(); ++it) {
    // Conventient notation.
    auto z = it->first;
    pair<int, int> neg_z = pair<int, int>(-z.first, -z.second);
    long double &ddr = derivatives[z][0], &dds = derivatives[z][1];

    series.coefficients[z] -=
        eta * complex<long double>(ddr, dds) / (long double)(2);

    if (z.first != 0 || z.second != 0) {
      series.coefficients[neg_z] -=
          eta * complex<long double>(ddr, -dds) / (long double)(2);
    }
  }
}

// Do one iteration of stochastic gradient descent. Note that the
// unordered_map coefficients is changed. eta_divisor controls how large of
// a step to take (a larger value means a smaller step).
void sgd_step(FourierSeries &series, vector<pair<int, int> > &sum_keys,
              const vector<pair<int, int> > &half_keys,
              const long double M[2][2], long double h, long double p,
              long double eta_divisor = 1) {
  random_shuffle(sum_keys.begin(), sum_keys.end());

  for (const auto &k : sum_keys) {
    if (k.first == 0 && k.second == 0) {
      continue;
    }

    complex<long double> scaled_hess_det_coef =
        (long double)(sum_keys.size()) /
        ((k.first * k.first + k.second * k.second) *
         (k.first * k.first + k.second * k.second)) *
        series.hessian_determinant_coefficient(k);

    // All of the derivatives must be stored at once to make the best gradient
    // approximation
    unordered_map<pair<int, int>, long double[2], FourierSeries::pair_hash>
        derivatives;

    // This loop calculates the derivatives of our objective function with
    // respect to p_z, q_z, r_z, and s_z for every valid choice of z.
    for (const auto &z : half_keys) {
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
      derivatives[z][0] +=
          (stretch_multiplier * conj(at_k_minus_z + at_k_plus_z)).real();
      derivatives[z][1] +=
          (stretch_multiplier * conj(at_k_minus_z - at_k_plus_z)).imag();
    }

    d_e_mat(series, half_keys, derivatives, M);
    d_e_bend(series, half_keys, h, derivatives);
    d_e_height(series, half_keys, h, p, derivatives);

    take_step(series, derivatives, eta_divisor);
  }
}

// Do one iteration of  gradient descent. Note that the unordered_map
// coefficients is changed. eta_divisor controls how large of a step to take (a
// larger value means a smaller step).
void gd_step(FourierSeries &series, vector<pair<int, int> > &sum_keys,
             const vector<pair<int, int> > &half_keys,
             const long double M[2][2], long double h, long double p,
             long double eta_divisor = 10) {
  unordered_map<pair<int, int>, long double[2], FourierSeries::pair_hash>
      derivatives;

  d_e_stretch(series, sum_keys, half_keys, derivatives);
  d_e_mat(series, half_keys, derivatives, M);
  d_e_bend(series, half_keys, h, derivatives);
  d_e_height(series, half_keys, h, p, derivatives);

  take_step(series, derivatives, eta_divisor);
}

// Do many iterations of  gradient descent. Repeat until either the objective
// function is less than max_e or until our learning rate is very small. The
// variable initial is the beginning value of eta_divisor. The variable initial
// is how much to multiply (divide) eta_divisor by when the current value is
// doing poorly (well).
void gd(FourierSeries &series, long double h, long double p,
        bool verbose = false, long double max_e = 2, long double initial = 10,
        long double multiplier = 1.1, const long double M[2][2] = ID) {
  auto best_coefficients = series.coefficients;

  // These key lists are made here so that we don't have to recreate them every
  // time we run an iteration.
  auto sum_keys = series.sum_keys();
  auto half_keys = series.half_keys();

  long double eta_divisor = initial,
              e = series.e_stretch() + series.e_mat(ID) +
                  h * h * series.e_bend() + pow(h, -p) * series.e_height(),
              best_e = e;

  unsigned consecutive_correct = 0;

  if (verbose) {
    cout << "Initial e: " << e << '\n' << endl;
  }

  // Get a "good" start
  sgd_step(series, sum_keys, half_keys, M, h, p, eta_divisor);

  best_coefficients = series.coefficients;
  e = series.e_stretch() + series.e_mat(M) + h * h * series.e_bend() +
      pow(h, -p) * series.e_height();

  while (e > max_e && eta_divisor < 10000) {
    if (verbose) {
      cout << "eta_divisor = " << eta_divisor << '\n';
    }

    gd_step(series, sum_keys, half_keys, M, h, p, eta_divisor);

    long double e_stretch = series.e_stretch(), e_mat = series.e_mat(M),
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
  auto M = ID;

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

  gd(series, h, p, true, max_e, 10, 1.1, M);

  for (size_t i = 0; i != 80; ++i) cout << '=';
  cout << endl;

  cout.precision(15);
  long double e_stretch = series.e_stretch(), e_mat = series.e_mat(ID),
              e_bend = series.e_bend(), e_height = series.e_height(),
              e = e_stretch + e_mat + h * h * e_bend + pow(h, -p) * e_height;
  cout << "Final results:" << endl;
  cout << "\tE_stretch = " << e_stretch << "\n\tE_mat = " << e_mat
       << "\n\tE_bend = " << e_bend << "\n\tE_height = " << e_height
       << "\n\tsum = " << e << endl
       << endl;

  // ofs.open("coefficients_" + to_string(rho1) + "_" + to_string(rho2) +
  // ".txt");
  ofs.open("coefficients.txt");

  ofs << "# ";
  for (int i = 0; i != argc; ++i) {
    ofs << argv[i] << ' ';
  }
  ofs << "{{" << M[0][0] << ", " << M[0][1] << "}, {" << M[1][0] << ", "
      << M[1][1] << "}}" << endl;

  ofs << "Final results:" << endl << "\tE_stretch = " << e_stretch << "\n\tE_mat = " << e_mat
       << "\n\tE_bend = " << e_bend << "\n\tE_height = " << e_height
       << "\n\tsum = " << e << endl
       << endl;

  ofs << series << endl;
  ofs.close();

  return 0;
}

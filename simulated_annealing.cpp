#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fourier.h"

using namespace std;

const complex<long double> ONE(1), MINUS_ONE(-1), I(0, 1), MINUS_I(0, -1);
const complex<long double> OFFSETS[4][4] = {{ONE, ONE, ONE, ONE},
                                            {MINUS_I, I, MINUS_I, I},
                                            {MINUS_I, MINUS_I, I, I},
                                            {MINUS_ONE, ONE, ONE, MINUS_ONE}};

void simulated_annealing(unordered_map<pair<int, int>, complex<long double>,
                                       FourierSeries::pair_hash> &coefficients,
                         long double c_norm = 1, size_t iterations = 1000,
                         bool verbose = false) {
  auto &best_coefficients = coefficients;
  auto best_series = FourierSeries(best_coefficients);
  long double best_e =
      best_series.e_stretch() + c_norm * best_series.e_mat();

  vector<pair<int, int> > non_neg_keys = best_series.non_neg_keys();

  mt19937 twister(chrono::system_clock::now().time_since_epoch().count());
  uniform_int_distribution<size_t> key_picker(0, non_neg_keys.size() - 1);
  uniform_int_distribution<size_t> category_picker(0, 3);
  uniform_real_distribution<long double> uniform;

  for (size_t t = 0; t != iterations; ++t) {
    long double T = 1 - (long double)t / iterations;
    auto new_coefficients = best_coefficients;

    for (size_t i = 0; i != (size_t)sqrt(best_coefficients.size()); ++i) {
      auto k = non_neg_keys[key_picker(twister)];
      int k1 = k.first, k2 = k.second;
      size_t category = category_picker(twister);
      long double magnitude =
          (uniform(twister) - 0.5) * 10 * T * T / best_coefficients.size();

      new_coefficients[k] += OFFSETS[category][0] * magnitude;
      new_coefficients[pair<int, int>(k1, -k2)] +=
          OFFSETS[category][1] * magnitude;
      new_coefficients[pair<int, int>(-k1, k2)] +=
          OFFSETS[category][2] * magnitude;
      new_coefficients[pair<int, int>(-k1, -k2)] +=
          OFFSETS[category][3] * magnitude;
    }

    auto new_series = FourierSeries(new_coefficients);
    long double new_e =
        new_series.e_stretch() + c_norm * new_series.e_mat();

    if (verbose) {
      // cout << "T = " << T << ":\t"
      cout << best_series.e_stretch() << '\t' << best_series.e_mat()
           << endl;
    }

    if (new_e < best_e) {
      best_coefficients = new_coefficients;
      best_series = new_series;
      best_e = new_e;
    }
  }
}

int main() {
  auto coefficients = FourierSeries::random_coefficients(6, 8);
  simulated_annealing(coefficients, 1, 40 * coefficients.size(), true);
  auto series = FourierSeries(coefficients);

  cout << series.e_stretch() << endl;
  cout << series.e_mat() << endl;
  cout << series << endl;
}

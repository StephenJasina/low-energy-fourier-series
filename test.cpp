#include <complex>
#include <iostream>
#include <unordered_map>
#include <utility>

#include "fourier.h"

using namespace std;

int main() {
  std::unordered_map<std::pair<int, int>, std::complex<long double>,
                     FourierSeries::pair_hash>
      coefficients(
          {{pair<int, int>(1, 2),
            complex<long double>(0.3235797047275578, 0.6287873748493056)},
           {pair<int, int>(1, -2),
            complex<long double>(0.3122900989983116, -0.1724455909684981)},
           {pair<int, int>(-1, 2),
            complex<long double>(0.3122900989983116, 0.1724455909684981)},
           {pair<int, int>(-1, -2),
            complex<long double>(0.3235797047275578, -0.6287873748493056)},
           {pair<int, int>(3, 0),
            complex<long double>(0.2411864650260896, 1.2750052238366405)},
           {pair<int, int>(-3, 0),
            complex<long double>(0.2411864650260896, -1.2750052238366405)},
           {pair<int, int>(2, 1),
            complex<long double>(-0.04845896843079334, 0.07883355148696393)},
           {pair<int, int>(2, -1),
            complex<long double>(-0.26297777641099135, 0.06402041838718318)},
           {pair<int, int>(-2, 1),
            complex<long double>(-0.26297777641099135, -0.06402041838718318)},
           {pair<int, int>(-2, -1),
            complex<long double>(-0.04845896843079334, -0.07883355148696393)},
           {pair<int, int>(2, 0),
            complex<long double>(0.9573505832663471, 0.7440599487145432)},
           {pair<int, int>(-2, 0),
            complex<long double>(0.9573505832663471, -0.7440599487145432)},
           {pair<int, int>(2, 2),
            complex<long double>(0.05952078718811665, 0.28257529370973483)},
           {pair<int, int>(2, -2),
            complex<long double>(0.20599731790941028, -0.15138822206559316)},
           {pair<int, int>(-2, 2),
            complex<long double>(0.20599731790941028, 0.15138822206559316)},
           {pair<int, int>(-2, -2),
            complex<long double>(0.05952078718811665, -0.28257529370973483)},
           {pair<int, int>(0, 3),
            complex<long double>(-0.5039321269420596, 0.5063981748474629)},
           {pair<int, int>(0, -3),
            complex<long double>(-0.5039321269420596, -0.5063981748474629)},
           {pair<int, int>(0, 2),
            complex<long double>(0.440309145730365, -0.6710658787226833)},
           {pair<int, int>(0, -2),
            complex<long double>(0.440309145730365, 0.6710658787226833)}});
  FourierSeries series(coefficients);

  std::cout.precision(16);

  cout << series.e_stretch() << endl << series.e_bend() << endl;
  cout << series.outer_product_integral()[0][0] << '\t'
       << series.outer_product_integral()[0][1] << '\n'
       << series.outer_product_integral()[1][0] << '\t'
       << series.outer_product_integral()[1][1] << endl;
  cout << series.at(pair<long double, long double>(0.1, 0.6)) << endl;
  cout << series.hessian_determinant(pair<long double, long double>(0.1, 0.6))
       << endl;
  cout << series.hessian_determinant_coefficient(pair<int, int>(2, 3)) << endl;
  cout << series << endl;

  return 0;
}

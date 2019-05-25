#include "fourier.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <iterator>
#include <random>
#include <unordered_map>
#include <utility>

using namespace std;

const long double PI = 3.14159265358979323846L;
const complex<long double> I(0, 1);

// Returns the dot product of the vectors that are represented by the pairs a
// and b.
long double dot(const pair<long double, long double> &a,
                const pair<long double, long double> &b) {
  return a.first * b.first + a.second * b.second;
}

size_t FourierSeries::pair_hash::operator()(const pair<int, int> &p) const {
  hash<int> h;

  auto hash1 = h(p.first);
  auto hash2 = h(p.second);

  // The bitwise XOR of the hashes is returned so that the hash is "more
  // independent" of either one of the values in the pairs.
  return hash1 ^ hash2;
}

FourierSeries::FourierSeries(
    const unordered_map<pair<int, int>, complex<long double>, pair_hash>
        &coefficients) {
  this->coefficients = coefficients;
}

ostream &operator<<(ostream &os, const FourierSeries &series) {
  auto coefficients = series.get_coefficients();
  auto it = coefficients.cbegin();

  os << '{';
  for (; it != coefficients.cend(); ++it) {
    os << '(' << it->first.first << ", " << it->first.second << "): ("
       << it->second.real();

    if (it->second.imag() < 0) {
      os << " - " << -it->second.imag();
    } else {
      os << " + " << it->second.imag();
    }
    os << "j)";

    if (next(it) != coefficients.end()) {
      os << ", ";
    }
  }
  os << '}';

  return os;
}

unordered_map<pair<int, int>, complex<long double>, FourierSeries::pair_hash>
FourierSeries::get_coefficients() const {
  return this->coefficients;
}

vector<pair<int, int> > FourierSeries::keys() const {
  vector<pair<int, int> > keys;

  // Reserve a bit of space to avoid having to reallocate space.
  keys.reserve(this->coefficients.size());

  for (auto it = this->coefficients.cbegin(); it != this->coefficients.cend();
       ++it) {
    keys.push_back(it->first);
  }

  return keys;
}

vector<pair<int, int> > FourierSeries::non_neg_keys() const {
  vector<pair<int, int> > keys;

  // Reserve a bit of space to avoid having to reallocate space too much. A
  // denominator of 3 (as opposed to 4) is chosen to account for the values
  // lying on the axes.
  keys.reserve(this->coefficients.size() / 3);

  for (auto it = this->coefficients.cbegin(); it != this->coefficients.cend();
       ++it) {
    if (it->first.first >= 0 && it->first.second >= 0) {
      keys.push_back(it->first);
    }
  }

  return keys;
}

long double FourierSeries::at(const pair<long double, long double> &x) const {
  complex<long double> result;

  for (auto it = this->coefficients.cbegin(); it != this->coefficients.cend();
       ++it) {
    result += it->second * exp(2 * PI * I * dot(it->first, x));
  }

  return result.real();
}

vector<vector<long double> > FourierSeries::outer_product_integral() const {
  vector<vector<long double> > m(2, vector<long double>(2));

  // First calculate the summations...
  for (auto it = this->coefficients.cbegin(); it != this->coefficients.cend();
       ++it) {
    m[0][0] += it->first.first * it->first.first * norm(it->second);
    m[0][1] += it->first.first * it->first.second * norm(it->second);
    m[1][1] += it->first.second * it->first.second * norm(it->second);
  }

  // ... then multiply by the appropriate leading constants.
  m[0][0] *= 4 * PI * PI;
  m[0][1] *= 4 * PI * PI;
  m[1][1] *= 4 * PI * PI;

  // This matrix is symmetric, so we save a little time by just setting these
  // two values to be equal manually.
  m[1][0] = m[0][1];

  return m;
}

long double FourierSeries::e_mat() const {
  auto m = this->outer_product_integral();

  // We take slight advantage of the fact that the matrix m is symmetric.
  return (1 - m[0][0] / 2) * (1 - m[0][0] / 2) + m[0][1] * m[0][1] / 2 +
         (1 - m[1][1] / 2) * (1 - m[1][1] / 2);
}

vector<vector<long double> > FourierSeries::hessian(
    const pair<long double, long double> &x) const {
  vector<vector<long double> > h(2, vector<long double>(2));

  // While the Hessian is ultimately real valued, some intermediate complex
  // values will need to be calculated.
  vector<vector<complex<long double> > > h_complex(
      2, vector<complex<long double> >(2));

  // First calculate the summations...
  for (auto it = this->coefficients.cbegin(); it != this->coefficients.cend();
       ++it) {
    h_complex[0][0] += (long double)(it->first.first * it->first.first) *
                       it->second * exp(2 * PI * I * dot(it->first, x));
    h_complex[0][1] += (long double)(it->first.first) * it->first.second *
                       it->second * exp(2 * PI * I * dot(it->first, x));
    h_complex[1][1] += (long double)(it->first.second * it->first.second) *
                       it->second * exp(2 * PI * I * dot(it->first, x));
  }

  // ... then multiply by the appropriate leading constants, and take the real
  // parts. Note that the imaginary parts are (or at least, should be) 0.
  h[0][0] = -4 * PI * PI * h_complex[0][0].real();
  h[0][1] = -4 * PI * PI * h_complex[0][1].real();
  h[1][1] = -4 * PI * PI * h_complex[1][1].real();

  // This matrix is symmetric, so we save a little time by just setting these
  // two values to be equal manually.
  h[1][0] = h[0][1];

  return h;
}

long double FourierSeries::hessian_determinant(
    const pair<long double, long double> &x) const {
  auto h = this->hessian(x);

  return h[0][0] * h[1][1] - h[0][1] * h[1][0];
}

complex<long double> FourierSeries::hessian_determinant_coefficient(
    const pair<int, int> &k) const {
  complex<long double> coefficient;

  for (auto it = this->coefficients.cbegin(); it != this->coefficients.cend();
       ++it) {
    auto other =
        pair<int, int>(k.first - it->first.first, k.second - it->first.second);
    if (coefficients.count(other)) {
      coefficient += (long double)it->first.second * other.first *
                     (k.first * it->first.second - k.second * it->first.first) *
                     it->second * coefficients.at(other);
    }
  }

  return 16 * PI * PI * PI * PI * coefficient;
}

long double FourierSeries::e_stretch() const {
  long double e = 0;
  unordered_map<pair<int, int>, complex<long double>, pair_hash> summands;

  // First, calculate the summand of the interior sum for every valid pair of j
  // and k.
  for (auto it1 = this->coefficients.cbegin(); it1 != this->coefficients.cend();
       ++it1) {
    for (auto it2 = this->coefficients.cbegin();
         it2 != this->coefficients.cend(); ++it2) {
      summands[pair<int, int>(it1->first.first + it2->first.first,
                              it1->first.second + it2->first.second)] +=
          (long double)it1->first.first * it2->first.second *
          (it1->first.first * it2->first.second -
           it1->first.second * it2->first.first) *
          it1->second * it2->second;
    }
  }

  // Next, combine these values to find e.
  for (auto it = summands.cbegin(); it != summands.cend(); ++it) {
    if (it->first.first != 0 || it->first.second != 0) {
      e += norm(it->second) /
           (dot(it->first, it->first) * dot(it->first, it->first));
    }
  }

  return 16 * PI * PI * PI * PI * e;
}

long double FourierSeries::e_bend() const {
  long double e = 0;

  for (auto it = this->coefficients.cbegin(); it != this->coefficients.cend();
       ++it) {
    e += norm(it->second) * dot(it->first, it->first) *
         dot(it->first, it->first);
  }

  return 16 * PI * PI * PI * PI * e;
}

unordered_map<pair<int, int>, complex<long double>, FourierSeries::pair_hash>
FourierSeries::random_coefficients(unsigned rho1, unsigned rho2) {
  mt19937 twister(chrono::system_clock::now().time_since_epoch().count());
  normal_distribution<double> normal;
  unordered_map<pair<int, int>, complex<long double>, pair_hash> coefficients;

  for (int k1 = 0; k1 != rho2; ++k1) {
    for (int k2 = 0; k2 != rho2; ++k2) {
      // Small tolerances have been put in to ensure that every coefficient one
      // would expect to have a value does indeed have a value (as there is a
      // chance for small rounding error with long doubles).
      if (rho1 * rho1 - 0.0001 <= k1 * k1 + k2 * k2 &&
          k1 * k1 + k2 * k2 <= rho2 * rho2 + 0.0001) {
        double p = normal(twister), q = normal(twister), r = normal(twister),
               s = normal(twister);

        // Ignore extraneous values in the case that we're on an axis.
        if (k1 == 0) {
          r = 0;
          s = 0;
        }

        if (k2 == 0) {
          q = 0;
          s = 0;
        }

        // Defining the coefficients like this ensures that the series will be
        // real valued.
        coefficients[pair<int, int>(k1, k2)] =
            complex<long double>(p - s, -q - r) / (long double)4;
        coefficients[pair<int, int>(k1, -k2)] =
            complex<long double>(p + s, q - r) / (long double)4;
        coefficients[pair<int, int>(-k1, k2)] =
            complex<long double>(p + s, -q + r) / (long double)4;
        coefficients[pair<int, int>(-k1, -k2)] =
            complex<long double>(p - s, q + r) / (long double)4;
      }
    }
  }

  return coefficients;
}

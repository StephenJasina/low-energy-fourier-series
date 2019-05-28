#ifndef FOURIER_H
#define FOURIER_H

#include <complex>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

// Represents a real-valued band-limited Fourier series w on the domain [0, 1] x
// [0, 1] that can be written as the sum of terms like a_k * exp(2 * pi * i *
// <k, x>). Here, k takes on all values in Z^2. All functions herein assume that
// sum is real-valued for all x in R^2.
class FourierSeries {
 public:
  // Structure that can hash a pair of ints. Used to create and use the
  // coefficients variable.
  struct pair_hash {
    std::size_t operator()(const std::pair<int, int> &) const;
  };

  // Holds the coefficients of this series. The value a_k can be accessed by
  // coefficients[k], where k here is represented as a pair of ints.
  std::unordered_map<std::pair<int, int>, std::complex<long double>,
                     FourierSeries::pair_hash>
      coefficients;

  // Constructor. Takes in an unordered_map that will represent the coefficients
  // of the Fourier series as described above the coefficients variable.
  FourierSeries(const std::unordered_map<
                std::pair<int, int>, std::complex<long double>, pair_hash> &);

  // Prints the coefficient list into the ostream that is passed in. The list is
  // formatted as a Python dictionary for easy plotting.
  friend std::ostream &operator<<(std::ostream &, const FourierSeries &);

  // Evaluates the Fourier series at a point. Since the series is assumed to be
  // real-valued, the series is evaluated and then the real part is returned.
  long double at(const std::pair<long double, long double> &) const;

  // Returns a copy of the unordered_map coefficients.
  std::unordered_map<std::pair<int, int>, std::complex<long double>, pair_hash>
  get_coefficients() const;

  // Returns the keys of the unordered_map coefficients. In other words, this
  // returns a superset of the support of the Fourier coefficients of our
  // series.
  std::vector<std::pair<int, int> > keys() const;

  // Returns the keys of the unordered_map coefficients that lie within the
  // first quadrant, inclusive of the axes. Since the series is assumed to be
  // real-valued, this set of keys could be used to generate the entire set of
  // keys by reflecting across the x and y axes.
  std::vector<std::pair<int, int> > half_keys() const;

  // Returns the integral of the outer product of the gradient of the series
  // with itself over the domain [0, 1] x [0, 1].
  std::vector<std::vector<long double> > outer_product_integral() const;

  // Returns the square of the Frobenius norm of the difference between the
  // identity matrix and half of the outer product integral.
  long double e_mat() const;

  long double e_height() const;

  // Returns the Hessian of the series at a point. Since the series is assumed
  // to be real-valued, the Hessian is evaluated and then the real part is
  // returned.
  std::vector<std::vector<long double> > hessian(
      const std::pair<long double, long double> &) const;

  // Returns the determinant of the Hessian at a point.
  long double hessian_determinant(
      const std::pair<long double, long double> &) const;

  // Returns the corresponding coefficient of the Fourier series of the Hessian
  // determinant.
  std::complex<long double> hessian_determinant_coefficient(
      const std::pair<int, int> &) const;

  long double e_stretch() const;

  long double e_bend() const;

  // Generates a random set of coefficients for a real-valued band-limited
  // Fourier series.
  static std::unordered_map<std::pair<int, int>, std::complex<long double>,
                            pair_hash>
  random_coefficients(long double, long double);
};

#endif

#ifndef MAT_OP_H
#define MAT_OP_H

#include <vector>				// std::vector<T>
#include <memory>				// std::unique_ptr<T[]>
#include <cassert>				// assert()
#include <iostream>				// std::ostream
#include <algorithm>			// std::max, std::min
#include <functional>			// std::function<>
#include <initializer_list>		// std::initializer_list

#define SAFE_GUARD true

namespace matrix_op
{
	// Notice the choice of non-type template parameter M, N
	// rather than constructor time argument.
	template<class T = float, int M = 1, int N = M>
	class Mat
	{
		protected:
			const int _size = M * N;
			int _lastIdx = 0;
			std::unique_ptr<T[]> _data = nullptr;
			inline int index2internal(const int& i, const int& j) const
			{
				return std::min(std::max(i * N + j, 0), _lastIdx);
			}
		public:
			// Named Constructor
			static Mat Empty(void)
			{
				return Mat({}); // Making assertion pop
			}
			static Mat Zeros(void)
			{
				return Mat(M * N);
			}
			static Mat Identity(void)
			{
				Mat tmp = Zeros();
				int K = std::min(M, N);
				for( int i = 0; i < K; i++ )
				{
					tmp(i, i) = T(1);
				}
				return tmp;
			}
			// Constructor
			explicit Mat(const int& inN)
				: _data(new T[inN]{}), _lastIdx(inN-1)	// uniform initialization since c++11
			{
				//
			}
			explicit Mat(const std::initializer_list<T>& inList = {})
				: _data(new T[inList.size()]), _lastIdx(inList.size()-1)
			{
				int i{0};
				for(auto const& it : inList )
				{
					_data[i++] = it;
				}
			}
			explicit Mat(const std::vector<T>& inVec)
				: _data(new T[inVec.size()]), _lastIdx(inVec.size()-1)
			{
				int i{0};
				memcpy(_data.get(), inVec.data(), size() * sizeof(T));
			}
			// Copy Constructor
			Mat(const Mat& inCopy)
				: _data(new T[inCopy.size()]), _lastIdx(inCopy._lastIdx)
			{
				memcpy(_data.get(), inCopy.get(), size() * sizeof(T));
			}
			// Operator overloading : assignment
			Mat& operator=(const Mat& inAssign)
			{
				memcpy(_data.get(), inAssign.get(), size() * sizeof(T));
				return *this;
			}

			// Operator overloading : data fetch
			T& operator[](const int& i)
			{
				#if SAFE_GUARD
					assert(size());
				#endif
				return _data[i];
			}
			const T& operator[](const int& i) const
			{
				#if SAFE_GUARD
					assert(size());
				#endif
				return _data[i];
			}

			// >> Functor
			T& operator()(const int& i, const int& j)
			{
				#if SAFE_GUARD
					assert(size());
				#endif
				return _data[index2internal(i, j)];
			}

			const T& operator()(const int& i, const int& j) const
			{
				#if SAFE_GUARD
					assert(size());
				#endif
				return operator[](index2internal(i, j));
			}

			// Member Functions
			inline size_t size(void) const
			{
				return _size;
			}
			T* get(const int& idx = 0)
			{
				return _data.get() + idx;
			}
			const T* get(const int& idx = 0) const
			{
				return _data.get() + idx;
			}

			// Math operations
			Mat<T, N, M> Inverse(void) const
			{
				if( N == M )
				{
					// Gauss-Jordan Method, direct transcription from pseudo-code in wikipedia
					return Gauss_Jordan_Method();
				}
				else
				{
					// Moore-Penrose Inverse
					const Mat& A(*this);
					return (A.t() * A).Inverse() * A.t();
				}
				// Not implement
				return Mat::Empty();
			}

			Mat<T, N, M> Mat<T, M, N>::Gauss_Jordan_Method(void) const;

			// Use operator overloading and type inference
			template<int K>
			auto operator*(const Mat<T, N, K>& inR) -> Mat<T, M, K>
			{
				// parallelization required
				Mat<T, M, K> ans = Mat<T, M, K>::Zeros();
				for( int i = 0; i < M; i++ )
				{
					for( int k = 0; k < K; k++ )
					{
						T sum{};
						for( int j = 0; j < N; j++ )
						{
							sum += (*this)(i, j) * inR(j, k);
						}
						ans(i, k) = sum;
					}
				}
				return ans;
				// return Mat<T, M, K>::Zeros();
			}

			Mat<T, N, M> transpose(void) const
			{
				std::vector<T> tmp(size());
				for( int i = 0; i < M; i++ )
				{
					for( int j = 0; j < N; j++ )
					{
						tmp[j*M+i] = operator()(i, j);
					}
				}
				return Mat<T, N, M>(tmp); // std::vector<T> to std::initializer_list<T>
			}

			// function alias using std::function and std::bind
			std::function<Mat<T, N, M>(void)> t = std::bind(&Mat::transpose, this);

			// Friend function
			friend std::ostream& operator<<(std::ostream& os, const Mat& out)
			{
				size_t idx(0);
				for(int i = 0; i < M; i++)
				{
					for(int j = 0; j < N; j++)
					{
						os << out[idx++] << " ";
					}
					os << std::endl;
				}
				return os;
			}
	};


	using Mat1x1 = Mat<float, 1, 1>;
	using Mat2x2 = Mat<float, 2, 2>;

	// template specialization
	template<>
	Mat1x1 Mat1x1::Inverse(void) const
	{
		return Mat1x1({1.f/(*this)[0]});
	}

	template<>
	Mat2x2 Mat2x2::Inverse(void) const
	{
		float a((*this)(0,0)), b((*this)(0,1)), c((*this)(1,0)), d((*this)(1,1));
		float det(a*d-b*c);
		#if SAFE_GUARD
			assert(det != 0.f);
		#endif
		return Mat2x2({d/det, -b/det, -c/det, a/det});
	}

	using Mat3x3 = Mat<float, 3, 3>;
	using Mat3x3d = Mat<double, 3, 3>;
}

#include "Mat_InverseMethods.hpp"

// + + + + + + + + ++ + + + + ++ + + + + ++ + + + + ++ + ++ + + + + +
// Agenda : Demonstrate and review various C++11 syntex and idioms
// Author : Aics Hwo(Jen-Huan Hu)
// Date : 2020.04.02 ~ 03
// + + + + + + + + ++ + + + + ++ + + + + ++ + + + + ++ + ++ + + + + +

#endif 

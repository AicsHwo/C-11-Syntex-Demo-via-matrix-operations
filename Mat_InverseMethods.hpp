namespace matrix_op
{
	template<class T, int M, int N>
	Mat<T, N, M> Mat<T, M, N>::Gauss_Jordan_Method(void) const
	{
		Mat<T, M, N> A = *this;
		Mat<T, N, M> inv = Mat<T, N, M>::Identity();
		
		// lambda function as local function since c++11
		auto argmax = [&, this](int h, int k) -> int
		{
			T _min = std::abs(A(h, k));
			int idx = h;
			for(int i = h+1; i < M; i++)
			{
				T abs_value = std::abs(A(i, k));
				if( _min > abs_value )
				{
					_min = abs_value;
					idx = i;
				}
			}
			return idx;
		};
		auto swap_rows = [&, this](int h, int i_max) -> void
		{
			for( int j = 0; j < N; j++ )
			{
				std::swap(A(h, j), A(i_max, j));
				std::swap(inv(h, j), inv(i_max, j));
			}
		};
		int h = 0;
		int k = 0;
		while( h <= M, k <= N )
		{
			int i_max = argmax(h, k);
			if( A(i_max, k) == T{} )
			{
				k = k+1;
			}
			else
			{
				swap_rows(h, i_max);
				// make pivot 1.
				T f = T(1) / A(h, h);
				for( int j = 0; j < N; j++ )
				{
					A(h, j) *= f;
					inv(h, j) *= f;
				}

				for( int i = h+1; i < M; i++ )
				{
					T f = A(i, k) / A(h, k);
					A(i, k) = 0;
					for( int j = k+1; j < N; j++ )
					{
						A(i, j) = A(i, j) - A(h, j) * f;
					}
					for( int j = 0; j < N; j++ )
					{
						inv(i, j) = inv(i, j) - inv(h, j) * f;
					}
				}
			}
			h++;
			k++;
		}

		// Back propagation
		for( int i = M-1; i >= 0; i-- )
		{
			for( int k = i-1; k >= 0; k-- )
			{
				T f = A(k, i) / A(i, i);
				for( int j = k; j < N; j++ )
				{
					A(k, j) -= A(i, j) * f;
				}
				for( int j = 0; j < N; j++ )
				{
					inv(k, j) -= inv(i, j) * f;
				}
			}
		}
		return inv;
	}
}

#include <iostream>

#include "Mat.h"


int main(int argc, char* argv[])
{
	using namespace matrix_op;
	Mat1x1 a({1.});
	std::cout << "Test Inverse of Mat1x1, as part of template specialization" << std::endl;
	std::cout << "a = \n" << a << std::endl;
	std::cout << "inv(a) = \n" << a.Inverse() << std::endl;
	std::cout << "a * inv(a) = \n" << a * a.Inverse() << std::endl;

	Mat2x2 b({1., 2., 3., 4.});
	std::cout << "Test Inverse of Mat2x2, as demo of template specialization" << std::endl;
	std::cout << "b = \n" << b << std::endl;
	std::cout << "inv(b) = \n" << b.Inverse() << std::endl;
	std::cout << "b * inv(b) = \n" << b * b.Inverse() << std::endl;

	std::cout << "Test Matrix Multiplication" << std::endl;
	Mat<float, 3, 2> c = Mat<float, 3, 2>::Zeros();
	Mat<float, 2, 5> d = Mat<float, 2, 5>::Zeros();
	std::cout << "c = \n" << c << std::endl;
	std::cout << "d = \n" << d << std::endl;
	std::cout << "c * d = \n" << c * d << std::endl;
	std::cout << "Comfirm operator= has been properly overloaded" << std::endl;
	c = Mat<float, 3, 2>{1,2,3,4,5,6};
	d = Mat<float, 2, 5>{-9,8,-7,6,-5,4,-3,2,-1,0};
	std::cout << "c = \n" << c << std::endl;
	std::cout << "d = \n" << d << std::endl;
	std::cout << "c * d = \n" << c * d << std::endl;

	std::cout << "Test transpose, also demo the function alias syntex" << std::endl;
	c = Mat<float, 3, 2>{1,2,3,4,5,6};
	std::cout << "c = \n" << c << std::endl;
	std::cout << "c.transpose() = \n" << c.transpose() << std::endl;
	std::cout << "c.t() = \n" << c.t() << std::endl;

	std::cout << "Test Martix Inverse by implementing Gauss-Jordan Method" << std::endl;
	Mat3x3d e{5,3,3,3,6,2,6,8,7};
	std::cout << "e = \n" << e << std::endl;
	std::cout << "inv(e) = \n" << e.Inverse() << std::endl;
	std::cout << "check e * inv(e) = \n" << e * e.Inverse() << std::endl;

	return 0;
}
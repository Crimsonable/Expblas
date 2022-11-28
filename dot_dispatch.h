#pragma once
#include "expbase.h"

namespace Expblas {
	template <typename DataType, Device device> struct DotEngine;

	template <typename DataType, Device device> struct BlasExcute;

	template <typename DataType> struct AdvanceEngine {
		template <Device device, int Tdim, int Ldim, int Rdim, bool transL,
			bool transR>
			inline static void eval(Tensor<DataType, Tdim, device> *dst,
				const DotExp<Tensor<DataType, Ldim, device>,
				Tensor<DataType, Rdim, device>, DataType,
				transL, transR> &exp) {
			DotEngine<DataType, device>::Execute(dst, exp);
		}
	};

	template <> struct BlasExcute<float, Device::CPU> {
		inline static void gemm(size_t m, size_t n, size_t k, float *A, size_t lda,
			bool transA, float *B, size_t ldb, bool transB,
			float *C, size_t ldc) {
#ifdef USE_MKL
			cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
				transB ? CblasTrans : CblasNoTrans, m, n, k, 1.0f, A, lda, B,
				ldb, 0.0f, C, ldc);
#endif // USE_MKL
		}

		inline static void gemv(size_t m, size_t n, float *A, size_t lda, bool transA,
			float *x, size_t incx, float *y, size_t incy) {
#ifdef USE_MKL
			cblas_sgemv(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, m, n, 1.0f,
				A, lda, x, incx, 0.0f, y, incy);
#endif // USE_MKL
		}

		static inline void BatchGemm(size_t m, size_t n, size_t k, size_t batch,
			float *A, size_t lda, bool transA, float *B,
			size_t ldb, bool transB, float *C, size_t ldc) {
#pragma omp parallel for
			for (int i = 0; i < batch; ++i) {
				BlasExcute<float, Device::CPU>::gemm(m, n, k, A + lda * i, lda, transA, B,
					ldb, transB, C + m * i, ldc);
			}
		}

		static inline void BatchGemv(size_t m, size_t n, size_t batch, float *A,
			size_t lda, bool transA, float *x, size_t incx,
			float *y, size_t incy) {
#pragma omp parallel for
			for (int i = 0; i < batch; ++i) {
				BlasExcute<float, Device::CPU>::gemv(m, n, A + lda * i, lda, transA, x,
					incx, y + incy * m * i, incy);
			}
		}
	};

	template <typename DataType, Device device> struct DotEngine {
		template <int Ldim, int Rdim, bool transL, bool transR>
		inline static void Execute(Tensor<DataType, Ldim, device> *dst,
			const DotExp<Tensor<DataType, Ldim, device>,
			Tensor<DataType, Rdim, device>,
			DataType, transL, transR> &exp) {
			auto dst_shape = dst->shape.flat2D();
			auto lhs_shape = exp.lhs.shape.flat2D();
			auto rhs_shape = exp.rhs.shape.flat2D();
			EXP_ASSERT(lhs_shape[1] == rhs_shape[0] && dst_shape[0] == lhs_shape[0] &&
				dst_shape[1] == rhs_shape[1],
				"Dot shapes don't match!");
			BlasExcute<DataType, device>::gemm(
				dst_shape[0], dst_shape[1], lhs_shape[1], exp.lhs.data->dataptr,
				exp.lhs.data->stride, transL, exp.rhs.data->dataptr,
				exp.rhs.data->stride, transR, dst->data->dataptr, dst->data->stride);
		}

		template <int Ldim, bool transL, bool transR>
		inline static void Execute(
			Tensor<DataType, 1, device> *dst,
			const DotExp<Tensor<DataType, Ldim, device>, Tensor<DataType, 1, device>,
			DataType, transL, transR> &exp) {
			auto dst_shape = dst->shape;
			auto lhs_shape = exp.lhs.shape.flat2D();
			EXP_ASSERT(lhs_shape[1] == exp.rhs.shape[0] && dst_shape[0] == lhs_shape[0],
				"Dot shapes don't match");
			BlasExcute<DataType, device>::gemv(
				lhs_shape[0], lhs_shape[1], exp.lhs.data->dataptr, exp.lhs.data->stride,
				transL, exp.rhs.data->dataptr, 1, dst->data->dataptr, 1);
		}
	};

} // namespace Expblas
#include <cstdio>
#ifndef __ARM_FEATURE_SME2
#error "kernel_sme2.cpp must be compiled with SME2 enabled (e.g. -march=armv9.2-a+sme2)"
#endif

#include "kernel.h"
#include "kernel_utils.h"
#include <iostream>
#include <arm_sme.h>

static void cactus_matmul_f16_sme2_worker(
	const __fp16* a,
	const __fp16* b_transposed,
	__fp16* c,
	size_t M,
	size_t K,
	size_t N,
	size_t start_row,
	size_t end_row
) __arm_streaming __arm_inout("za") {
	(void) M;
	if (start_row >= end_row) return;
	
	const size_t SVLw = svcntsw();

	const size_t TILE_M = SVLw;
	const size_t TILE_N = SVLw;

	std::memset(c, 0, M * N * sizeof(__fp16));
	return;

	// TODO: multi-vector / multi-tile operations
	for (size_t row = start_row; row < end_row; row += TILE_M) {
		const size_t active_rows = std::min(TILE_M, end_row - row);
		const svbool_t pM16 = svwhilelt_b16(0, active_rows);

		// TODO: no VLA
		__fp16 a_panel[K * TILE_M];
		std::memset(a_panel, 0, K * TILE_M * sizeof(__fp16));
		for (size_t k = 0; k < K; ++k) {
			for (size_t m = 0; m < active_rows; ++m) {
				a_panel[k * TILE_M + m] = a[(row + m) * K + k];
			}
		}

		for (size_t col = 0; col < N; col += TILE_N) {
			const size_t active_cols = std::min(TILE_N, N - col);
			const svbool_t pN16 = svwhilelt_b16(0, active_cols);
			const svbool_t pN32 = svwhilelt_b32((uint64_t) 0, (uint64_t) active_cols);


			// TODO: no VLA
			__fp16 b_panel[K * TILE_N];
			std::memset(b_panel, 0, K * TILE_N * sizeof(__fp16));
			for (size_t k = 0; k < K; ++k) {
				for (size_t n = 0; n < active_cols; ++n) {
					b_panel[k * TILE_N + n] = b_transposed[(col + n) * K + k];
				}
			}

			svzero_za();
			for (size_t k = 0; k < K; ++k) {
				const svfloat16_t zL = svld1(pM16, &a_panel[k * TILE_M]);
				const svfloat16_t zR = svld1(pN16, &b_panel[k * TILE_N]);
				svmopa_za32_f16_m(0, pM16, pN16, zL, zR);
			}

			
			for (size_t m = 0; m < active_rows; ++m) {
				const svfloat32_t outRow32 = svread_hor_za32_f32_m(svdup_n_f32(0.0f), pN32, 0, m);
				const svfloat16_t outRow16 = svcvt_f16_f32_z(pN32, outRow32);
				svst1(pN16, &c[(row + m) * N + col], outRow16);
			}
		}
	}
}

__arm_new("za") __arm_locally_streaming
void cactus_matmul_f16_sme2_caller(
	const __fp16* a,
	const __fp16* b_transposed,
	__fp16* c,
	size_t M,
	size_t K,
	size_t N,
	size_t start_row,
	size_t end_row
) {
	cactus_matmul_f16_sme2_worker(
		a, b_transposed, c,
		M, K, N,
		start_row, end_row
	);
}

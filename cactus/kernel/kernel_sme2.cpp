#ifndef __ARM_FEATURE_SME2
#error "kernel_sme2.cpp must be compiled with SME2 enabled (e.g. -march=armv9.2-a+sme2)"
#endif

#include "kernel.h"
#include "kernel_utils.h"
#include <vector>
#include <arm_sme.h>

static void cactus_matmul_f16_sme2_worker(
	const __fp16* a_transposed,
	const __fp16* b,
	__fp16* c,
	size_t M,
	size_t K,
	size_t N,
	size_t start_row,
	size_t end_row,
	size_t TILE_M,
	size_t TILE_N
) __arm_streaming __arm_inout("za") {
	(void) M;
	if (start_row >= end_row) return;
	
	// TODO: multi-vector / multi-tile operations
	for (size_t row = start_row; row < end_row; row += TILE_M) {
		const size_t active_rows = std::min(TILE_M, end_row - row);
		const svbool_t pM16 = svwhilelt_b16(0, active_rows);

		for (size_t col = 0; col < N; col += TILE_N) {
			const size_t active_cols = std::min(TILE_N, N - col);
			const svbool_t pN16 = svwhilelt_b16(0, active_cols);
			const svbool_t pN32 = svwhilelt_b32((uint64_t) 0, (uint64_t) active_cols);

			svzero_za();
			for (size_t k = 0; k < K; ++k) {
				const svfloat16_t zL = svld1(pM16, &a_transposed[k * M + row]);
				const svfloat16_t zR = svld1(pN16, &b[k * N + col]);
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
static void cactus_matmul_f16_sme2_thread_entry(
	const __fp16* a_transposed, const __fp16* b, __fp16* c,
	size_t M, size_t K, size_t N,
	size_t ROW_BLOCK_SIZE, size_t start_block, size_t end_block
) {
	const size_t TILE_M = svcntsw();
	const size_t TILE_N = TILE_M;
	
	for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
        size_t start_row = block_idx * ROW_BLOCK_SIZE;
		size_t end_row = std::min(start_row + ROW_BLOCK_SIZE, M);

		cactus_matmul_f16_sme2_worker(
			a_transposed, b, c,
			M, K, N,
			start_row, end_row,
			TILE_M, TILE_N
		);
    }
}

static inline void cactus_transpose_2d_f16_parallel(
	const __fp16* source,
	__fp16* destination,
	size_t num_rows,
	size_t num_cols
) {
	constexpr size_t TILE_ROWS = 32;
	const size_t num_row_blocks = (num_rows + TILE_ROWS - 1) / TILE_ROWS;

	CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
		[=](size_t start_block, size_t end_block) {
			for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
				const size_t start_row = block_idx * TILE_ROWS;
				const size_t end_row = std::min(start_row + TILE_ROWS, num_rows);
				cactus_transpose_2d_f16(source, destination, num_rows, num_cols, start_row, end_row);
			}
		});
}

__arm_new("za") __arm_locally_streaming
void cactus_matmul_f16_sme2_caller(
	const __fp16* a,
	const __fp16* b_transposed,
	__fp16* c,
	size_t M,
	size_t K,
	size_t N
) {

	std::vector<__fp16> aT_storage(K * M);
	std::vector<__fp16> b_storage(K * N);

	__fp16* a_T = aT_storage.data();
	__fp16* b = b_storage.data();

	cactus_transpose_2d_f16_parallel(a, a_T, M, K);
	cactus_transpose_2d_f16_parallel(b_transposed, b, N, K);

	const size_t TILE_M = svcntsw();

	constexpr size_t TILES_PER_THREAD = 4;
	const size_t ROW_BLOCK_SIZE = TILES_PER_THREAD * TILE_M;
    const size_t num_row_blocks = (M + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE;

    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
			cactus_matmul_f16_sme2_thread_entry(
				a_T, b, c,
				M, K, N,
				ROW_BLOCK_SIZE, start_block, end_block
			);
        });
}

#include <arm_sve.h>
#include <cstdint>
#ifndef __ARM_FEATURE_SME2
#error "kernel_sme2.cpp must be compiled with SME2 enabled (e.g. -march=armv9.2-a+sme2)"
#endif

#include "kernel.h"
#include "kernel_utils.h"
#include <arm_sme.h>
#include <algorithm>

static void cactus_pack_a_f16(
    const __fp16* a,
    __fp16* a_packed,
    size_t M,
    size_t K,
    size_t tile_rows,
    size_t tile_pairs
) {
    const size_t row_blocks = (M + tile_rows - 1) / tile_rows;
    const size_t k_pairs = (K + 1) / 2;
    const size_t block_stride = k_pairs * tile_pairs;

    std::fill(a_packed, a_packed + row_blocks * block_stride, static_cast<__fp16>(0));

    CactusThreading::parallel_for(row_blocks * k_pairs, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t rb = idx / k_pairs;
                const size_t kp = idx % k_pairs;

                const size_t row0 = rb * tile_rows;
                const size_t active_r = (row0 < M) ? std::min(tile_rows, M - row0) : 0;
                if (active_r == 0) continue;

                const size_t k0 = kp * 2;
                const size_t k1 = k0 + 1;

                __fp16* dst = a_packed + rb * block_stride + kp * tile_pairs;

                for (size_t r = 0; r < active_r; ++r) {
                    const size_t src_row = row0 + r;
                    dst[2 * r] = a[src_row * K + k0];
                    dst[2 * r + 1] = (k1 < K) ? a[src_row * K + k1] : static_cast<__fp16>(0);
                }
            }
        });
}

static void cactus_pack_b_f16_from_bt(
    const __fp16* b_transposed,
    __fp16* b_packed,
    size_t K,
    size_t N,
    size_t tile_cols,
    size_t tile_pairs
) {
    const size_t k_pairs = (K + 1) / 2;
    const size_t col_blocks = (N + tile_cols - 1) / tile_cols;

    std::fill(b_packed, b_packed + k_pairs * col_blocks * tile_pairs, static_cast<__fp16>(0));

    CactusThreading::parallel_for(k_pairs * col_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t kp = idx / col_blocks;
                const size_t cb = idx % col_blocks;

                const size_t col0 = cb * tile_cols;
                const size_t active_c = (col0 < N) ? std::min(tile_cols, N - col0) : 0;
                if (active_c == 0) continue;

                const size_t k0 = kp * 2;
                const size_t k1 = k0 + 1;

                __fp16* dst = b_packed + (kp * col_blocks + cb) * tile_pairs;

                for (size_t c = 0; c < active_c; ++c) {
                    const size_t n = col0 + c;
                    dst[2 * c] = b_transposed[n * K + k0];
                    dst[2 * c + 1] = (k1 < K) ? b_transposed[n * K + k1] : static_cast<__fp16>(0);
                }
            }
        });
}

static void cactus_matmul_f16_sme2_worker(
    const __fp16* a_packed,
    const __fp16* b_packed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row,
    size_t tile_rows,
    size_t tile_pairs,
    float* tmp_pair
) __arm_streaming __arm_inout("za") {
    if (start_row >= end_row) return;
    (void)M;

    const size_t k_pairs = (K + 1) / 2;
    const size_t col_blocks = (N + tile_rows - 1) / tile_rows;

    const size_t a_row_block_stride = k_pairs * tile_pairs;

    const svbool_t pNh_full = svptrue_b16();
    const svbool_t pN32_full = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(tile_rows));
    const svcount_t pNh_full_c = svptrue_c16();
    const svcount_t pOut16_full_c = svwhilelt_c16(static_cast<uint64_t>(0), static_cast<uint64_t>(tile_rows), 2);

    for (size_t row = start_row; row < end_row; row += tile_rows) {
        const size_t rb = row / tile_rows;
        const size_t active_r = std::min(tile_rows, end_row - row);
        const svbool_t pMh = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_r * 2));

        size_t cb = 0;

        for (; cb + 3 < col_blocks && (cb + 4) * tile_rows <= N; cb += 4) {
            svzero_za();

            size_t kp = 0;
            for (; kp + 1 < k_pairs; kp += 2) {
                const __fp16* a_ptr0 = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr0 = b_packed + (kp * col_blocks + cb) * tile_pairs;
                const __fp16* a_ptr1 = a_ptr0 + tile_pairs;
                const __fp16* b_ptr1 = b_ptr0 + col_blocks * tile_pairs;

                const svfloat16_t zA0 = svld1(pMh, a_ptr0);
                const svfloat16x4_t zB40 = svld1_f16_x4(pNh_full_c, b_ptr0);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA0, svget4(zB40, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA0, svget4(zB40, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA0, svget4(zB40, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA0, svget4(zB40, 3));

                const svfloat16_t zA1 = svld1(pMh, a_ptr1);
                const svfloat16x4_t zB41 = svld1_f16_x4(pNh_full_c, b_ptr1);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA1, svget4(zB41, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA1, svget4(zB41, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA1, svget4(zB41, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA1, svget4(zB41, 3));
            }

            for (; kp < k_pairs; ++kp) {
                const __fp16* a_ptr = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr = b_packed + (kp * col_blocks + cb) * tile_pairs;

                const svfloat16_t zA = svld1(pMh, a_ptr);
                const svfloat16x4_t zB4 = svld1_f16_x4(pNh_full_c, b_ptr);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA, svget4(zB4, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA, svget4(zB4, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA, svget4(zB4, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA, svget4(zB4, 3));
            }

            const size_t col = cb * tile_rows;

            size_t trow = 0;
            for (; trow + 3 < active_r; trow += 4) {
                for (size_t u = 0; u < 4; ++u) {
                    const size_t tr = trow + u;

                    svst1_hor_za32(0, static_cast<uint32_t>(tr), pN32_full, tmp_pair);
                    svst1_hor_za32(1, static_cast<uint32_t>(tr), pN32_full, tmp_pair + tile_rows);
                    svfloat32_t out0_32 = svld1(pN32_full, tmp_pair);
                    svfloat32_t out1_32 = svld1(pN32_full, tmp_pair + tile_rows);
                    svfloat16_t out0 = svcvt_f16_f32_z(pN32_full, out0_32);
                    svfloat16_t out1 = svcvt_f16_f32_z(pN32_full, out1_32);
                    out0 = svuzp1_f16(out0, out0);
                    out1 = svuzp1_f16(out1, out1);
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col], svcreate2(out0, out0));
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col + tile_rows], svcreate2(out1, out1));

                    svst1_hor_za32(2, static_cast<uint32_t>(tr), pN32_full, tmp_pair);
                    svst1_hor_za32(3, static_cast<uint32_t>(tr), pN32_full, tmp_pair + tile_rows);
                    svfloat32_t out2_32 = svld1(pN32_full, tmp_pair);
                    svfloat32_t out3_32 = svld1(pN32_full, tmp_pair + tile_rows);
                    svfloat16_t out2 = svcvt_f16_f32_z(pN32_full, out2_32);
                    svfloat16_t out3 = svcvt_f16_f32_z(pN32_full, out3_32);
                    out2 = svuzp1_f16(out2, out2);
                    out3 = svuzp1_f16(out3, out3);
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col + 2 * tile_rows], svcreate2(out2, out2));
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col + 3 * tile_rows], svcreate2(out3, out3));
                }
            }

            for (; trow + 1 < active_r; trow += 2) {
                for (size_t u = 0; u < 2; ++u) {
                    const size_t tr = trow + u;

                    svst1_hor_za32(0, static_cast<uint32_t>(tr), pN32_full, tmp_pair);
                    svst1_hor_za32(1, static_cast<uint32_t>(tr), pN32_full, tmp_pair + tile_rows);
                    svfloat32_t out0_32 = svld1(pN32_full, tmp_pair);
                    svfloat32_t out1_32 = svld1(pN32_full, tmp_pair + tile_rows);
                    svfloat16_t out0 = svcvt_f16_f32_z(pN32_full, out0_32);
                    svfloat16_t out1 = svcvt_f16_f32_z(pN32_full, out1_32);
                    out0 = svuzp1_f16(out0, out0);
                    out1 = svuzp1_f16(out1, out1);
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col], svcreate2(out0, out0));
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col + tile_rows], svcreate2(out1, out1));

                    svst1_hor_za32(2, static_cast<uint32_t>(tr), pN32_full, tmp_pair);
                    svst1_hor_za32(3, static_cast<uint32_t>(tr), pN32_full, tmp_pair + tile_rows);
                    svfloat32_t out2_32 = svld1(pN32_full, tmp_pair);
                    svfloat32_t out3_32 = svld1(pN32_full, tmp_pair + tile_rows);
                    svfloat16_t out2 = svcvt_f16_f32_z(pN32_full, out2_32);
                    svfloat16_t out3 = svcvt_f16_f32_z(pN32_full, out3_32);
                    out2 = svuzp1_f16(out2, out2);
                    out3 = svuzp1_f16(out3, out3);
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col + 2 * tile_rows], svcreate2(out2, out2));
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col + 3 * tile_rows], svcreate2(out3, out3));
                }
            }

            for (; trow < active_r; ++trow) {
                svst1_hor_za32(0, static_cast<uint32_t>(trow), pN32_full, tmp_pair);
                svst1_hor_za32(1, static_cast<uint32_t>(trow), pN32_full, tmp_pair + tile_rows);
                svfloat32_t out0_32 = svld1(pN32_full, tmp_pair);
                svfloat32_t out1_32 = svld1(pN32_full, tmp_pair + tile_rows);
                svfloat16_t out0 = svcvt_f16_f32_z(pN32_full, out0_32);
                svfloat16_t out1 = svcvt_f16_f32_z(pN32_full, out1_32);
                out0 = svuzp1_f16(out0, out0);
                out1 = svuzp1_f16(out1, out1);
                svst1_f16_x2(pOut16_full_c, &c[(row + trow) * N + col], svcreate2(out0, out0));
                svst1_f16_x2(pOut16_full_c, &c[(row + trow) * N + col + tile_rows], svcreate2(out1, out1));

                svst1_hor_za32(2, static_cast<uint32_t>(trow), pN32_full, tmp_pair);
                svst1_hor_za32(3, static_cast<uint32_t>(trow), pN32_full, tmp_pair + tile_rows);
                svfloat32_t out2_32 = svld1(pN32_full, tmp_pair);
                svfloat32_t out3_32 = svld1(pN32_full, tmp_pair + tile_rows);
                svfloat16_t out2 = svcvt_f16_f32_z(pN32_full, out2_32);
                svfloat16_t out3 = svcvt_f16_f32_z(pN32_full, out3_32);
                out2 = svuzp1_f16(out2, out2);
                out3 = svuzp1_f16(out3, out3);
                svst1_f16_x2(pOut16_full_c, &c[(row + trow) * N + col + 2 * tile_rows], svcreate2(out2, out2));
                svst1_f16_x2(pOut16_full_c, &c[(row + trow) * N + col + 3 * tile_rows], svcreate2(out3, out3));
            }
        }

        for (; cb + 1 < col_blocks && (cb + 2) * tile_rows <= N; cb += 2) {
            svzero_za();

            size_t kp = 0;
            for (; kp + 1 < k_pairs; kp += 2) {
                const __fp16* a_ptr0 = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr0 = b_packed + (kp * col_blocks + cb) * tile_pairs;
                const __fp16* a_ptr1 = a_ptr0 + tile_pairs;
                const __fp16* b_ptr1 = b_ptr0 + col_blocks * tile_pairs;

                const svfloat16_t zA0 = svld1(pMh, a_ptr0);
                const svfloat16x2_t zB20 = svld1_f16_x2(pNh_full_c, b_ptr0);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA0, svget2(zB20, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA0, svget2(zB20, 1));

                const svfloat16_t zA1 = svld1(pMh, a_ptr1);
                const svfloat16x2_t zB21 = svld1_f16_x2(pNh_full_c, b_ptr1);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA1, svget2(zB21, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA1, svget2(zB21, 1));
            }

            for (; kp < k_pairs; ++kp) {
                const __fp16* a_ptr = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr = b_packed + (kp * col_blocks + cb) * tile_pairs;

                const svfloat16_t zA = svld1(pMh, a_ptr);
                const svfloat16x2_t zB2 = svld1_f16_x2(pNh_full_c, b_ptr);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA, svget2(zB2, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA, svget2(zB2, 1));
            }

            const size_t col = cb * tile_rows;

            size_t trow = 0;
            for (; trow + 3 < active_r; trow += 4) {
                for (size_t u = 0; u < 4; ++u) {
                    const size_t tr = trow + u;
                    svst1_hor_za32(0, static_cast<uint32_t>(tr), pN32_full, tmp_pair);
                    svst1_hor_za32(1, static_cast<uint32_t>(tr), pN32_full, tmp_pair + tile_rows);

                    svfloat32_t out0_32 = svld1(pN32_full, tmp_pair);
                    svfloat32_t out1_32 = svld1(pN32_full, tmp_pair + tile_rows);
                    svfloat16_t out0 = svcvt_f16_f32_z(pN32_full, out0_32);
                    svfloat16_t out1 = svcvt_f16_f32_z(pN32_full, out1_32);
                    out0 = svuzp1_f16(out0, out0);
                    out1 = svuzp1_f16(out1, out1);
                    svfloat16x2_t out_pair0 = svcreate2(out0, out0);
                    svfloat16x2_t out_pair1 = svcreate2(out1, out1);
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col], out_pair0);
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col + tile_rows], out_pair1);
                }
            }

            for (; trow + 1 < active_r; trow += 2) {
                for (size_t u = 0; u < 2; ++u) {
                    const size_t tr = trow + u;
                    svst1_hor_za32(0, static_cast<uint32_t>(tr), pN32_full, tmp_pair);
                    svst1_hor_za32(1, static_cast<uint32_t>(tr), pN32_full, tmp_pair + tile_rows);

                    svfloat32_t out0_32 = svld1(pN32_full, tmp_pair);
                    svfloat32_t out1_32 = svld1(pN32_full, tmp_pair + tile_rows);
                    svfloat16_t out0 = svcvt_f16_f32_z(pN32_full, out0_32);
                    svfloat16_t out1 = svcvt_f16_f32_z(pN32_full, out1_32);
                    out0 = svuzp1_f16(out0, out0);
                    out1 = svuzp1_f16(out1, out1);
                    svfloat16x2_t out_pair0 = svcreate2(out0, out0);
                    svfloat16x2_t out_pair1 = svcreate2(out1, out1);
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col], out_pair0);
                    svst1_f16_x2(pOut16_full_c, &c[(row + tr) * N + col + tile_rows], out_pair1);
                }
            }

            for (; trow < active_r; ++trow) {
                svst1_hor_za32(0, static_cast<uint32_t>(trow), pN32_full, tmp_pair);
                svst1_hor_za32(1, static_cast<uint32_t>(trow), pN32_full, tmp_pair + tile_rows);

                svfloat32_t out0_32 = svld1(pN32_full, tmp_pair);
                svfloat32_t out1_32 = svld1(pN32_full, tmp_pair + tile_rows);
                svfloat16_t out0 = svcvt_f16_f32_z(pN32_full, out0_32);
                svfloat16_t out1 = svcvt_f16_f32_z(pN32_full, out1_32);
                out0 = svuzp1_f16(out0, out0);
                out1 = svuzp1_f16(out1, out1);
                svfloat16x2_t out_pair0 = svcreate2(out0, out0);
                svfloat16x2_t out_pair1 = svcreate2(out1, out1);
                svst1_f16_x2(pOut16_full_c, &c[(row + trow) * N + col], out_pair0);
                svst1_f16_x2(pOut16_full_c, &c[(row + trow) * N + col + tile_rows], out_pair1);
            }
        }

        for (; cb < col_blocks; ++cb) {
            const size_t col = cb * tile_rows;
            const size_t active_c = std::min(tile_rows, N - col);
            const svbool_t pNh = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c * 2));
            const svbool_t pN32 = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c));
            const svbool_t pOut16 = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c));

            svzero_za();

            size_t kp = 0;
            for (; kp + 1 < k_pairs; kp += 2) {
                const __fp16* a_ptr0 = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr0 = b_packed + (kp * col_blocks + cb) * tile_pairs;
                const __fp16* a_ptr1 = a_ptr0 + tile_pairs;
                const __fp16* b_ptr1 = b_ptr0 + col_blocks * tile_pairs;

                const svfloat16_t zA0 = svld1(pMh, a_ptr0);
                const svfloat16_t zB0 = svld1(pNh, b_ptr0);
                svmopa_za32_f16_m(0, pMh, pNh, zA0, zB0);

                const svfloat16_t zA1 = svld1(pMh, a_ptr1);
                const svfloat16_t zB1 = svld1(pNh, b_ptr1);
                svmopa_za32_f16_m(0, pMh, pNh, zA1, zB1);
            }

            for (; kp < k_pairs; ++kp) {
                const __fp16* a_ptr = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr = b_packed + (kp * col_blocks + cb) * tile_pairs;

                const svfloat16_t zA = svld1(pMh, a_ptr);
                const svfloat16_t zB = svld1(pNh, b_ptr);
                svmopa_za32_f16_m(0, pMh, pNh, zA, zB);
            }

            size_t trow = 0;
            for (; trow + 3 < active_r; trow += 4) {
                for (size_t u = 0; u < 4; ++u) {
                    const size_t tr = trow + u;
                    svst1_hor_za32(0, static_cast<uint32_t>(tr), pN32, tmp_pair);
                    svfloat32_t out32 = svld1(pN32, tmp_pair);
                    svfloat16_t out16 = svcvt_f16_f32_z(pN32, out32);
                    out16 = svuzp1_f16(out16, out16);
                    svst1(pOut16, &c[(row + tr) * N + col], out16);
                }
            }

            for (; trow + 1 < active_r; trow += 2) {
                for (size_t u = 0; u < 2; ++u) {
                    const size_t tr = trow + u;
                    svst1_hor_za32(0, static_cast<uint32_t>(tr), pN32, tmp_pair);
                    svfloat32_t out32 = svld1(pN32, tmp_pair);
                    svfloat16_t out16 = svcvt_f16_f32_z(pN32, out32);
                    out16 = svuzp1_f16(out16, out16);
                    svst1(pOut16, &c[(row + tr) * N + col], out16);
                }
            }

            for (; trow < active_r; ++trow) {
                svst1_hor_za32(0, static_cast<uint32_t>(trow), pN32, tmp_pair);
                svfloat32_t out32 = svld1(pN32, tmp_pair);
                svfloat16_t out16 = svcvt_f16_f32_z(pN32, out32);
                out16 = svuzp1_f16(out16, out16);
                svst1(pOut16, &c[(row + trow) * N + col], out16);
            }
        }
    }
}

__arm_new("za") __arm_locally_streaming
static void cactus_matmul_f16_sme2_thread_entry(
    const __fp16* a_packed,
    const __fp16* b_packed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t row_block_size,
    size_t start_block,
    size_t end_block
) {
    const size_t tile_rows = svcntsw();
    const size_t tile_pairs = svcnth();
    std::vector<float> tmp_pair(2 * tile_rows, 0.0f);

    for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
        const size_t start_row = block_idx * row_block_size;
        const size_t end_row = std::min(start_row + row_block_size, M);

        cactus_matmul_f16_sme2_worker(
            a_packed,
            b_packed,
            c,
            M,
            K,
            N,
            start_row,
            end_row,
            tile_rows,
            tile_pairs,
            tmp_pair.data()
        );
    }
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
    const size_t tile_rows = svcntsw();
    const size_t tile_pairs = svcnth();

    const size_t row_blocks = (M + tile_rows - 1) / tile_rows;
    const size_t k_pairs = (K + 1) / 2;
    const size_t col_blocks = (N + tile_rows - 1) / tile_rows;

    std::vector<__fp16> a_packed(row_blocks * k_pairs * tile_pairs);
    std::vector<__fp16> b_packed(k_pairs * col_blocks * tile_pairs);

    cactus_pack_a_f16(a, a_packed.data(), M, K, tile_rows, tile_pairs);
    cactus_pack_b_f16_from_bt(b_transposed, b_packed.data(), K, N, tile_rows, tile_pairs);

    constexpr size_t TILES_PER_THREAD = 2;
    const size_t row_block_size = TILES_PER_THREAD * tile_rows;
    const size_t num_row_blocks = (M + row_block_size - 1) / row_block_size;

    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
            cactus_matmul_f16_sme2_thread_entry(
                a_packed.data(),
                b_packed.data(),
                c,
                M,
                K,
                N,
                row_block_size,
                start_block,
                end_block
            );
        });
}

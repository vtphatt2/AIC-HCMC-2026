// Shared-memory-tiled WMMA kernel with dual accumulators (acc_hi for the
// dominant Xh@Wh term, acc_lo for the 3 small correction terms, combined
// only once at the end -- avoids repeatedly swamping small corrections into
// an already-large fp32 running sum across ~96 K-steps).
//
// One thread block = 4 warps (128 threads) covering a 64(M) x 64(N) output
// tile. Each warp owns one 16-row band and loops over 4 16-wide N-sub-tiles.
//
// BUG FIXED vs. the first draft of this file: that version declared ONE
// acc_hi/acc_lo pair per warp and reused it across all 4 N-sub-tiles inside
// the same K-step loop, so results from 4 different output tiles got summed
// into a single accumulator, and the store loop then wrote that one
// corrupted value to all 4 output locations. Fixed by giving each warp 4
// independent accumulator pairs (acc_hi[4]/acc_lo[4], one per N-sub-tile),
// each accumulated across the full K-loop before being stored to its own
// distinct output location.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int BLOCK_M = 64;
const int BLOCK_N = 64;
const int BLOCK_K = 16;
const int N_SUBTILES = BLOCK_N / WMMA_N;  // 4

__global__ void wmma_emulated_matmul_kernel(
    const half *__restrict__ Xh, const half *__restrict__ Xl,
    const half *__restrict__ Wh, const half *__restrict__ Wl,
    float *__restrict__ Out,
    int M, int N, int K) {
    int warpId = threadIdx.x / 32;
    int warpM = warpId * WMMA_M;  // 0, 16, 32, 48 -- this warp's row band within the block

    int blockRow = blockIdx.x * BLOCK_M;
    int blockCol = blockIdx.y * BLOCK_N;

    __shared__ half sh_Xh[BLOCK_M][BLOCK_K];
    __shared__ half sh_Xl[BLOCK_M][BLOCK_K];
    __shared__ half sh_Wh[BLOCK_K][BLOCK_N];
    __shared__ half sh_Wl[BLOCK_K][BLOCK_N];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_hi[N_SUBTILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_lo[N_SUBTILES];
    for (int n = 0; n < N_SUBTILES; n++) {
        wmma::fill_fragment(acc_hi[n], 0.0f);
        wmma::fill_fragment(acc_lo[n], 0.0f);
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_Xh, frag_Xl;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_Wh, frag_Wl;

    for (int k_step = 0; k_step < K; k_step += BLOCK_K) {
        // collaborative load: all 128 threads fill shared memory together
        for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int r = i / BLOCK_K, c = i % BLOCK_K;
            if (blockRow + r < M && k_step + c < K) {
                sh_Xh[r][c] = Xh[(blockRow + r) * K + (k_step + c)];
                sh_Xl[r][c] = Xl[(blockRow + r) * K + (k_step + c)];
            } else {
                sh_Xh[r][c] = __float2half(0.0f);
                sh_Xl[r][c] = __float2half(0.0f);
            }
        }
        for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
            int r = i / BLOCK_N, c = i % BLOCK_N;
            if (k_step + r < K && blockCol + c < N) {
                sh_Wh[r][c] = Wh[(k_step + r) * N + (blockCol + c)];
                sh_Wl[r][c] = Wl[(k_step + r) * N + (blockCol + c)];
            } else {
                sh_Wh[r][c] = __float2half(0.0f);
                sh_Wl[r][c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(frag_Xh, &sh_Xh[warpM][0], BLOCK_K);
        wmma::load_matrix_sync(frag_Xl, &sh_Xl[warpM][0], BLOCK_K);

        for (int n = 0; n < N_SUBTILES; n++) {
            int stepN = n * WMMA_N;
            wmma::load_matrix_sync(frag_Wh, &sh_Wh[0][stepN], BLOCK_N);
            wmma::load_matrix_sync(frag_Wl, &sh_Wl[0][stepN], BLOCK_N);

            wmma::mma_sync(acc_hi[n], frag_Xh, frag_Wh, acc_hi[n]);
            wmma::mma_sync(acc_lo[n], frag_Xh, frag_Wl, acc_lo[n]);
            wmma::mma_sync(acc_lo[n], frag_Xl, frag_Wh, acc_lo[n]);
            wmma::mma_sync(acc_lo[n], frag_Xl, frag_Wl, acc_lo[n]);
        }
        __syncthreads();
    }

    int globalRow = blockRow + warpM;
    for (int n = 0; n < N_SUBTILES; n++) {
        int globalCol = blockCol + n * WMMA_N;
        if (globalRow < M && globalCol < N) {
            for (int i = 0; i < acc_hi[n].num_elements; i++) {
                acc_hi[n].x[i] = acc_hi[n].x[i] + acc_lo[n].x[i];
            }
            wmma::store_matrix_sync(&Out[globalRow * N + globalCol], acc_hi[n], N, wmma::mem_row_major);
        }
    }
}

torch::Tensor emulated_matmul_cuda(torch::Tensor Xh, torch::Tensor Xl, torch::Tensor Wh, torch::Tensor Wl) {
    int M = Xh.size(0);
    int K = Xh.size(1);
    int N = Wh.size(1);

    auto Out = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(Xh.device()));

    dim3 threads(128);  // 4 warps
    dim3 blocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);

    wmma_emulated_matmul_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half *>(Xh.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(Xl.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(Wh.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(Wl.data_ptr<at::Half>()),
        Out.data_ptr<float>(),
        M, N, K);

    return Out;
}

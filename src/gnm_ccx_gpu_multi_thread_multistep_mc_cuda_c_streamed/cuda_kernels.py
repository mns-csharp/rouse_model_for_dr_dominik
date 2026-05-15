"""CUDA kernels for the GPU single-stream conventional MC.

One RawKernel via CuPy:
  - propose_batch_kernel : B blocks x 32 threads/warp; rotates Calpha + SG
                            of one segment per block. Identical to the
                            multistep variant's propose kernel -- the kernel
                            is naturally batch-parameterized, so B=1 (the
                            conventional MC use case) is just a single
                            block launch.

Delta-E and the per-segment Metropolis accept stay on torch / host --
the conventional MC's per-step delta-E is small enough that the cost of
launching a custom delta-E kernel does not pay off (the multistep folder
ships a cell-list-based delta-E kernel for the batched B>>1 case).
"""

from __future__ import annotations

import cupy as cp


COMMON = r"""
__device__ __forceinline__ float wrap_mic(float dx, float box, float inv_box) {
    return dx - box * rintf(dx * inv_box);
}
__device__ __forceinline__ float wrap_box(float v, float box, float half, float inv_box) {
    return v - box * floorf((v + half) * inv_box);
}
"""


_PROPOSE_SRC = COMMON + r"""
extern "C" __global__ void propose_batch_kernel(
    const float* __restrict__ ca,        // [n_chains*N, 3]
    const float* __restrict__ sg,        // [n_chains*N, 3]
    float* __restrict__ old_ca_buf,      // [B, M, 3]
    float* __restrict__ new_ca_buf,
    float* __restrict__ old_sg_buf,
    float* __restrict__ new_sg_buf,
    const long long* __restrict__ chain_idx,
    const long long* __restrict__ bead_start,
    const long long* __restrict__ n_moved_in,
    const long long* __restrict__ seg_type,
    const long long* __restrict__ anchor_a,
    const long long* __restrict__ anchor_b,
    const float* __restrict__ rand_angles,
    long long* __restrict__ n_moved_out,
    long long* __restrict__ move_type,
    int N, int M, int B,
    float box, float inv_box, float half_box,
    float max_angle, float l0)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int t = threadIdx.x;

    int nm = (int)n_moved_in[b];
    if (nm == 0) {
        if (t == 0) n_moved_out[b] = 0;
        return;
    }
    int ci = (int)chain_idx[b];
    int ms = (int)bead_start[b];
    int stype = (int)seg_type[b];
    int a_idx = (int)anchor_a[b];
    int b_idx = (int)anchor_b[b];

    for (int k = t; k < nm; k += blockDim.x) {
        int gi = ci * N + ms + k;
        old_ca_buf[(b * M + k) * 3 + 0] = ca[gi * 3 + 0];
        old_ca_buf[(b * M + k) * 3 + 1] = ca[gi * 3 + 1];
        old_ca_buf[(b * M + k) * 3 + 2] = ca[gi * 3 + 2];
        old_sg_buf[(b * M + k) * 3 + 0] = sg[gi * 3 + 0];
        old_sg_buf[(b * M + k) * 3 + 1] = sg[gi * 3 + 1];
        old_sg_buf[(b * M + k) * 3 + 2] = sg[gi * 3 + 2];
    }
    __syncthreads();

    extern __shared__ float smem[];
    float* uw_ca = smem;
    float* anc = uw_ca + M * 3;
    float* u_axis = anc + 3;
    float* sh_angle = u_axis + 3;
    float* sh_valid = sh_angle + 1;

    if (t == 0) {
        float anc_x = ca[(ci * N + a_idx) * 3 + 0];
        float anc_y = ca[(ci * N + a_idx) * 3 + 1];
        float anc_z = ca[(ci * N + a_idx) * 3 + 2];
        anc[0] = anc_x; anc[1] = anc_y; anc[2] = anc_z;

        if (a_idx <= ms) {
            float dx = ca[(ci * N + ms) * 3 + 0] - anc_x;
            float dy = ca[(ci * N + ms) * 3 + 1] - anc_y;
            float dz = ca[(ci * N + ms) * 3 + 2] - anc_z;
            dx = wrap_mic(dx, box, inv_box);
            dy = wrap_mic(dy, box, inv_box);
            dz = wrap_mic(dz, box, inv_box);
            uw_ca[0] = anc_x + dx; uw_ca[1] = anc_y + dy; uw_ca[2] = anc_z + dz;
            for (int k = 1; k < nm; k++) {
                int gi = ci * N + ms + k;
                int gp = ci * N + ms + k - 1;
                float gx = wrap_mic(ca[gi*3+0] - ca[gp*3+0], box, inv_box);
                float gy = wrap_mic(ca[gi*3+1] - ca[gp*3+1], box, inv_box);
                float gz = wrap_mic(ca[gi*3+2] - ca[gp*3+2], box, inv_box);
                uw_ca[k*3+0] = uw_ca[(k-1)*3+0] + gx;
                uw_ca[k*3+1] = uw_ca[(k-1)*3+1] + gy;
                uw_ca[k*3+2] = uw_ca[(k-1)*3+2] + gz;
            }
        } else {
            int last = nm - 1;
            int gi = ci * N + ms + last;
            float dx = wrap_mic(ca[gi*3+0] - anc_x, box, inv_box);
            float dy = wrap_mic(ca[gi*3+1] - anc_y, box, inv_box);
            float dz = wrap_mic(ca[gi*3+2] - anc_z, box, inv_box);
            uw_ca[last*3+0] = anc_x + dx;
            uw_ca[last*3+1] = anc_y + dy;
            uw_ca[last*3+2] = anc_z + dz;
            for (int k = last - 1; k >= 0; k--) {
                int gj = ci * N + ms + k;
                int gn = ci * N + ms + k + 1;
                float gx = wrap_mic(ca[gj*3+0] - ca[gn*3+0], box, inv_box);
                float gy = wrap_mic(ca[gj*3+1] - ca[gn*3+1], box, inv_box);
                float gz = wrap_mic(ca[gj*3+2] - ca[gn*3+2], box, inv_box);
                uw_ca[k*3+0] = uw_ca[(k+1)*3+0] + gx;
                uw_ca[k*3+1] = uw_ca[(k+1)*3+1] + gy;
                uw_ca[k*3+2] = uw_ca[(k+1)*3+2] + gz;
            }
        }

        float ax_dx, ax_dy, ax_dz, axis_len;
        if (stype == 2) { // INNER
            int last = nm - 1;
            int gi_b = ci * N + b_idx;
            int gi_l = ci * N + ms + last;
            float ux_b = wrap_mic(ca[gi_b*3+0] - ca[gi_l*3+0], box, inv_box);
            float uy_b = wrap_mic(ca[gi_b*3+1] - ca[gi_l*3+1], box, inv_box);
            float uz_b = wrap_mic(ca[gi_b*3+2] - ca[gi_l*3+2], box, inv_box);
            float ub_x = uw_ca[last*3+0] + ux_b;
            float ub_y = uw_ca[last*3+1] + uy_b;
            float ub_z = uw_ca[last*3+2] + uz_b;
            ax_dx = ub_x - anc_x; ax_dy = ub_y - anc_y; ax_dz = ub_z - anc_z;
        } else {
            int gi_b = ci * N + b_idx;
            ax_dx = wrap_mic(ca[gi_b*3+0] - anc_x, box, inv_box);
            ax_dy = wrap_mic(ca[gi_b*3+1] - anc_y, box, inv_box);
            ax_dz = wrap_mic(ca[gi_b*3+2] - anc_z, box, inv_box);
        }
        axis_len = sqrtf(ax_dx*ax_dx + ax_dy*ax_dy + ax_dz*ax_dz);
        if (axis_len < 1e-7f) {
            sh_valid[0] = 0.0f;
        } else {
            sh_valid[0] = 1.0f;
            float inv_l = 1.0f / axis_len;
            u_axis[0] = ax_dx * inv_l;
            u_axis[1] = ax_dy * inv_l;
            u_axis[2] = ax_dz * inv_l;
            sh_angle[0] = (2.0f * rand_angles[b] - 1.0f) * max_angle;
        }
        n_moved_out[b] = sh_valid[0] > 0.5f ? nm : 0;
        if (stype == 2) move_type[b] = 0;
        else if (stype == 1) move_type[b] = 2;
        else move_type[b] = 1;
    }
    __syncthreads();

    if (sh_valid[0] < 0.5f) {
        for (int k = t; k < nm; k += blockDim.x) {
            new_ca_buf[(b*M + k)*3+0] = old_ca_buf[(b*M + k)*3+0];
            new_ca_buf[(b*M + k)*3+1] = old_ca_buf[(b*M + k)*3+1];
            new_ca_buf[(b*M + k)*3+2] = old_ca_buf[(b*M + k)*3+2];
            new_sg_buf[(b*M + k)*3+0] = old_sg_buf[(b*M + k)*3+0];
            new_sg_buf[(b*M + k)*3+1] = old_sg_buf[(b*M + k)*3+1];
            new_sg_buf[(b*M + k)*3+2] = old_sg_buf[(b*M + k)*3+2];
        }
        return;
    }

    float angle = sh_angle[0];
    float ux = u_axis[0], uy = u_axis[1], uz = u_axis[2];
    float c = cosf(angle), s = sinf(angle), tt = 1.0f - c;
    float r00 = tt*ux*ux + c, r01 = tt*ux*uy - s*uz, r02 = tt*ux*uz + s*uy;
    float r10 = tt*ux*uy + s*uz, r11 = tt*uy*uy + c, r12 = tt*uy*uz - s*ux;
    float r20 = tt*ux*uz - s*uy, r21 = tt*uy*uz + s*ux, r22 = tt*uz*uz + c;
    float ax = anc[0], ay = anc[1], az = anc[2];

    for (int k = t; k < nm; k += blockDim.x) {
        float rx = uw_ca[k*3+0] - ax;
        float ry = uw_ca[k*3+1] - ay;
        float rz = uw_ca[k*3+2] - az;
        float nx = ax + r00*rx + r01*ry + r02*rz;
        float ny = ay + r10*rx + r11*ry + r12*rz;
        float nz = az + r20*rx + r21*ry + r22*rz;
        nx = wrap_box(nx, box, half_box, inv_box);
        ny = wrap_box(ny, box, half_box, inv_box);
        nz = wrap_box(nz, box, half_box, inv_box);
        new_ca_buf[(b*M + k)*3+0] = nx;
        new_ca_buf[(b*M + k)*3+1] = ny;
        new_ca_buf[(b*M + k)*3+2] = nz;

        int gi = ci * N + ms + k;
        float ox = wrap_mic(sg[gi*3+0] - ca[gi*3+0], box, inv_box);
        float oy = wrap_mic(sg[gi*3+1] - ca[gi*3+1], box, inv_box);
        float oz = wrap_mic(sg[gi*3+2] - ca[gi*3+2], box, inv_box);
        float sx = uw_ca[k*3+0] + ox - ax;
        float sy = uw_ca[k*3+1] + oy - ay;
        float sz = uw_ca[k*3+2] + oz - az;
        float mx = ax + r00*sx + r01*sy + r02*sz;
        float my = ay + r10*sx + r11*sy + r12*sz;
        float mz = az + r20*sx + r21*sy + r22*sz;
        mx = wrap_box(mx, box, half_box, inv_box);
        my = wrap_box(my, box, half_box, inv_box);
        mz = wrap_box(mz, box, half_box, inv_box);
        new_sg_buf[(b*M + k)*3+0] = mx;
        new_sg_buf[(b*M + k)*3+1] = my;
        new_sg_buf[(b*M + k)*3+2] = mz;
    }
}
"""


_DELTA_E_SRC = COMMON + r"""
extern "C" __global__ void delta_e_segment_kernel(
    const float* __restrict__ ca_full,        // [n_chains*N, 3]
    const float* __restrict__ sg_full,        // [n_chains*N, 3]
    const float* __restrict__ old_ca_buf,     // [B, M, 3]
    const float* __restrict__ new_ca_buf,
    const float* __restrict__ old_sg_buf,
    const float* __restrict__ new_sg_buf,
    const long long* __restrict__ n_moved,    // [B]
    const long long* __restrict__ chain_idx,  // [B]
    const long long* __restrict__ bead_start, // [B]
    float* __restrict__ delta_e,              // [B]
    int M, int B,
    int n_chains, int N,
    float box, float inv_box,
    float r_rep_sq, float r_max_sq,
    float rep_e, float contact_e,
    int min_caca, int min_casg, int min_sgsg)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;

    int nm = (int)n_moved[b];
    if (nm == 0) {
        if (tid == 0) delta_e[b] = 0.0f;
        return;
    }
    int ci = (int)chain_idx[b];
    int ms = (int)bead_start[b];

    int total = n_chains * N;
    int total_pairs = nm * total;

    float local_de = 0.0f;
    bool has_contact = (contact_e != 0.0f);

    for (int p = tid; p < total_pairs; p += blockDim.x) {
        int i = p / total;
        int j = p % total;
        int j_chain = j / N;
        int j_resid = j - j_chain * N;
        bool same_chain = (j_chain == ci);
        bool j_in_moved = same_chain && (j_resid >= ms) && (j_resid < ms + nm);
        if (j_in_moved) continue;

        int moved_resid = ms + i;
        int seq_diff = abs(j_resid - moved_resid);

        float jcx = ca_full[j*3+0], jcy = ca_full[j*3+1], jcz = ca_full[j*3+2];
        float jsx = sg_full[j*3+0], jsy = sg_full[j*3+1], jsz = sg_full[j*3+2];

        int off = (b * M + i) * 3;
        float ocx = old_ca_buf[off+0], ocy = old_ca_buf[off+1], ocz = old_ca_buf[off+2];
        float ncx = new_ca_buf[off+0], ncy = new_ca_buf[off+1], ncz = new_ca_buf[off+2];
        float osx = old_sg_buf[off+0], osy = old_sg_buf[off+1], osz = old_sg_buf[off+2];
        float nsx = new_sg_buf[off+0], nsy = new_sg_buf[off+1], nsz = new_sg_buf[off+2];

        // Calpha(moved) vs Calpha(j)
        if (!same_chain || seq_diff >= min_caca) {
            float dx, dy, dz, r2_old, r2_new, e_old, e_new;
            dx = wrap_mic(jcx - ocx, box, inv_box);
            dy = wrap_mic(jcy - ocy, box, inv_box);
            dz = wrap_mic(jcz - ocz, box, inv_box);
            r2_old = dx*dx + dy*dy + dz*dz;
            dx = wrap_mic(jcx - ncx, box, inv_box);
            dy = wrap_mic(jcy - ncy, box, inv_box);
            dz = wrap_mic(jcz - ncz, box, inv_box);
            r2_new = dx*dx + dy*dy + dz*dz;
            e_old = (r2_old < r_rep_sq) ? rep_e : ((has_contact && r2_old < r_max_sq) ? contact_e : 0.0f);
            e_new = (r2_new < r_rep_sq) ? rep_e : ((has_contact && r2_new < r_max_sq) ? contact_e : 0.0f);
            local_de += (e_new - e_old);
        }

        // Calpha(moved) vs SG(j)
        if (!same_chain || seq_diff >= min_casg) {
            float dx, dy, dz, r2_old, r2_new, e_old, e_new;
            dx = wrap_mic(jsx - ocx, box, inv_box);
            dy = wrap_mic(jsy - ocy, box, inv_box);
            dz = wrap_mic(jsz - ocz, box, inv_box);
            r2_old = dx*dx + dy*dy + dz*dz;
            dx = wrap_mic(jsx - ncx, box, inv_box);
            dy = wrap_mic(jsy - ncy, box, inv_box);
            dz = wrap_mic(jsz - ncz, box, inv_box);
            r2_new = dx*dx + dy*dy + dz*dz;
            e_old = (r2_old < r_rep_sq) ? rep_e : ((has_contact && r2_old < r_max_sq) ? contact_e : 0.0f);
            e_new = (r2_new < r_rep_sq) ? rep_e : ((has_contact && r2_new < r_max_sq) ? contact_e : 0.0f);
            local_de += (e_new - e_old);
        }

        // SG(moved) vs Calpha(j)
        if (!same_chain || seq_diff >= min_casg) {
            float dx, dy, dz, r2_old, r2_new, e_old, e_new;
            dx = wrap_mic(jcx - osx, box, inv_box);
            dy = wrap_mic(jcy - osy, box, inv_box);
            dz = wrap_mic(jcz - osz, box, inv_box);
            r2_old = dx*dx + dy*dy + dz*dz;
            dx = wrap_mic(jcx - nsx, box, inv_box);
            dy = wrap_mic(jcy - nsy, box, inv_box);
            dz = wrap_mic(jcz - nsz, box, inv_box);
            r2_new = dx*dx + dy*dy + dz*dz;
            e_old = (r2_old < r_rep_sq) ? rep_e : ((has_contact && r2_old < r_max_sq) ? contact_e : 0.0f);
            e_new = (r2_new < r_rep_sq) ? rep_e : ((has_contact && r2_new < r_max_sq) ? contact_e : 0.0f);
            local_de += (e_new - e_old);
        }

        // SG(moved) vs SG(j)
        if (!same_chain || seq_diff >= min_sgsg) {
            float dx, dy, dz, r2_old, r2_new, e_old, e_new;
            dx = wrap_mic(jsx - osx, box, inv_box);
            dy = wrap_mic(jsy - osy, box, inv_box);
            dz = wrap_mic(jsz - osz, box, inv_box);
            r2_old = dx*dx + dy*dy + dz*dz;
            dx = wrap_mic(jsx - nsx, box, inv_box);
            dy = wrap_mic(jsy - nsy, box, inv_box);
            dz = wrap_mic(jsz - nsz, box, inv_box);
            r2_new = dx*dx + dy*dy + dz*dz;
            e_old = (r2_old < r_rep_sq) ? rep_e : ((has_contact && r2_old < r_max_sq) ? contact_e : 0.0f);
            e_new = (r2_new < r_rep_sq) ? rep_e : ((has_contact && r2_new < r_max_sq) ? contact_e : 0.0f);
            local_de += (e_new - e_old);
        }
    }

    extern __shared__ float sdata[];
    sdata[tid] = local_de;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) delta_e[b] = sdata[0];
}
"""


_ACCEPT_SRC = COMMON + r"""
extern "C" __global__ void accept_segment_kernel(
    float* __restrict__ ca,             // [n_chains*N, 3], mutated in place
    float* __restrict__ sg,             // [n_chains*N, 3]
    const float* __restrict__ new_ca_buf, // [B, M, 3]
    const float* __restrict__ new_sg_buf,
    const float* __restrict__ delta_e,    // [B]
    const long long* __restrict__ n_moved,    // [B]
    const long long* __restrict__ move_type,  // [B]
    const long long* __restrict__ chain_idx,  // [B]
    const long long* __restrict__ bead_start, // [B]
    const float* __restrict__ rand_accept,    // [B]
    long long* __restrict__ attempted_counts, // [3] hinge,n_tail,c_tail
    long long* __restrict__ accepted_counts,  // [3]
    int M, int B, int N,
    float inv_kBT)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int nm = (int)n_moved[b];
    int mt = (int)move_type[b];
    if (mt < 0 || mt > 2) return;

    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)&attempted_counts[mt], (unsigned long long)1ULL);
    }
    if (nm == 0) return;

    float de = delta_e[b];
    float prob;
    if (de <= 0.0f) {
        prob = 1.0f;
    } else {
        float exponent = -de * inv_kBT;
        if (exponent <= -88.7f) prob = 0.0f;
        else if (exponent >= 88.7f) prob = 1.0f;
        else prob = expf(exponent);
    }
    bool accept = rand_accept[b] < prob;
    if (!accept) return;

    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)&accepted_counts[mt], (unsigned long long)1ULL);
    }
    int ci = (int)chain_idx[b];
    int ms = (int)bead_start[b];
    for (int k = threadIdx.x; k < nm; k += blockDim.x) {
        int g = (ci * N + ms + k) * 3;
        int s = (b * M + k) * 3;
        ca[g+0] = new_ca_buf[s+0];
        ca[g+1] = new_ca_buf[s+1];
        ca[g+2] = new_ca_buf[s+2];
        sg[g+0] = new_sg_buf[s+0];
        sg[g+1] = new_sg_buf[s+1];
        sg[g+2] = new_sg_buf[s+2];
    }
}
"""


_MC_SWEEP_SRC = COMMON + r"""
extern "C" __global__ void mc_sweep_kernel(
    float* __restrict__ ca,                      // [n_chains*N, 3] mutated in place
    float* __restrict__ sg,
    const long long* __restrict__ meta_table,    // [total_segs, 6]
    const long long* __restrict__ perm,          // [n_segs] segment ids in order
    const float* __restrict__ rand_angles,       // [n_segs]
    const float* __restrict__ rand_accepts,      // [n_segs]
    long long* __restrict__ attempted_counts,    // [3]
    long long* __restrict__ accepted_counts,     // [3]
    int n_segs, int N, int M, int n_chains,
    float box, float inv_box, float half_box,
    float max_angle, float inv_kBT,
    float r_rep_sq, float r_max_sq, float rep_e, float contact_e,
    int min_caca, int min_casg, int min_sgsg)
{
    int tid = threadIdx.x;
    int blk = blockDim.x;

    // Shared memory layout (non-overlapping; partial_de placed after all
    // position buffers to avoid clobbering old/new positions during the
    // delta_e reduction at small M):
    //   uw_ca[M*3] + anc[3] + u_axis[3] + sh_angle[1] + sh_valid[1]
    //   + old_ca[M*3] + new_ca[M*3] + old_sg[M*3] + new_sg[M*3]
    //   + partial_de[blockDim.x] + sh_accept[1]
    extern __shared__ float smem[];
    float* uw_ca   = smem;                  // [M*3]
    float* anc     = uw_ca + M * 3;         // [3]
    float* u_axis  = anc + 3;               // [3]
    float* sh_angle = u_axis + 3;           // [1]
    float* sh_valid = sh_angle + 1;         // [1]
    float* old_ca_s = sh_valid + 1;         // [M*3]
    float* new_ca_s = old_ca_s + M * 3;     // [M*3]
    float* old_sg_s = new_ca_s + M * 3;     // [M*3]
    float* new_sg_s = old_sg_s + M * 3;     // [M*3]
    float* partial_de = new_sg_s + M * 3;   // [blockDim.x]
    float* sh_accept = partial_de + blk;    // [1]

    // Iterate over all segments in the prescribed permutation.
    for (int step = 0; step < n_segs; step++) {
        int gs = (int)perm[step];
        int ci   = (int)meta_table[gs * 6 + 0];
        int ms   = (int)meta_table[gs * 6 + 1];
        int nm   = (int)meta_table[gs * 6 + 2];
        int stype = (int)meta_table[gs * 6 + 3];
        int a_idx = (int)meta_table[gs * 6 + 4];
        int b_idx = (int)meta_table[gs * 6 + 5];

        if (nm == 0) {
            __syncthreads();
            continue;
        }

        // ============== PHASE 1: PROPOSE ==============
        // Lane 0 unwraps + computes axis + chooses angle; broadcast via shared mem.
        // Threads 0..nm-1 load old positions to shared memory.
        for (int k = tid; k < nm; k += blk) {
            int gi = ci * N + ms + k;
            old_ca_s[k*3+0] = ca[gi*3+0];
            old_ca_s[k*3+1] = ca[gi*3+1];
            old_ca_s[k*3+2] = ca[gi*3+2];
            old_sg_s[k*3+0] = sg[gi*3+0];
            old_sg_s[k*3+1] = sg[gi*3+1];
            old_sg_s[k*3+2] = sg[gi*3+2];
        }
        __syncthreads();

        if (tid == 0) {
            float anc_x = ca[(ci * N + a_idx) * 3 + 0];
            float anc_y = ca[(ci * N + a_idx) * 3 + 1];
            float anc_z = ca[(ci * N + a_idx) * 3 + 2];
            anc[0] = anc_x; anc[1] = anc_y; anc[2] = anc_z;

            // Unwrap moved Calphas relative to anchor.
            if (a_idx <= ms) {
                float dx = ca[(ci * N + ms) * 3 + 0] - anc_x;
                float dy = ca[(ci * N + ms) * 3 + 1] - anc_y;
                float dz = ca[(ci * N + ms) * 3 + 2] - anc_z;
                dx = wrap_mic(dx, box, inv_box);
                dy = wrap_mic(dy, box, inv_box);
                dz = wrap_mic(dz, box, inv_box);
                uw_ca[0] = anc_x + dx; uw_ca[1] = anc_y + dy; uw_ca[2] = anc_z + dz;
                for (int k = 1; k < nm; k++) {
                    int gi = ci * N + ms + k;
                    int gp = ci * N + ms + k - 1;
                    float gx = wrap_mic(ca[gi*3+0] - ca[gp*3+0], box, inv_box);
                    float gy = wrap_mic(ca[gi*3+1] - ca[gp*3+1], box, inv_box);
                    float gz = wrap_mic(ca[gi*3+2] - ca[gp*3+2], box, inv_box);
                    uw_ca[k*3+0] = uw_ca[(k-1)*3+0] + gx;
                    uw_ca[k*3+1] = uw_ca[(k-1)*3+1] + gy;
                    uw_ca[k*3+2] = uw_ca[(k-1)*3+2] + gz;
                }
            } else {
                int last = nm - 1;
                int gi = ci * N + ms + last;
                float dx = wrap_mic(ca[gi*3+0] - anc_x, box, inv_box);
                float dy = wrap_mic(ca[gi*3+1] - anc_y, box, inv_box);
                float dz = wrap_mic(ca[gi*3+2] - anc_z, box, inv_box);
                uw_ca[last*3+0] = anc_x + dx;
                uw_ca[last*3+1] = anc_y + dy;
                uw_ca[last*3+2] = anc_z + dz;
                for (int k = last - 1; k >= 0; k--) {
                    int gj = ci * N + ms + k;
                    int gn = ci * N + ms + k + 1;
                    float gx = wrap_mic(ca[gj*3+0] - ca[gn*3+0], box, inv_box);
                    float gy = wrap_mic(ca[gj*3+1] - ca[gn*3+1], box, inv_box);
                    float gz = wrap_mic(ca[gj*3+2] - ca[gn*3+2], box, inv_box);
                    uw_ca[k*3+0] = uw_ca[(k+1)*3+0] + gx;
                    uw_ca[k*3+1] = uw_ca[(k+1)*3+1] + gy;
                    uw_ca[k*3+2] = uw_ca[(k+1)*3+2] + gz;
                }
            }

            float ax_dx, ax_dy, ax_dz, axis_len;
            if (stype == 2) { // INNER
                int last = nm - 1;
                int gi_b = ci * N + b_idx;
                int gi_l = ci * N + ms + last;
                float ux_b = wrap_mic(ca[gi_b*3+0] - ca[gi_l*3+0], box, inv_box);
                float uy_b = wrap_mic(ca[gi_b*3+1] - ca[gi_l*3+1], box, inv_box);
                float uz_b = wrap_mic(ca[gi_b*3+2] - ca[gi_l*3+2], box, inv_box);
                float ub_x = uw_ca[last*3+0] + ux_b;
                float ub_y = uw_ca[last*3+1] + uy_b;
                float ub_z = uw_ca[last*3+2] + uz_b;
                ax_dx = ub_x - anc_x; ax_dy = ub_y - anc_y; ax_dz = ub_z - anc_z;
            } else {
                int gi_b = ci * N + b_idx;
                ax_dx = wrap_mic(ca[gi_b*3+0] - anc_x, box, inv_box);
                ax_dy = wrap_mic(ca[gi_b*3+1] - anc_y, box, inv_box);
                ax_dz = wrap_mic(ca[gi_b*3+2] - anc_z, box, inv_box);
            }
            axis_len = sqrtf(ax_dx*ax_dx + ax_dy*ax_dy + ax_dz*ax_dz);
            if (axis_len < 1e-7f) {
                sh_valid[0] = 0.0f;
            } else {
                sh_valid[0] = 1.0f;
                float inv_l = 1.0f / axis_len;
                u_axis[0] = ax_dx * inv_l;
                u_axis[1] = ax_dy * inv_l;
                u_axis[2] = ax_dz * inv_l;
                sh_angle[0] = (2.0f * rand_angles[step] - 1.0f) * max_angle;
            }

            int mt;
            if (stype == 2) mt = 0;          // hinge
            else if (stype == 1) mt = 2;     // c_tail
            else mt = 1;                     // n_tail
            atomicAdd((unsigned long long*)&attempted_counts[mt], (unsigned long long)1ULL);
        }
        __syncthreads();

        if (sh_valid[0] < 0.5f) {
            // Degenerate axis -> no-op (already counted as attempted).
            continue;
        }

        // Compute new positions (rotate uw_ca around anchor, then wrap).
        float angle = sh_angle[0];
        float ux = u_axis[0], uy = u_axis[1], uz = u_axis[2];
        float c = cosf(angle), s = sinf(angle), tt = 1.0f - c;
        float r00 = tt*ux*ux + c, r01 = tt*ux*uy - s*uz, r02 = tt*ux*uz + s*uy;
        float r10 = tt*ux*uy + s*uz, r11 = tt*uy*uy + c, r12 = tt*uy*uz - s*ux;
        float r20 = tt*ux*uz - s*uy, r21 = tt*uy*uz + s*ux, r22 = tt*uz*uz + c;
        float ax = anc[0], ay = anc[1], az = anc[2];

        for (int k = tid; k < nm; k += blk) {
            int gi = ci * N + ms + k;
            float rx = uw_ca[k*3+0] - ax;
            float ry = uw_ca[k*3+1] - ay;
            float rz = uw_ca[k*3+2] - az;
            float nx = ax + r00*rx + r01*ry + r02*rz;
            float ny = ay + r10*rx + r11*ry + r12*rz;
            float nz = az + r20*rx + r21*ry + r22*rz;
            nx = wrap_box(nx, box, half_box, inv_box);
            ny = wrap_box(ny, box, half_box, inv_box);
            nz = wrap_box(nz, box, half_box, inv_box);
            new_ca_s[k*3+0] = nx;
            new_ca_s[k*3+1] = ny;
            new_ca_s[k*3+2] = nz;

            float ox = wrap_mic(sg[gi*3+0] - ca[gi*3+0], box, inv_box);
            float oy = wrap_mic(sg[gi*3+1] - ca[gi*3+1], box, inv_box);
            float oz = wrap_mic(sg[gi*3+2] - ca[gi*3+2], box, inv_box);
            float sx = uw_ca[k*3+0] + ox - ax;
            float sy = uw_ca[k*3+1] + oy - ay;
            float sz = uw_ca[k*3+2] + oz - az;
            float mx = ax + r00*sx + r01*sy + r02*sz;
            float my = ay + r10*sx + r11*sy + r12*sz;
            float mz = az + r20*sx + r21*sy + r22*sz;
            mx = wrap_box(mx, box, half_box, inv_box);
            my = wrap_box(my, box, half_box, inv_box);
            mz = wrap_box(mz, box, half_box, inv_box);
            new_sg_s[k*3+0] = mx;
            new_sg_s[k*3+1] = my;
            new_sg_s[k*3+2] = mz;
        }
        __syncthreads();

        // ============== PHASE 2: DELTA_E ==============
        // partial_de overlays smem; uw_ca/anc/u_axis no longer needed.
        // Pair each (i in 0..nm-1) with each j in 0..n_chains*N-1.
        bool has_contact = (contact_e != 0.0f);
        int total = n_chains * N;
        int total_pairs = nm * total;
        float local_de = 0.0f;
        for (int p = tid; p < total_pairs; p += blk) {
            int i = p / total;
            int j = p % total;
            int j_chain = j / N;
            int j_resid = j - j_chain * N;
            bool same_chain = (j_chain == ci);
            bool j_in_moved = same_chain && (j_resid >= ms) && (j_resid < ms + nm);
            if (j_in_moved) continue;

            int moved_resid = ms + i;
            int seq_diff = abs(j_resid - moved_resid);

            float jcx = ca[j*3+0], jcy = ca[j*3+1], jcz = ca[j*3+2];
            float jsx = sg[j*3+0], jsy = sg[j*3+1], jsz = sg[j*3+2];
            float ocx = old_ca_s[i*3+0], ocy = old_ca_s[i*3+1], ocz = old_ca_s[i*3+2];
            float ncx = new_ca_s[i*3+0], ncy = new_ca_s[i*3+1], ncz = new_ca_s[i*3+2];
            float osx = old_sg_s[i*3+0], osy = old_sg_s[i*3+1], osz = old_sg_s[i*3+2];
            float nsx = new_sg_s[i*3+0], nsy = new_sg_s[i*3+1], nsz = new_sg_s[i*3+2];

            // Calpha(moved) vs Calpha(j)
            if (!same_chain || seq_diff >= min_caca) {
                float dx, dy, dz, r2_old, r2_new, e_old, e_new;
                dx = wrap_mic(jcx - ocx, box, inv_box);
                dy = wrap_mic(jcy - ocy, box, inv_box);
                dz = wrap_mic(jcz - ocz, box, inv_box);
                r2_old = dx*dx + dy*dy + dz*dz;
                dx = wrap_mic(jcx - ncx, box, inv_box);
                dy = wrap_mic(jcy - ncy, box, inv_box);
                dz = wrap_mic(jcz - ncz, box, inv_box);
                r2_new = dx*dx + dy*dy + dz*dz;
                e_old = (r2_old < r_rep_sq) ? rep_e : ((has_contact && r2_old < r_max_sq) ? contact_e : 0.0f);
                e_new = (r2_new < r_rep_sq) ? rep_e : ((has_contact && r2_new < r_max_sq) ? contact_e : 0.0f);
                local_de += (e_new - e_old);
            }
            // Calpha(moved) vs SG(j)
            if (!same_chain || seq_diff >= min_casg) {
                float dx, dy, dz, r2_old, r2_new, e_old, e_new;
                dx = wrap_mic(jsx - ocx, box, inv_box);
                dy = wrap_mic(jsy - ocy, box, inv_box);
                dz = wrap_mic(jsz - ocz, box, inv_box);
                r2_old = dx*dx + dy*dy + dz*dz;
                dx = wrap_mic(jsx - ncx, box, inv_box);
                dy = wrap_mic(jsy - ncy, box, inv_box);
                dz = wrap_mic(jsz - ncz, box, inv_box);
                r2_new = dx*dx + dy*dy + dz*dz;
                e_old = (r2_old < r_rep_sq) ? rep_e : ((has_contact && r2_old < r_max_sq) ? contact_e : 0.0f);
                e_new = (r2_new < r_rep_sq) ? rep_e : ((has_contact && r2_new < r_max_sq) ? contact_e : 0.0f);
                local_de += (e_new - e_old);
            }
            // SG(moved) vs Calpha(j)
            if (!same_chain || seq_diff >= min_casg) {
                float dx, dy, dz, r2_old, r2_new, e_old, e_new;
                dx = wrap_mic(jcx - osx, box, inv_box);
                dy = wrap_mic(jcy - osy, box, inv_box);
                dz = wrap_mic(jcz - osz, box, inv_box);
                r2_old = dx*dx + dy*dy + dz*dz;
                dx = wrap_mic(jcx - nsx, box, inv_box);
                dy = wrap_mic(jcy - nsy, box, inv_box);
                dz = wrap_mic(jcz - nsz, box, inv_box);
                r2_new = dx*dx + dy*dy + dz*dz;
                e_old = (r2_old < r_rep_sq) ? rep_e : ((has_contact && r2_old < r_max_sq) ? contact_e : 0.0f);
                e_new = (r2_new < r_rep_sq) ? rep_e : ((has_contact && r2_new < r_max_sq) ? contact_e : 0.0f);
                local_de += (e_new - e_old);
            }
            // SG(moved) vs SG(j)
            if (!same_chain || seq_diff >= min_sgsg) {
                float dx, dy, dz, r2_old, r2_new, e_old, e_new;
                dx = wrap_mic(jsx - osx, box, inv_box);
                dy = wrap_mic(jsy - osy, box, inv_box);
                dz = wrap_mic(jsz - osz, box, inv_box);
                r2_old = dx*dx + dy*dy + dz*dz;
                dx = wrap_mic(jsx - nsx, box, inv_box);
                dy = wrap_mic(jsy - nsy, box, inv_box);
                dz = wrap_mic(jsz - nsz, box, inv_box);
                r2_new = dx*dx + dy*dy + dz*dz;
                e_old = (r2_old < r_rep_sq) ? rep_e : ((has_contact && r2_old < r_max_sq) ? contact_e : 0.0f);
                e_new = (r2_new < r_rep_sq) ? rep_e : ((has_contact && r2_new < r_max_sq) ? contact_e : 0.0f);
                local_de += (e_new - e_old);
            }
        }

        partial_de[tid] = local_de;
        __syncthreads();
        for (int off = blk / 2; off > 0; off >>= 1) {
            if (tid < off) partial_de[tid] += partial_de[tid + off];
            __syncthreads();
        }

        // ============== PHASE 3: ACCEPT ==============
        if (tid == 0) {
            float de = partial_de[0];
            float prob;
            if (de <= 0.0f) prob = 1.0f;
            else {
                float exponent = -de * inv_kBT;
                if (exponent <= -88.7f) prob = 0.0f;
                else if (exponent >= 88.7f) prob = 1.0f;
                else prob = expf(exponent);
            }
            sh_accept[0] = (rand_accepts[step] < prob) ? 1.0f : 0.0f;
            if (sh_accept[0] > 0.5f) {
                int mt;
                if (stype == 2) mt = 0;
                else if (stype == 1) mt = 2;
                else mt = 1;
                atomicAdd((unsigned long long*)&accepted_counts[mt], (unsigned long long)1ULL);
            }
        }
        __syncthreads();

        // ============== PHASE 4: WRITEBACK ==============
        if (sh_accept[0] > 0.5f) {
            for (int k = tid; k < nm; k += blk) {
                int gi = (ci * N + ms + k) * 3;
                ca[gi+0] = new_ca_s[k*3+0];
                ca[gi+1] = new_ca_s[k*3+1];
                ca[gi+2] = new_ca_s[k*3+2];
                sg[gi+0] = new_sg_s[k*3+0];
                sg[gi+1] = new_sg_s[k*3+1];
                sg[gi+2] = new_sg_s[k*3+2];
            }
        }
        __syncthreads();
    }
}
"""


# ---------------------------------------------------------------------------
# Streamed rank-1 EMM correction (g1m_ccx optimization, Workstream B).
# Two kernels replace the dense (B,B,2M,2M) torch correction tensor:
#   corr_bounds_kernel    : O(B*M) bounding-sphere prelude (ref point + radius)
#   correction_pairs_kernel : one block per (i,j) pair; an exact bounding-
#       sphere prefilter skips pairs whose spheres are farther apart than the
#       interaction cutoff (their cross-energy is provably zero), then the
#       surviving pairs stream the nm_i*nm_j moved-bead pairs in registers.
# ---------------------------------------------------------------------------

_CORR_BOUNDS_SRC = COMMON + r"""
extern "C" __global__ void corr_bounds_kernel(
    const float* __restrict__ old_ca,   // [B,M,3]
    const float* __restrict__ new_ca,
    const float* __restrict__ old_sg,
    const float* __restrict__ new_sg,
    const long long* __restrict__ n_moved,  // [B]
    float* __restrict__ ref,            // [B,3] out
    float* __restrict__ radius,         // [B]   out
    int M, int B, float box, float inv_box)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    int nm = (int)n_moved[b];

    __shared__ float sref[3];
    if (tid == 0) {
        if (nm > 0) {
            sref[0] = old_ca[(b*M+0)*3+0];
            sref[1] = old_ca[(b*M+0)*3+1];
            sref[2] = old_ca[(b*M+0)*3+2];
        } else {
            sref[0] = 0.0f; sref[1] = 0.0f; sref[2] = 0.0f;
        }
    }
    __syncthreads();

    float rx = sref[0], ry = sref[1], rz = sref[2];
    float local_max = 0.0f;
    const float* arrs[4] = { old_ca, new_ca, old_sg, new_sg };
    for (int k = tid; k < nm; k += blockDim.x) {
        #pragma unroll
        for (int a = 0; a < 4; a++) {
            const float* arr = arrs[a];
            float dx = wrap_mic(arr[(b*M+k)*3+0] - rx, box, inv_box);
            float dy = wrap_mic(arr[(b*M+k)*3+1] - ry, box, inv_box);
            float dz = wrap_mic(arr[(b*M+k)*3+2] - rz, box, inv_box);
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 > local_max) local_max = d2;
        }
    }
    extern __shared__ float sdata[];
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        ref[b*3+0] = rx; ref[b*3+1] = ry; ref[b*3+2] = rz;
        radius[b] = sqrtf(sdata[0]);
    }
}
"""


_CORR_PAIRS_SRC = COMMON + r"""
__device__ __forceinline__ float zone_e_pair(
    float ax, float ay, float az, float bx, float by, float bz,
    float box, float inv_box, float r_rep_sq, float r_max_sq,
    float rep_e, float contact_e, bool has_contact)
{
    float dx = wrap_mic(ax - bx, box, inv_box);
    float dy = wrap_mic(ay - by, box, inv_box);
    float dz = wrap_mic(az - bz, box, inv_box);
    float r2 = dx*dx + dy*dy + dz*dz;
    return (r2 < r_rep_sq) ? rep_e
         : ((has_contact && r2 < r_max_sq) ? contact_e : 0.0f);
}

__device__ __forceinline__ float cross4(
    float oix, float oiy, float oiz, float nix, float niy, float niz,
    float ojx, float ojy, float ojz, float njx, float njy, float njz,
    float box, float inv_box, float r_rep_sq, float r_max_sq,
    float rep_e, float contact_e, bool has_contact)
{
    float e00 = zone_e_pair(oix,oiy,oiz, ojx,ojy,ojz, box,inv_box,r_rep_sq,r_max_sq,rep_e,contact_e,has_contact);
    float e01 = zone_e_pair(oix,oiy,oiz, njx,njy,njz, box,inv_box,r_rep_sq,r_max_sq,rep_e,contact_e,has_contact);
    float e10 = zone_e_pair(nix,niy,niz, ojx,ojy,ojz, box,inv_box,r_rep_sq,r_max_sq,rep_e,contact_e,has_contact);
    float e11 = zone_e_pair(nix,niy,niz, njx,njy,njz, box,inv_box,r_rep_sq,r_max_sq,rep_e,contact_e,has_contact);
    return e11 - e01 - e10 + e00;
}

extern "C" __global__ void correction_pairs_kernel(
    const float* __restrict__ old_ca,   // [B,M,3]
    const float* __restrict__ new_ca,
    const float* __restrict__ old_sg,
    const float* __restrict__ new_sg,
    const long long* __restrict__ n_moved,    // [B]
    const long long* __restrict__ chain_idx,  // [B]
    const long long* __restrict__ bead_start, // [B]
    const float* __restrict__ ref,            // [B,3]
    const float* __restrict__ radius,         // [B]
    float* __restrict__ corr,                 // [B,B] out
    int M, int B,
    float box, float inv_box,
    float r_rep_sq, float r_max_sq, float rep_e, float contact_e,
    float cutoff,
    int min_caca, int min_casg, int min_sgsg)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= B || j >= B) return;
    if (j <= i) return;            // upper triangle only
    int tid = threadIdx.x;

    int nm_i = (int)n_moved[i];
    int nm_j = (int)n_moved[j];
    if (nm_i == 0 || nm_j == 0) {
        if (tid == 0) corr[i*B+j] = 0.0f;
        return;
    }

    // Exact bounding-sphere prefilter: if the two proposals' bounding
    // spheres are farther apart than the interaction cutoff, every
    // moved-bead pair distance exceeds the cutoff -> all four cross
    // energies are zero -> corr[i,j] is exactly zero.
    float dcx = wrap_mic(ref[i*3+0] - ref[j*3+0], box, inv_box);
    float dcy = wrap_mic(ref[i*3+1] - ref[j*3+1], box, inv_box);
    float dcz = wrap_mic(ref[i*3+2] - ref[j*3+2], box, inv_box);
    float dc = sqrtf(dcx*dcx + dcy*dcy + dcz*dcz);
    if (dc > radius[i] + radius[j] + cutoff) {
        if (tid == 0) corr[i*B+j] = 0.0f;
        return;
    }

    int ci = (int)chain_idx[i];
    int cj = (int)chain_idx[j];
    int ms_i = (int)bead_start[i];
    int ms_j = (int)bead_start[j];
    bool same_chain = (ci == cj);
    bool has_contact = (contact_e != 0.0f);

    int total = nm_i * nm_j;
    float local = 0.0f;
    for (int p = tid; p < total; p += blockDim.x) {
        int a = p / nm_j;
        int bb = p - a * nm_j;
        int seq = abs((ms_i + a) - (ms_j + bb));

        int oi = (i*M + a) * 3;
        int oj = (j*M + bb) * 3;
        float oicx=old_ca[oi+0], oicy=old_ca[oi+1], oicz=old_ca[oi+2];
        float nicx=new_ca[oi+0], nicy=new_ca[oi+1], nicz=new_ca[oi+2];
        float oisx=old_sg[oi+0], oisy=old_sg[oi+1], oisz=old_sg[oi+2];
        float nisx=new_sg[oi+0], nisy=new_sg[oi+1], nisz=new_sg[oi+2];
        float ojcx=old_ca[oj+0], ojcy=old_ca[oj+1], ojcz=old_ca[oj+2];
        float njcx=new_ca[oj+0], njcy=new_ca[oj+1], njcz=new_ca[oj+2];
        float ojsx=old_sg[oj+0], ojsy=old_sg[oj+1], ojsz=old_sg[oj+2];
        float njsx=new_sg[oj+0], njsy=new_sg[oj+1], njsz=new_sg[oj+2];

        // Calpha(i) vs Calpha(j)
        if (!same_chain || seq >= min_caca)
            local += cross4(oicx,oicy,oicz, nicx,nicy,nicz,
                            ojcx,ojcy,ojcz, njcx,njcy,njcz,
                            box,inv_box,r_rep_sq,r_max_sq,rep_e,contact_e,has_contact);
        // Calpha(i) vs SG(j)
        if (!same_chain || seq >= min_casg)
            local += cross4(oicx,oicy,oicz, nicx,nicy,nicz,
                            ojsx,ojsy,ojsz, njsx,njsy,njsz,
                            box,inv_box,r_rep_sq,r_max_sq,rep_e,contact_e,has_contact);
        // SG(i) vs Calpha(j)
        if (!same_chain || seq >= min_casg)
            local += cross4(oisx,oisy,oisz, nisx,nisy,nisz,
                            ojcx,ojcy,ojcz, njcx,njcy,njcz,
                            box,inv_box,r_rep_sq,r_max_sq,rep_e,contact_e,has_contact);
        // SG(i) vs SG(j)
        if (!same_chain || seq >= min_sgsg)
            local += cross4(oisx,oisy,oisz, nisx,nisy,nisz,
                            ojsx,ojsy,ojsz, njsx,njsy,njsz,
                            box,inv_box,r_rep_sq,r_max_sq,rep_e,contact_e,has_contact);
    }

    extern __shared__ float sdata[];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) corr[i*B+j] = sdata[0];
}
"""


# ---------------------------------------------------------------------------
# On-GPU causal accept (g1m_ccx optimization, Workstream C).
# Single-block kernel replacing the host-side `_fused_accept` loop and the
# per-batch delta_e/corr GPU->host round-trips. Sequential over proposal i
# (the causal dependency is inherent to the rank-1 EMM scheme); on accept,
# threads update delta_e[j] += corr[i,j] for j>i in parallel and scatter the
# accepted move straight into GPU-resident state. Per-move-type
# attempted/accepted counters are accumulated with atomics.
# ---------------------------------------------------------------------------

_CAUSAL_ACCEPT_SRC = COMMON + r"""
extern "C" __global__ void causal_accept_kernel(
    float* __restrict__ ca,                   // [n_chains*N, 3] mutated in place
    float* __restrict__ sg,
    const float* __restrict__ new_ca,         // [B, M, 3]
    const float* __restrict__ new_sg,
    float* __restrict__ delta_e,              // [B] mutated (rank-1 updates)
    const float* __restrict__ corr,           // [B, B] upper-triangular
    const long long* __restrict__ n_moved,    // [B]
    const long long* __restrict__ move_type,  // [B]
    const long long* __restrict__ chain_idx,  // [B]
    const long long* __restrict__ bead_start, // [B]
    const float* __restrict__ uniforms,       // [B] (consumed densely over nm>0)
    long long* __restrict__ attempted_counts, // [3] hinge,n_tail,c_tail
    long long* __restrict__ accepted_counts,  // [3]
    int M, int B, int N, float inv_kBT)
{
    int tid = threadIdx.x;
    int blk = blockDim.x;
    __shared__ int sh_u_idx;
    __shared__ int sh_accept;
    if (tid == 0) sh_u_idx = 0;
    __syncthreads();

    for (int i = 0; i < B; i++) {
        int nm = (int)n_moved[i];
        int mt = (int)move_type[i];
        if (tid == 0 && mt >= 0 && mt <= 2)
            atomicAdd((unsigned long long*)&attempted_counts[mt], (unsigned long long)1ULL);
        if (nm == 0) continue;          // all threads agree (n_moved is uniform)

        if (tid == 0) {
            float de = delta_e[i];
            float prob;
            if (de <= 0.0f) {
                prob = 1.0f;
            } else {
                float e = -de * inv_kBT;
                if (e <= -88.7f) prob = 0.0f;
                else if (e >= 88.7f) prob = 1.0f;
                else prob = expf(e);
            }
            sh_accept = (uniforms[sh_u_idx] < prob) ? 1 : 0;
            sh_u_idx += 1;
            if (sh_accept && mt >= 0 && mt <= 2)
                atomicAdd((unsigned long long*)&accepted_counts[mt], (unsigned long long)1ULL);
        }
        __syncthreads();

        if (sh_accept) {
            // rank-1 update: propagate accepted move into downstream delta_e.
            for (int j = i + 1 + tid; j < B; j += blk) {
                if (n_moved[j] == 0) continue;
                float c = corr[i * B + j];
                if (c != 0.0f) delta_e[j] += c;
            }
            // scatter the accepted move straight into GPU-resident state.
            int ci = (int)chain_idx[i];
            int ms = (int)bead_start[i];
            for (int k = tid; k < nm; k += blk) {
                int g = (ci * N + ms + k) * 3;
                int s = (i * M + k) * 3;
                ca[g+0] = new_ca[s+0];
                ca[g+1] = new_ca[s+1];
                ca[g+2] = new_ca[s+2];
                sg[g+0] = new_sg[s+0];
                sg[g+1] = new_sg[s+1];
                sg[g+2] = new_sg[s+2];
            }
        }
        __syncthreads();   // delta_e[j] writes visible before next i reads them
    }
}
"""


_propose_kernel = None
_delta_e_kernel = None
_accept_kernel = None
_mc_sweep_kernel = None
_corr_bounds_kernel = None
_corr_pairs_kernel = None
_causal_accept_kernel = None


def get_propose_kernel():
    global _propose_kernel
    if _propose_kernel is None:
        _propose_kernel = cp.RawKernel(_PROPOSE_SRC, "propose_batch_kernel")
    return _propose_kernel


def get_delta_e_kernel():
    global _delta_e_kernel
    if _delta_e_kernel is None:
        _delta_e_kernel = cp.RawKernel(_DELTA_E_SRC, "delta_e_segment_kernel")
    return _delta_e_kernel


def get_accept_kernel():
    global _accept_kernel
    if _accept_kernel is None:
        _accept_kernel = cp.RawKernel(_ACCEPT_SRC, "accept_segment_kernel")
    return _accept_kernel


def get_mc_sweep_kernel():
    global _mc_sweep_kernel
    if _mc_sweep_kernel is None:
        _mc_sweep_kernel = cp.RawKernel(_MC_SWEEP_SRC, "mc_sweep_kernel")
    return _mc_sweep_kernel


def get_corr_bounds_kernel():
    global _corr_bounds_kernel
    if _corr_bounds_kernel is None:
        _corr_bounds_kernel = cp.RawKernel(_CORR_BOUNDS_SRC, "corr_bounds_kernel")
    return _corr_bounds_kernel


def get_corr_pairs_kernel():
    global _corr_pairs_kernel
    if _corr_pairs_kernel is None:
        _corr_pairs_kernel = cp.RawKernel(_CORR_PAIRS_SRC, "correction_pairs_kernel")
    return _corr_pairs_kernel


def get_causal_accept_kernel():
    global _causal_accept_kernel
    if _causal_accept_kernel is None:
        _causal_accept_kernel = cp.RawKernel(_CAUSAL_ACCEPT_SRC, "causal_accept_kernel")
    return _causal_accept_kernel

"""
CUDA C kernels for GPU fast mode, compiled via CuPy RawKernel.

All kernels operate on FP32 positions for energy computation.
Position state is FP64 on host but converted to FP32 for kernel input.

Kernel inventory:
  - segment_proposal_kernel: generate B segment move proposals in parallel
  - pivot_proposal_kernel: generate n_chains pivot proposals in parallel
  - delta_e_kernel: compute overlap counts for B proposals vs cell list
  - emm_kernel: compute 4 pairwise energy matrices [B, B] for rank-1
  - apply_moves_kernel: scatter accepted proposals into position array
"""

import cupy as cp

# ---------------------------------------------------------------------------
# Common CUDA C header shared by all kernels
# ---------------------------------------------------------------------------

_COMMON_HEADER = r"""
__device__ __forceinline__ float mic_wrap(float dx, float box, float inv_box) {
    return dx - box * rintf(dx * inv_box);
}

__device__ __forceinline__ void mic_delta(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float box, float inv_box,
    float &dx, float &dy, float &dz)
{
    dx = bx - ax; dy = by - ay; dz = bz - az;
    dx = mic_wrap(dx, box, inv_box);
    dy = mic_wrap(dy, box, inv_box);
    dz = mic_wrap(dz, box, inv_box);
}

__device__ __forceinline__ float wrap_coord(float x, float box, float half_box, float inv_box) {
    return x - box * floorf((x + half_box) * inv_box);
}
"""

# ---------------------------------------------------------------------------
# Segment proposal kernel
# ---------------------------------------------------------------------------

_SEGMENT_PROPOSAL_SRC = _COMMON_HEADER + r"""
// Rodrigues rotation: rotate vector (vx,vy,vz) around unit axis (ux,uy,uz) by angle
__device__ void rodrigues_rotate(
    float vx, float vy, float vz,
    float ux, float uy, float uz,
    float angle,
    float &rx, float &ry, float &rz)
{
    float c = cosf(angle);
    float s = sinf(angle);
    float dot = vx*ux + vy*uy + vz*uz;
    float cx_ = vy*uz - vz*uy;
    float cy_ = vz*ux - vx*uz;
    float cz_ = vx*uy - vy*ux;
    rx = vx*c + cx_*s + ux*dot*(1.0f - c);
    ry = vy*c + cy_*s + uy*dot*(1.0f - c);
    rz = vz*c + cz_*s + uz*dot*(1.0f - c);
}

/*
 * Segment proposal kernel: one thread block per proposal, 1 thread per block.
 *
 * For each proposal b in [0, B):
 *   1. Read segment metadata
 *   2. Copy old positions
 *   3. Sequential MIC unwrap from anchor
 *   4. Compute rotation axis
 *   5. Rodrigues rotation
 *   6. Wrap into box
 *
 * seg_type: 0=N_TERMINAL, 1=C_TERMINAL, 2=INNER (hinge), 3=BOTH
 * (matches SegmentInfo constants in chain.py)
 */
extern "C" __global__ void segment_proposal_kernel(
    const float* __restrict__ positions,  // [n_total, 3]
    float* __restrict__ old_pos,          // [B, max_moved, 3]
    float* __restrict__ new_pos,          // [B, max_moved, 3]
    const int* __restrict__ chain_idx,    // [B]
    const int* __restrict__ bead_start,   // [B]
    const int* __restrict__ n_moved,      // [B]
    const int* __restrict__ seg_type,     // [B] 0=N_TERM,1=C_TERM,2=INNER,3=BOTH
    const int* __restrict__ anchor_a,     // [B] primary anchor global index
    const int* __restrict__ anchor_b,     // [B] secondary anchor global index
    const float* __restrict__ rand_angles,// [B] random angle in [-max_angle, max_angle]
    float box, float inv_box, float half_box,
    int N, int max_moved, int B)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int nm = n_moved[b];
    if (nm == 0) return;

    int gs = chain_idx[b] * N + bead_start[b];  // global start of moved beads
    int a_idx = anchor_a[b];  // global anchor index
    int b_idx = anchor_b[b];  // global secondary anchor index
    int stype = seg_type[b];
    float angle = rand_angles[b];

    // Base pointers for this proposal
    float* old_p = old_pos + b * max_moved * 3;
    float* new_p = new_pos + b * max_moved * 3;

    // Copy old positions
    for (int k = 0; k < nm; k++) {
        int gi = gs + k;
        old_p[k*3+0] = positions[gi*3+0];
        old_p[k*3+1] = positions[gi*3+1];
        old_p[k*3+2] = positions[gi*3+2];
    }

    // Anchor position
    float ax = positions[a_idx*3+0];
    float ay = positions[a_idx*3+1];
    float az = positions[a_idx*3+2];

    // Sequential MIC unwrap from anchor
    // unwrapped[nm][3] in local memory
    float unwrapped[1024 * 3];  // max 1024 beads per segment

    if (stype == 0 || stype == 3) {
        // N_TERMINAL(0) or BOTH(3): anchor is AFTER the segment, unwrap backward
        // Anchor is at anchor_a = min(seg_end, N-1)
        int last = nm - 1;
        int gi_last = gs + last;
        float dx, dy, dz;
        mic_delta(ax, ay, az,
                  positions[gi_last*3+0], positions[gi_last*3+1], positions[gi_last*3+2],
                  box, inv_box, dx, dy, dz);
        unwrapped[last*3+0] = ax + dx;
        unwrapped[last*3+1] = ay + dy;
        unwrapped[last*3+2] = az + dz;
        for (int k = last - 1; k >= 0; k--) {
            int gi = gs + k;
            int gi_next = gs + k + 1;
            dx = positions[gi*3+0] - positions[gi_next*3+0];
            dy = positions[gi*3+1] - positions[gi_next*3+1];
            dz = positions[gi*3+2] - positions[gi_next*3+2];
            dx = mic_wrap(dx, box, inv_box);
            dy = mic_wrap(dy, box, inv_box);
            dz = mic_wrap(dz, box, inv_box);
            unwrapped[k*3+0] = unwrapped[(k+1)*3+0] + dx;
            unwrapped[k*3+1] = unwrapped[(k+1)*3+1] + dy;
            unwrapped[k*3+2] = unwrapped[(k+1)*3+2] + dz;
        }
    } else {
        // INNER(2) or C_TERMINAL(1): anchor is BEFORE the segment, unwrap forward
        float dx, dy, dz;
        mic_delta(ax, ay, az,
                  positions[gs*3+0], positions[gs*3+1], positions[gs*3+2],
                  box, inv_box, dx, dy, dz);
        unwrapped[0] = ax + dx;
        unwrapped[1] = ay + dy;
        unwrapped[2] = az + dz;
        for (int k = 1; k < nm; k++) {
            int gi = gs + k;
            int gi_prev = gs + k - 1;
            dx = positions[gi*3+0] - positions[gi_prev*3+0];
            dy = positions[gi*3+1] - positions[gi_prev*3+1];
            dz = positions[gi*3+2] - positions[gi_prev*3+2];
            dx = mic_wrap(dx, box, inv_box);
            dy = mic_wrap(dy, box, inv_box);
            dz = mic_wrap(dz, box, inv_box);
            unwrapped[k*3+0] = unwrapped[(k-1)*3+0] + dx;
            unwrapped[k*3+1] = unwrapped[(k-1)*3+1] + dy;
            unwrapped[k*3+2] = unwrapped[(k-1)*3+2] + dz;
        }
    }

    // Compute rotation axis
    float ux, uy, uz, axis_len;
    float AXIS_EPS = 1e-7f;

    if (stype == 2) {
        // INNER (hinge): axis from anchor_a to anchor_b (through unwrapped chain)
        // Unwrap anchor_b from the last unwrapped bead via chain path
        float dx = positions[b_idx*3+0] - positions[(gs + nm - 1)*3+0];
        float dy = positions[b_idx*3+1] - positions[(gs + nm - 1)*3+1];
        float dz = positions[b_idx*3+2] - positions[(gs + nm - 1)*3+2];
        dx = mic_wrap(dx, box, inv_box);
        dy = mic_wrap(dy, box, inv_box);
        dz = mic_wrap(dz, box, inv_box);
        float ubx = unwrapped[(nm-1)*3+0] + dx;
        float uby = unwrapped[(nm-1)*3+1] + dy;
        float ubz = unwrapped[(nm-1)*3+2] + dz;
        ux = ubx - ax;
        uy = uby - ay;
        uz = ubz - az;
    } else {
        // N_TERMINAL(0), C_TERMINAL(1), or BOTH(3): axis from anchor_a to anchor_b
        float dx, dy, dz;
        mic_delta(ax, ay, az,
                  positions[b_idx*3+0], positions[b_idx*3+1], positions[b_idx*3+2],
                  box, inv_box, dx, dy, dz);
        ux = dx; uy = dy; uz = dz;
    }

    axis_len = sqrtf(ux*ux + uy*uy + uz*uz);

    if (axis_len < AXIS_EPS) {
        // Degenerate axis: copy old positions as new (no-op move)
        for (int k = 0; k < nm; k++) {
            new_p[k*3+0] = old_p[k*3+0];
            new_p[k*3+1] = old_p[k*3+1];
            new_p[k*3+2] = old_p[k*3+2];
        }
        return;
    }

    float inv_len = 1.0f / axis_len;
    ux *= inv_len; uy *= inv_len; uz *= inv_len;

    // Rotate unwrapped positions around anchor and wrap into box
    for (int k = 0; k < nm; k++) {
        float vx = unwrapped[k*3+0] - ax;
        float vy = unwrapped[k*3+1] - ay;
        float vz = unwrapped[k*3+2] - az;
        float rx, ry, rz;
        rodrigues_rotate(vx, vy, vz, ux, uy, uz, angle, rx, ry, rz);
        new_p[k*3+0] = wrap_coord(ax + rx, box, half_box, inv_box);
        new_p[k*3+1] = wrap_coord(ay + ry, box, half_box, inv_box);
        new_p[k*3+2] = wrap_coord(az + rz, box, half_box, inv_box);
    }
}
"""

# ---------------------------------------------------------------------------
# Pivot proposal kernel
# ---------------------------------------------------------------------------

_PIVOT_PROPOSAL_SRC = _COMMON_HEADER + r"""
/*
 * Pivot proposal kernel: one thread block per chain, 1 thread per block.
 *
 * Each thread:
 *   1. Pick random pivot index and side from pre-generated randoms
 *   2. Unwrap chain half from pivot (sequential MIC)
 *   3. Build SO(3) rotation matrix (Marsaglia uniform random rotation)
 *   4. Rotate, wrap, write new positions
 *
 * rand_buf layout per chain: [pivot_frac, side_frac, u, v, angle_frac, spare]
 */
extern "C" __global__ void pivot_proposal_kernel(
    const float* __restrict__ positions,  // [n_chains * N, 3]
    float* __restrict__ old_pos,          // [n_chains, max_moved, 3]
    float* __restrict__ new_pos,          // [n_chains, max_moved, 3]
    int* __restrict__ rot_start_out,      // [n_chains] output: start bead index
    int* __restrict__ n_moved_out,        // [n_chains] output: number moved
    const float* __restrict__ rand_buf,   // [n_chains, 6]
    const int* __restrict__ chain_perm,   // [n_chains] permuted chain indices
    float box, float inv_box, float half_box,
    int N, int n_chains, int max_moved)
{
    int tid = blockIdx.x;
    if (tid >= n_chains) return;

    int c = chain_perm[tid];
    int chain_base = c * N;

    // Read pre-generated random values
    float pivot_frac = rand_buf[tid * 6 + 0];
    float side_frac  = rand_buf[tid * 6 + 1];
    float ru         = rand_buf[tid * 6 + 2];
    float rv         = rand_buf[tid * 6 + 3];
    float angle_frac = rand_buf[tid * 6 + 4];

    if (N <= 2) {
        n_moved_out[tid] = 0;
        rot_start_out[tid] = 0;
        return;
    }

    // Pick pivot index in [1, N-2]
    int pivot_idx = 1 + (int)(pivot_frac * (N - 2));
    if (pivot_idx > N - 2) pivot_idx = N - 2;

    int side = (side_frac < 0.5f) ? 0 : 1;

    int rot_start, rot_end;
    if (side == 0) {
        rot_start = 0;
        rot_end = pivot_idx;
    } else {
        rot_start = pivot_idx + 1;
        rot_end = N;
    }

    int n_rot = rot_end - rot_start;
    rot_start_out[tid] = rot_start;
    n_moved_out[tid] = n_rot;

    if (n_rot == 0) return;

    // Pivot position (anchor)
    int pivot_global = chain_base + pivot_idx;
    float px = positions[pivot_global*3+0];
    float py = positions[pivot_global*3+1];
    float pz = positions[pivot_global*3+2];

    // Copy old positions
    float* old_p = old_pos + tid * max_moved * 3;
    float* new_p = new_pos + tid * max_moved * 3;
    for (int k = 0; k < n_rot; k++) {
        int gi = chain_base + rot_start + k;
        old_p[k*3+0] = positions[gi*3+0];
        old_p[k*3+1] = positions[gi*3+1];
        old_p[k*3+2] = positions[gi*3+2];
    }

    // Sequential unwrap from pivot along chain backbone
    // Process beads in chain order from pivot outward
    float unwrapped[1024 * 3];  // max 1024 beads

    if (side == 0) {
        // N-terminal side: unwrap backward from pivot-1 to 0
        // First bead is pivot-1 (closest to pivot)
        int first_gi = chain_base + pivot_idx - 1;
        float dx, dy, dz;
        mic_delta(px, py, pz,
                  positions[first_gi*3+0], positions[first_gi*3+1], positions[first_gi*3+2],
                  box, inv_box, dx, dy, dz);
        // Map to array index: bead pivot_idx-1 is at array position n_rot-1
        unwrapped[(n_rot-1)*3+0] = px + dx;
        unwrapped[(n_rot-1)*3+1] = py + dy;
        unwrapped[(n_rot-1)*3+2] = pz + dz;
        for (int k = n_rot - 2; k >= 0; k--) {
            int gi = chain_base + rot_start + k;
            int gi_next = gi + 1;  // next bead toward pivot
            dx = positions[gi*3+0] - positions[gi_next*3+0];
            dy = positions[gi*3+1] - positions[gi_next*3+1];
            dz = positions[gi*3+2] - positions[gi_next*3+2];
            dx = mic_wrap(dx, box, inv_box);
            dy = mic_wrap(dy, box, inv_box);
            dz = mic_wrap(dz, box, inv_box);
            unwrapped[k*3+0] = unwrapped[(k+1)*3+0] + dx;
            unwrapped[k*3+1] = unwrapped[(k+1)*3+1] + dy;
            unwrapped[k*3+2] = unwrapped[(k+1)*3+2] + dz;
        }
    } else {
        // C-terminal side: unwrap forward from pivot+1 to N-1
        int first_gi = chain_base + pivot_idx + 1;
        float dx, dy, dz;
        mic_delta(px, py, pz,
                  positions[first_gi*3+0], positions[first_gi*3+1], positions[first_gi*3+2],
                  box, inv_box, dx, dy, dz);
        unwrapped[0] = px + dx;
        unwrapped[1] = py + dy;
        unwrapped[2] = pz + dz;
        for (int k = 1; k < n_rot; k++) {
            int gi = chain_base + rot_start + k;
            int gi_prev = gi - 1;
            dx = positions[gi*3+0] - positions[gi_prev*3+0];
            dy = positions[gi*3+1] - positions[gi_prev*3+1];
            dz = positions[gi*3+2] - positions[gi_prev*3+2];
            dx = mic_wrap(dx, box, inv_box);
            dy = mic_wrap(dy, box, inv_box);
            dz = mic_wrap(dz, box, inv_box);
            unwrapped[k*3+0] = unwrapped[(k-1)*3+0] + dx;
            unwrapped[k*3+1] = unwrapped[(k-1)*3+1] + dy;
            unwrapped[k*3+2] = unwrapped[(k-1)*3+2] + dz;
        }
    }

    // Build random SO(3) rotation matrix (Marsaglia method)
    // ru, rv are uniform in [0,1) — transform to [-1,1) for Marsaglia
    float u = 2.0f * ru - 1.0f;
    float v = 2.0f * rv - 1.0f;
    float s2 = u*u + v*v;

    // If s2 >= 1 or too small, use identity rotation (rare, ~21% rejection)
    // This is acceptable because it just means this pivot is a no-op
    float R00, R01, R02, R10, R11, R12, R20, R21, R22;
    if (s2 >= 1.0f || s2 < 1e-10f) {
        // Identity rotation
        R00=1; R01=0; R02=0;
        R10=0; R11=1; R12=0;
        R20=0; R21=0; R22=1;
    } else {
        float factor = 2.0f * sqrtf(1.0f - s2);
        float ux = u * factor;
        float uy = v * factor;
        float uz = 1.0f - 2.0f * s2;
        float angle = 2.0f * 3.14159265358979f * angle_frac;
        float c = cosf(angle);
        float s = sinf(angle);
        float t = 1.0f - c;
        R00 = t*ux*ux + c;      R01 = t*ux*uy - s*uz;  R02 = t*ux*uz + s*uy;
        R10 = t*ux*uy + s*uz;   R11 = t*uy*uy + c;     R12 = t*uy*uz - s*ux;
        R20 = t*ux*uz - s*uy;   R21 = t*uy*uz + s*ux;  R22 = t*uz*uz + c;
    }

    // Rotate unwrapped positions around pivot and wrap into box
    for (int k = 0; k < n_rot; k++) {
        float vx = unwrapped[k*3+0] - px;
        float vy = unwrapped[k*3+1] - py;
        float vz = unwrapped[k*3+2] - pz;
        float rx = R00*vx + R01*vy + R02*vz;
        float ry = R10*vx + R11*vy + R12*vz;
        float rz = R20*vx + R21*vy + R22*vz;
        new_p[k*3+0] = wrap_coord(px + rx, box, half_box, inv_box);
        new_p[k*3+1] = wrap_coord(py + ry, box, half_box, inv_box);
        new_p[k*3+2] = wrap_coord(pz + rz, box, half_box, inv_box);
    }
}
"""

# ---------------------------------------------------------------------------
# Delta-E kernel (cell-list based overlap counting)
# ---------------------------------------------------------------------------

_DELTA_E_SRC = _COMMON_HEADER + r"""
/*
 * Delta-E kernel: one thread block per proposal, 256 threads per block.
 *
 * Each block:
 *   1. Load moved beads (old + new) into shared memory
 *   2. Build visited-cell bitmap (thread 0, in global scratch)
 *   3. Cooperatively iterate over neighbor cell atoms
 *   4. Count overlaps (r^2 < r_rep_sq) for old and new positions
 *   5. delta_e = (new_count - old_count) * rep_e
 *
 * Uses a global-memory bitmap for visited cells to handle large nm (pivots).
 * Shared memory used only for moved bead positions + reduction arrays.
 */
extern "C" __global__ void delta_e_kernel(
    const float* __restrict__ all_pos,       // [n_total, 3]
    const float* __restrict__ old_pos,       // [B, max_moved, 3]
    const float* __restrict__ new_pos,       // [B, max_moved, 3]
    const int* __restrict__ n_moved,         // [B]
    const int* __restrict__ global_starts,   // [B] = chain_idx * N + bead_start
    const int* __restrict__ global_ends,     // [B] = global_start + n_moved
    float* __restrict__ delta_e_out,         // [B]
    const int* __restrict__ sorted_order,    // [n_total]
    const int* __restrict__ cell_starts,     // [nc3]
    const int* __restrict__ cell_counts,     // [nc3]
    const int* __restrict__ neighbor_offsets, // [nc3, 27]
    unsigned char* __restrict__ visited_bitmaps, // [B, nc3] scratch space
    int* __restrict__ visited_lists,         // [B, max_visited] scratch space
    int* __restrict__ n_visited_out,         // [B] scratch space
    float box, float inv_box, float half_box,
    float inv_cs, int nc, int nc3,
    float r_rep_sq, float rep_e,
    int max_moved, int B, int max_visited)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int nm = n_moved[b];
    if (nm == 0) {
        if (threadIdx.x == 0) delta_e_out[b] = 0.0f;
        return;
    }

    int gs = global_starts[b];
    int ge = global_ends[b];
    int nc_m1 = nc - 1;

    // Shared memory for moved bead positions (old + new)
    extern __shared__ float smem[];
    float* s_old = smem;
    float* s_new = smem + nm * 3;

    // Load moved beads into shared memory (cooperative)
    const float* old_p = old_pos + b * max_moved * 3;
    const float* new_p = new_pos + b * max_moved * 3;
    for (int i = threadIdx.x; i < nm * 3; i += blockDim.x) {
        s_old[i] = old_p[i];
        s_new[i] = new_p[i];
    }
    __syncthreads();

    // Thread 0 builds visited-cell list using global-memory bitmap
    unsigned char* bitmap = visited_bitmaps + (long long)b * nc3;
    int* vlist = visited_lists + (long long)b * max_visited;

    __shared__ int s_n_visited;

    if (threadIdx.x == 0) {
        // Clear bitmap (cooperative would be faster but simple first)
        for (int i = 0; i < nc3; i++) bitmap[i] = 0;

        int nv = 0;
        for (int pass = 0; pass < 2; pass++) {
            const float* sp = (pass == 0) ? s_old : s_new;
            for (int k = 0; k < nm; k++) {
                int cx = max(0, min(nc_m1, (int)((sp[k*3+0] + half_box) * inv_cs)));
                int cy = max(0, min(nc_m1, (int)((sp[k*3+1] + half_box) * inv_cs)));
                int cz = max(0, min(nc_m1, (int)((sp[k*3+2] + half_box) * inv_cs)));
                int cell = (cx * nc + cy) * nc + cz;
                for (int ni = 0; ni < 27; ni++) {
                    int nbr = neighbor_offsets[cell * 27 + ni];
                    if (!bitmap[nbr]) {
                        bitmap[nbr] = 1;
                        if (nv < max_visited)
                            vlist[nv++] = nbr;
                    }
                }
            }
        }
        s_n_visited = nv;
    }
    __syncthreads();

    int nv = s_n_visited;

    // Cooperative overlap counting across all threads
    int e_old_local = 0;
    int e_new_local = 0;

    for (int vi = threadIdx.x; vi < nv; vi += blockDim.x) {
        int c = vlist[vi];
        int cnt = cell_counts[c];
        if (cnt == 0) continue;
        int start = cell_starts[c];

        for (int i = 0; i < cnt; i++) {
            int atom = sorted_order[start + i];
            if (atom >= gs && atom < ge) continue;

            float ax = all_pos[atom*3+0];
            float ay = all_pos[atom*3+1];
            float az = all_pos[atom*3+2];

            for (int m = 0; m < nm; m++) {
                float dx = ax - s_old[m*3+0];
                float dy = ay - s_old[m*3+1];
                float dz = az - s_old[m*3+2];
                dx = mic_wrap(dx, box, inv_box);
                dy = mic_wrap(dy, box, inv_box);
                dz = mic_wrap(dz, box, inv_box);
                if (dx*dx + dy*dy + dz*dz < r_rep_sq)
                    e_old_local++;
            }

            for (int m = 0; m < nm; m++) {
                float dx = ax - s_new[m*3+0];
                float dy = ay - s_new[m*3+1];
                float dz = az - s_new[m*3+2];
                dx = mic_wrap(dx, box, inv_box);
                dy = mic_wrap(dy, box, inv_box);
                dz = mic_wrap(dz, box, inv_box);
                if (dx*dx + dy*dy + dz*dz < r_rep_sq)
                    e_new_local++;
            }
        }
    }

    // Block-level tree reduction
    __shared__ int s_e_old[256];
    __shared__ int s_e_new[256];
    s_e_old[threadIdx.x] = e_old_local;
    s_e_new[threadIdx.x] = e_new_local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_e_old[threadIdx.x] += s_e_old[threadIdx.x + stride];
            s_e_new[threadIdx.x] += s_e_new[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        delta_e_out[b] = (float)(s_e_new[0] - s_e_old[0]) * rep_e;
    }
}
"""

# ---------------------------------------------------------------------------
# Emm kernel (pairwise segment energy matrices for rank-1 corrections)
# ---------------------------------------------------------------------------

_EMM_SRC = _COMMON_HEADER + r"""
/*
 * Emm kernel: one thread block per (i,j) pair (upper triangle).
 *
 * Each block computes 4 overlap counts between proposals i and j:
 *   E00 = overlaps(old_i, old_j)
 *   E01 = overlaps(old_i, new_j)
 *   E10 = overlaps(new_i, old_j)
 *   E11 = overlaps(new_i, new_j)
 *
 * Grid: B*(B-1)/2 blocks. Block index maps to (i,j) pair.
 * Block dim: 256 threads, cooperative over bead pairs.
 */
extern "C" __global__ void emm_kernel(
    const float* __restrict__ old_pos,    // [B, max_moved, 3]
    const float* __restrict__ new_pos,    // [B, max_moved, 3]
    const int* __restrict__ n_moved,      // [B]
    float* __restrict__ emm00,            // [B, B]
    float* __restrict__ emm01,            // [B, B]
    float* __restrict__ emm10,            // [B, B]
    float* __restrict__ emm11,            // [B, B]
    const int* __restrict__ pair_i,       // [n_pairs] first index
    const int* __restrict__ pair_j,       // [n_pairs] second index
    float box, float inv_box,
    float r_rep_sq, float rep_e,
    int max_moved, int B, int n_pairs)
{
    int pid = blockIdx.x;
    if (pid >= n_pairs) return;

    int pi = pair_i[pid];
    int pj = pair_j[pid];
    int nm_i = n_moved[pi];
    int nm_j = n_moved[pj];

    if (nm_i == 0 || nm_j == 0) {
        if (threadIdx.x == 0) {
            emm00[pi * B + pj] = 0.0f;
            emm01[pi * B + pj] = 0.0f;
            emm10[pi * B + pj] = 0.0f;
            emm11[pi * B + pj] = 0.0f;
        }
        return;
    }

    const float* old_i = old_pos + pi * max_moved * 3;
    const float* new_i = new_pos + pi * max_moved * 3;
    const float* old_j = old_pos + pj * max_moved * 3;
    const float* new_j = new_pos + pj * max_moved * 3;

    // Total bead pairs to check
    int total_pairs = nm_i * nm_j;

    // Each thread handles a subset of bead pairs
    int c00 = 0, c01 = 0, c10 = 0, c11 = 0;

    for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
        int mi = idx / nm_j;
        int mj = idx % nm_j;

        float oix = old_i[mi*3+0], oiy = old_i[mi*3+1], oiz = old_i[mi*3+2];
        float nix = new_i[mi*3+0], niy = new_i[mi*3+1], niz = new_i[mi*3+2];
        float ojx = old_j[mj*3+0], ojy = old_j[mj*3+1], ojz = old_j[mj*3+2];
        float njx = new_j[mj*3+0], njy = new_j[mj*3+1], njz = new_j[mj*3+2];

        float dx, dy, dz, r2;

        // E00: old_i vs old_j
        dx = mic_wrap(ojx - oix, box, inv_box);
        dy = mic_wrap(ojy - oiy, box, inv_box);
        dz = mic_wrap(ojz - oiz, box, inv_box);
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < r_rep_sq) c00++;

        // E01: old_i vs new_j
        dx = mic_wrap(njx - oix, box, inv_box);
        dy = mic_wrap(njy - oiy, box, inv_box);
        dz = mic_wrap(njz - oiz, box, inv_box);
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < r_rep_sq) c01++;

        // E10: new_i vs old_j
        dx = mic_wrap(ojx - nix, box, inv_box);
        dy = mic_wrap(ojy - niy, box, inv_box);
        dz = mic_wrap(ojz - niz, box, inv_box);
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < r_rep_sq) c10++;

        // E11: new_i vs new_j
        dx = mic_wrap(njx - nix, box, inv_box);
        dy = mic_wrap(njy - niy, box, inv_box);
        dz = mic_wrap(njz - niz, box, inv_box);
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < r_rep_sq) c11++;
    }

    // Block reduction
    __shared__ int s00[256], s01[256], s10[256], s11[256];
    s00[threadIdx.x] = c00;
    s01[threadIdx.x] = c01;
    s10[threadIdx.x] = c10;
    s11[threadIdx.x] = c11;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s00[threadIdx.x] += s00[threadIdx.x + stride];
            s01[threadIdx.x] += s01[threadIdx.x + stride];
            s10[threadIdx.x] += s10[threadIdx.x + stride];
            s11[threadIdx.x] += s11[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        emm00[pi * B + pj] = (float)s00[0] * rep_e;
        emm01[pi * B + pj] = (float)s01[0] * rep_e;
        emm10[pi * B + pj] = (float)s10[0] * rep_e;
        emm11[pi * B + pj] = (float)s11[0] * rep_e;
    }
}
"""

# ---------------------------------------------------------------------------
# Apply moves kernel
# ---------------------------------------------------------------------------

_APPLY_MOVES_SRC = _COMMON_HEADER + r"""
/*
 * Apply accepted moves: scatter new positions into the main position array.
 * Also converts FP32 proposal positions back to FP64 state positions.
 *
 * One thread block per accepted move, blockDim threads cooperate on beads.
 */
extern "C" __global__ void apply_moves_kernel(
    double* __restrict__ positions,       // [n_total, 3] FP64 state
    const float* __restrict__ new_pos,    // [B, max_moved, 3] FP32 proposals
    const int* __restrict__ accepted_idx, // [n_accepted] indices into proposal batch
    const int* __restrict__ global_starts,// [B] = chain_idx * N + bead_start
    const int* __restrict__ n_moved,      // [B]
    int max_moved, int n_accepted)
{
    int a = blockIdx.x;
    if (a >= n_accepted) return;

    int bidx = accepted_idx[a];
    int gs = global_starts[bidx];
    int nm = n_moved[bidx];
    const float* np = new_pos + bidx * max_moved * 3;

    for (int k = threadIdx.x; k < nm; k += blockDim.x) {
        int gi = gs + k;
        positions[gi*3+0] = (double)np[k*3+0];
        positions[gi*3+1] = (double)np[k*3+1];
        positions[gi*3+2] = (double)np[k*3+2];
    }
}
"""

# ---------------------------------------------------------------------------
# Also need a kernel to copy FP64 positions to FP32 for energy computation
# ---------------------------------------------------------------------------

_COPY_F64_TO_F32_SRC = r"""
extern "C" __global__ void copy_f64_to_f32(
    const double* __restrict__ src,   // [n, 3] FP64
    float* __restrict__ dst,          // [n, 3] FP32
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * 3) {
        dst[idx] = (float)src[idx];
    }
}
"""


# ---------------------------------------------------------------------------
# Kernel manager: compiles and caches all CuPy RawKernels
# ---------------------------------------------------------------------------

class GPUKernelManager:
    """Compiles and caches all CUDA kernels via CuPy RawKernel."""

    def __init__(self):
        self._kernels = {}
        self._compiled = False

    def compile_all(self):
        """Compile all kernels (one-time cost, cached by CuPy)."""
        if self._compiled:
            return

        self._kernels['segment_proposal'] = cp.RawKernel(
            _SEGMENT_PROPOSAL_SRC, 'segment_proposal_kernel')
        self._kernels['pivot_proposal'] = cp.RawKernel(
            _PIVOT_PROPOSAL_SRC, 'pivot_proposal_kernel')
        self._kernels['delta_e'] = cp.RawKernel(
            _DELTA_E_SRC, 'delta_e_kernel')
        self._kernels['emm'] = cp.RawKernel(
            _EMM_SRC, 'emm_kernel')
        self._kernels['apply_moves'] = cp.RawKernel(
            _APPLY_MOVES_SRC, 'apply_moves_kernel')
        self._kernels['copy_f64_to_f32'] = cp.RawKernel(
            _COPY_F64_TO_F32_SRC, 'copy_f64_to_f32')

        self._compiled = True

    def __getitem__(self, name: str) -> cp.RawKernel:
        if not self._compiled:
            self.compile_all()
        return self._kernels[name]

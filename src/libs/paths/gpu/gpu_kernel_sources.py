"""GPUKernelSources — CUDA C kernel source strings for the GPU fast backend.

All sources are class constants so the class itself is the single public surface.
No CuPy dependency — this module imports cleanly on CuPy-less machines.
"""


class GPUKernelSources:
    COMMON_HEADER = r"""
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

    SEGMENT_PROPOSAL = r"""
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

extern "C" __global__ void segment_proposal_kernel(
    const float* __restrict__ positions,
    float* __restrict__ old_pos,
    float* __restrict__ new_pos,
    const int* __restrict__ chain_idx,
    const int* __restrict__ bead_start,
    const int* __restrict__ n_moved,
    const int* __restrict__ seg_type,
    const int* __restrict__ anchor_a,
    const int* __restrict__ anchor_b,
    const float* __restrict__ rand_angles,
    float box, float inv_box, float half_box,
    float max_disp,
    int N, int max_moved, int B)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int nm = n_moved[b];
    if (nm == 0) return;

    int gs = chain_idx[b] * N + bead_start[b];
    int a_idx = anchor_a[b];
    int b_idx = anchor_b[b];
    int stype = seg_type[b];
    float angle = rand_angles[b];

    float* old_p = old_pos + b * max_moved * 3;
    float* new_p = new_pos + b * max_moved * 3;

    for (int k = 0; k < nm; k++) {
        int gi = gs + k;
        old_p[k*3+0] = positions[gi*3+0];
        old_p[k*3+1] = positions[gi*3+1];
        old_p[k*3+2] = positions[gi*3+2];
    }

    float ax = positions[a_idx*3+0];
    float ay = positions[a_idx*3+1];
    float az = positions[a_idx*3+2];

    float unwrapped[1024 * 3];

    if (stype == 0 || stype == 3) {
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

    float ux, uy, uz, axis_len;
    float AXIS_EPS = 1e-7f;

    if (stype == 2) {
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
        float dx, dy, dz;
        mic_delta(ax, ay, az,
                  positions[b_idx*3+0], positions[b_idx*3+1], positions[b_idx*3+2],
                  box, inv_box, dx, dy, dz);
        ux = dx; uy = dy; uz = dz;
    }

    axis_len = sqrtf(ux*ux + uy*uy + uz*uz);

    if (axis_len < AXIS_EPS) {
        for (int k = 0; k < nm; k++) {
            new_p[k*3+0] = old_p[k*3+0];
            new_p[k*3+1] = old_p[k*3+1];
            new_p[k*3+2] = old_p[k*3+2];
        }
        return;
    }

    float inv_len = 1.0f / axis_len;
    ux *= inv_len; uy *= inv_len; uz *= inv_len;

    float max_perp2 = 0.0f;
    for (int k = 0; k < nm; k++) {
        float vx = unwrapped[k*3+0] - ax;
        float vy = unwrapped[k*3+1] - ay;
        float vz = unwrapped[k*3+2] - az;
        float proj = vx*ux + vy*uy + vz*uz;
        float px = vx - proj*ux;
        float py = vy - proj*uy;
        float pz = vz - proj*uz;
        float p2 = px*px + py*py + pz*pz;
        if (p2 > max_perp2) max_perp2 = p2;
    }
    float max_perp = sqrtf(max_perp2);
    if (max_perp > 1e-12f) {
        float s_required = max_disp / (2.0f * max_perp);
        if (s_required < 1.0f) {
            float s_actual = fabsf(sinf(0.5f * angle));
            if (s_actual > s_required) {
                float theta_cap = 2.0f * asinf(s_required);
                angle = copysignf(theta_cap, angle);
            }
        }
    }

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

    PIVOT_PROPOSAL = r"""
extern "C" __global__ void pivot_proposal_kernel(
    const float* __restrict__ positions,
    float* __restrict__ old_pos,
    float* __restrict__ new_pos,
    int* __restrict__ rot_start_out,
    int* __restrict__ n_moved_out,
    const float* __restrict__ rand_buf,
    const int* __restrict__ chain_perm,
    float box, float inv_box, float half_box,
    float max_disp,
    int N, int n_chains, int max_moved)
{
    int tid = blockIdx.x;
    if (tid >= n_chains) return;

    int c = chain_perm[tid];
    int chain_base = c * N;

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

    int pivot_global = chain_base + pivot_idx;
    float px = positions[pivot_global*3+0];
    float py = positions[pivot_global*3+1];
    float pz = positions[pivot_global*3+2];

    float* old_p = old_pos + tid * max_moved * 3;
    float* new_p = new_pos + tid * max_moved * 3;
    for (int k = 0; k < n_rot; k++) {
        int gi = chain_base + rot_start + k;
        old_p[k*3+0] = positions[gi*3+0];
        old_p[k*3+1] = positions[gi*3+1];
        old_p[k*3+2] = positions[gi*3+2];
    }

    float unwrapped[1024 * 3];

    if (side == 0) {
        int first_gi = chain_base + pivot_idx - 1;
        float dx, dy, dz;
        mic_delta(px, py, pz,
                  positions[first_gi*3+0], positions[first_gi*3+1], positions[first_gi*3+2],
                  box, inv_box, dx, dy, dz);
        unwrapped[(n_rot-1)*3+0] = px + dx;
        unwrapped[(n_rot-1)*3+1] = py + dy;
        unwrapped[(n_rot-1)*3+2] = pz + dz;
        for (int k = n_rot - 2; k >= 0; k--) {
            int gi = chain_base + rot_start + k;
            int gi_next = gi + 1;
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

    float u = 2.0f * ru - 1.0f;
    float v = 2.0f * rv - 1.0f;
    float s2 = u*u + v*v;

    float R00, R01, R02, R10, R11, R12, R20, R21, R22;
    if (s2 >= 1.0f || s2 < 1e-10f) {
        R00=1; R01=0; R02=0;
        R10=0; R11=1; R12=0;
        R20=0; R21=0; R22=1;
    } else {
        float factor = 2.0f * sqrtf(1.0f - s2);
        float ux = u * factor;
        float uy = v * factor;
        float uz = 1.0f - 2.0f * s2;
        float angle = 2.0f * 3.14159265358979f * angle_frac;
        float max_perp2 = 0.0f;
        for (int k = 0; k < n_rot; k++) {
            float vx = unwrapped[k*3+0] - px;
            float vy = unwrapped[k*3+1] - py;
            float vz = unwrapped[k*3+2] - pz;
            float proj = vx*ux + vy*uy + vz*uz;
            float qx = vx - proj*ux;
            float qy = vy - proj*uy;
            float qz = vz - proj*uz;
            float p2 = qx*qx + qy*qy + qz*qz;
            if (p2 > max_perp2) max_perp2 = p2;
        }
        float max_perp = sqrtf(max_perp2);
        if (max_perp > 1e-12f) {
            float s_required = max_disp / (2.0f * max_perp);
            if (s_required < 1.0f) {
                float s_actual = fabsf(sinf(0.5f * angle));
                if (s_actual > s_required) {
                    float theta_cap = 2.0f * asinf(s_required);
                    angle = copysignf(theta_cap, angle);
                }
            }
        }
        float c = cosf(angle);
        float s = sinf(angle);
        float t = 1.0f - c;
        R00 = t*ux*ux + c;      R01 = t*ux*uy - s*uz;  R02 = t*ux*uz + s*uy;
        R10 = t*ux*uy + s*uz;   R11 = t*uy*uy + c;     R12 = t*uy*uz - s*ux;
        R20 = t*ux*uz - s*uy;   R21 = t*uy*uz + s*ux;  R22 = t*uz*uz + c;
    }

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

    DELTA_E = r"""
extern "C" __global__ void delta_e_kernel(
    const float* __restrict__ all_pos,
    const float* __restrict__ old_pos,
    const float* __restrict__ new_pos,
    const int* __restrict__ n_moved,
    const int* __restrict__ global_starts,
    const int* __restrict__ global_ends,
    float* __restrict__ delta_e_out,
    const int* __restrict__ sorted_order,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_counts,
    const int* __restrict__ neighbor_offsets,
    unsigned char* __restrict__ visited_bitmaps,
    int* __restrict__ visited_lists,
    int* __restrict__ n_visited_out,
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

    extern __shared__ float smem[];
    float* s_old = smem;
    float* s_new = smem + nm * 3;

    const float* old_p = old_pos + b * max_moved * 3;
    const float* new_p = new_pos + b * max_moved * 3;
    for (int i = threadIdx.x; i < nm * 3; i += blockDim.x) {
        s_old[i] = old_p[i];
        s_new[i] = new_p[i];
    }
    __syncthreads();

    unsigned char* bitmap = visited_bitmaps + (long long)b * nc3;
    int* vlist = visited_lists + (long long)b * max_visited;

    __shared__ int s_n_visited;

    if (threadIdx.x == 0) {
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

    EMM_CORRECTION = r"""
extern "C" __global__ void emm_correction_kernel(
    const float* __restrict__ old_pos,
    const float* __restrict__ new_pos,
    const int* __restrict__ n_moved,
    float* __restrict__ correction,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
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
            correction[pi * B + pj] = 0.0f;
        }
        return;
    }

    const float* old_i = old_pos + pi * max_moved * 3;
    const float* new_i = new_pos + pi * max_moved * 3;
    const float* old_j = old_pos + pj * max_moved * 3;
    const float* new_j = new_pos + pj * max_moved * 3;

    int total_pairs = nm_i * nm_j;

    int corr_count = 0;

    for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
        int mi = idx / nm_j;
        int mj = idx % nm_j;

        float oix = old_i[mi*3+0], oiy = old_i[mi*3+1], oiz = old_i[mi*3+2];
        float nix = new_i[mi*3+0], niy = new_i[mi*3+1], niz = new_i[mi*3+2];
        float ojx = old_j[mj*3+0], ojy = old_j[mj*3+1], ojz = old_j[mj*3+2];
        float njx = new_j[mj*3+0], njy = new_j[mj*3+1], njz = new_j[mj*3+2];

        float dx, dy, dz, r2;

        dx = mic_wrap(ojx - oix, box, inv_box);
        dy = mic_wrap(ojy - oiy, box, inv_box);
        dz = mic_wrap(ojz - oiz, box, inv_box);
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < r_rep_sq) corr_count++;

        dx = mic_wrap(njx - oix, box, inv_box);
        dy = mic_wrap(njy - oiy, box, inv_box);
        dz = mic_wrap(njz - oiz, box, inv_box);
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < r_rep_sq) corr_count--;

        dx = mic_wrap(ojx - nix, box, inv_box);
        dy = mic_wrap(ojy - niy, box, inv_box);
        dz = mic_wrap(ojz - niz, box, inv_box);
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < r_rep_sq) corr_count--;

        dx = mic_wrap(njx - nix, box, inv_box);
        dy = mic_wrap(njy - niy, box, inv_box);
        dz = mic_wrap(njz - niz, box, inv_box);
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < r_rep_sq) corr_count++;
    }

    __shared__ int s_corr[256];
    s_corr[threadIdx.x] = corr_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_corr[threadIdx.x] += s_corr[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        correction[pi * B + pj] = (float)s_corr[0] * rep_e;
    }
}
"""

    APPLY_MOVES = r"""
extern "C" __global__ void apply_moves_kernel(
    double* __restrict__ positions,
    const float* __restrict__ new_pos,
    const int* __restrict__ accepted_idx,
    const int* __restrict__ global_starts,
    const int* __restrict__ n_moved,
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

    COPY_F64_TO_F32 = r"""
extern "C" __global__ void copy_f64_to_f32(
    const double* __restrict__ src,
    float* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * 3) {
        dst[idx] = (float)src[idx];
    }
}
"""

    @classmethod
    def segment_proposal_src(cls) -> str:
        return cls.COMMON_HEADER + cls.SEGMENT_PROPOSAL

    @classmethod
    def pivot_proposal_src(cls) -> str:
        return cls.COMMON_HEADER + cls.PIVOT_PROPOSAL

    @classmethod
    def delta_e_src(cls) -> str:
        return cls.COMMON_HEADER + cls.DELTA_E

    @classmethod
    def emm_correction_src(cls) -> str:
        return cls.COMMON_HEADER + cls.EMM_CORRECTION

    @classmethod
    def apply_moves_src(cls) -> str:
        return cls.COMMON_HEADER + cls.APPLY_MOVES

    @classmethod
    def copy_f64_to_f32_src(cls) -> str:
        return cls.COPY_F64_TO_F32

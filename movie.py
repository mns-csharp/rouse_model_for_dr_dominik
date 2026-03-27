"""
Snapshot collector and movie renderer for Rouse-model MC simulations.

Captures 3D bead positions at configurable intervals during the simulation,
then renders frames and stitches them into an MP4 movie.

Snapshot collection: ~0.1ms per save (tensor clone) — negligible overhead.
Movie rendering: done after simulation completes — zero impact on speed.
"""

import os
import torch
import numpy as np
import math


class SnapshotCollector:
    """Collects position snapshots during simulation for movie rendering."""

    def __init__(self, save_every: int = 1, max_frames: int = 2000):
        """
        Args:
            save_every: save a snapshot every N sweeps
            max_frames: cap total frames to limit memory usage
        """
        self.save_every = save_every
        self.max_frames = max_frames
        self.frames = []       # list of (sweep_idx, positions_numpy)
        self.phase = []        # 'eq' or 'prod' per frame
        self._frame_count = 0

    def capture(self, positions: torch.Tensor, sweep: int, phase: str):
        """Save a snapshot if this sweep is due. Call every sweep."""
        if self._frame_count >= self.max_frames:
            return
        if sweep % self.save_every == 0:
            # Detach, move to CPU, convert to numpy — ~0.1ms
            pos_np = positions.detach().cpu().numpy().copy()
            self.frames.append((sweep, pos_np))
            self.phase.append(phase)
            self._frame_count += 1

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def render_movie(collector: SnapshotCollector, output_path: str,
                 box_size: float, N: int, n_chains: int,
                 fps: int = 30, dpi: int = 100,
                 max_chains_shown: int = 50):
    """
    Render collected snapshots into an MP4 movie.

    Args:
        collector: SnapshotCollector with captured frames
        output_path: path for output .mp4 file
        box_size: simulation box size (Angstrom)
        N: beads per chain
        n_chains: total chains
        fps: frames per second in output movie
        dpi: resolution
        max_chains_shown: limit chains drawn for clarity (random subset)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    n_frames = collector.n_frames
    if n_frames == 0:
        print("  No frames captured, skipping movie.")
        return

    print(f"  Rendering {n_frames} frames to {output_path} ...", flush=True)

    # Pick a random subset of chains to display for visual clarity
    rng = np.random.RandomState(42)
    if n_chains > max_chains_shown:
        chain_indices = sorted(rng.choice(n_chains, max_chains_shown, replace=False))
    else:
        chain_indices = list(range(n_chains))

    # Color map: each chain gets a distinct color
    n_show = len(chain_indices)
    cmap = plt.cm.tab20(np.linspace(0, 1, min(20, n_show)))

    half = box_size / 2.0

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.cla()
        sweep, positions = collector.frames[frame_idx]
        phase = collector.phase[frame_idx]

        for ci_idx, ci in enumerate(chain_indices):
            chain_pos = positions[ci]  # [N, 3]
            color = cmap[ci_idx % len(cmap)]
            # Draw beads
            ax.scatter(chain_pos[:, 0], chain_pos[:, 1], chain_pos[:, 2],
                       s=8, color=color, alpha=0.6, depthshade=True)
            # Draw bonds (skip if beads are wrapped across PBC — large gap)
            for b in range(N - 1):
                dx = chain_pos[b + 1] - chain_pos[b]
                if np.abs(dx).max() < box_size * 0.4:
                    ax.plot([chain_pos[b, 0], chain_pos[b + 1, 0]],
                            [chain_pos[b, 1], chain_pos[b + 1, 1]],
                            [chain_pos[b, 2], chain_pos[b + 1, 2]],
                            color=color, alpha=0.3, linewidth=0.5)

        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_zlim(-half, half)
        ax.set_xlabel('X (A)')
        ax.set_ylabel('Y (A)')
        ax.set_zlabel('Z (A)')
        ax.set_title(f'Sweep {sweep} ({phase})  |  '
                     f'N={N}, {n_chains} chains, '
                     f'{n_show} shown',
                     fontsize=10)

        if (frame_idx + 1) % max(1, n_frames // 20) == 0 or frame_idx == 0:
            print(f"    Frame {frame_idx + 1}/{n_frames} "
                  f"[{100*(frame_idx+1)/n_frames:.0f}%]", flush=True)

        return []

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    try:
        writer = FFMpegWriter(fps=fps, metadata={'title': 'Rouse MC Simulation'})
        anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close(fig)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Movie saved: {output_path} ({size_mb:.1f} MB, {n_frames} frames)")
    except (RuntimeError, FileNotFoundError) as e:
        # FFmpeg not available — fall back to saving as GIF or individual PNGs
        plt.close(fig)
        print(f"  FFmpeg not available ({e}). Saving as GIF instead...", flush=True)
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim = FuncAnimation(fig, update, frames=min(n_frames, 200), blit=False)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            anim = FuncAnimation(fig, update, frames=min(n_frames, 200), blit=False)
            anim.save(gif_path, writer='pillow', fps=min(fps, 15), dpi=dpi // 2)
            plt.close(fig)
            print(f"  GIF saved: {gif_path}")
        except Exception as e2:
            plt.close(fig)
            print(f"  Could not save movie: {e2}")
            print(f"  Snapshots are available in collector.frames ({n_frames} frames)")

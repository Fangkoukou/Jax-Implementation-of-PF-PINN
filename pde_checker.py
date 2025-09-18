import jax
import jax.numpy as jnp
from jax import config
import diffrax as dfx

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm


class PDE_checker:
    """
    Utility class for visualizing and verifying PDE simulation results.
    Provides multiple visualization styles:
    - Heatmaps of solution fields
    - Temporal snapshots (frames)
    - Side-by-side solver comparison
    - Quantitative diagnostics with error plots
    """

    def heatmaps(self, sol, title=""):
        """
        Generate 2D heatmaps for phi and c fields.

        Args:
            sol (dict): Contains 'x', 't', 'c', 'phi' arrays.
            title (str): Label prefix for plots.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        fig.suptitle(f'PDE Simulation Results: {title}', fontsize=14)

        # Extract simulation results
        x, t, c, phi = sol["x"], sol["t"], sol["c"], sol["phi"]
        extent = [x[0], x[-1], t[0], t[-1]]

        # Phi heatmap
        im1 = ax[0].imshow(phi, aspect='auto', origin='lower', extent=extent)
        ax[0].set_title(f"{title} phi")
        ax[0].set_xlabel(f"{title} Position (x)")
        ax[0].set_ylabel(f"{title} Time (t)")
        fig.colorbar(im1, ax=ax[0])

        # C heatmap
        im2 = ax[1].imshow(c, aspect='auto', origin='lower', extent=extent)
        ax[1].set_title(f"{title} c")
        ax[1].set_xlabel(f"{title} Position (x)")
        ax[1].set_ylabel(f"{title} Time (t)")
        fig.colorbar(im2, ax=ax[1])

        plt.show()

    def frames(self, sol, title="", num_frames=5):
        """
        Plot 1D frames of phi and c at selected times.

        Args:
            sol (dict): Contains 'x', 't', 'c', 'phi' arrays.
            title (str): Label prefix for plots.
            num_frames (int): Number of time snapshots to plot.
        """
        fig, (ax_phi, ax_c) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
        fig.suptitle(f'PDE Simulation Frames: {title}', fontsize=14)

        # Extract data
        x, t, c, phi = sol["x"], sol["t"], sol["c"], sol["phi"]

        # Select evenly spaced time indices
        time_indices = jnp.linspace(0, len(t) - 1, num_frames, dtype=int)
        colors = cm.viridis(np.linspace(0, 1, num_frames))

        # Plot phi and c separately
        plot_specs = [
            {'ax': ax_phi, 'data': phi, 'title_suffix': 'phi', 'ylabel': 'phi'},
            {'ax': ax_c,   'data': c,   'title_suffix': 'c',   'ylabel': 'c'}
        ]

        for spec in plot_specs:
            ax = spec['ax']
            for i, time_idx in enumerate(time_indices):
                ax.plot(x, spec['data'][time_idx, :],
                        color=colors[i], linewidth=2.5,
                        label=f't = {t[time_idx]:.2e}')
            ax.set_title(f"{title} {spec['title_suffix']}")
            ax.set_xlabel("Position (x)")
            ax.set_ylabel(spec['ylabel'])
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='best', fontsize=9, title="Time")

        # Remove redundant y-labels on the right subplot
        ax_c.tick_params(labelleft=False)
        plt.show()

    def check1(self, results1, results2, label1="Solver 1", label2="Solver 2", num_frames=5):
        """
        Compare two solvers visually with side-by-side overlay.

        Args:
            results1 (dict): Contains 'x', 't', 'c', 'phi' from first solver.
            results2 (dict): Contains 'x', 'c', 'phi' from second solver.
            label1 (str): Name of first solver.
            label2 (str): Name of second solver.
            num_frames (int): Number of time snapshots to plot.
        """
        x1, t1, c1, phi1 = results1["x"], results1["t"], results1["c"], results1["phi"]
        x2, c2, phi2 = results2["x"], results2["c"], results2["phi"]

        # Time indices and colors
        time_indices = jnp.linspace(0, len(t1) - 1, num_frames, dtype=int)
        colors = cm.viridis(np.linspace(0, 1, num_frames))

        fig, (ax_phi, ax_c) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle('Comparison of Profiles Over Time', fontsize=14)

        # Plot phi and c with overlay
        plot_specs = [
            {'ax': ax_phi, 'data1': phi1, 'data2': phi2, 'title': r'Phase Field, $\phi$', 'ylabel': r'$\phi$'},
            {'ax': ax_c,   'data1': c1,   'data2': c2,   'title': 'Concentration, c', 'ylabel': 'Concentration'}
        ]
        
        for spec in plot_specs:
            ax = spec['ax']
            for i, time_idx in enumerate(time_indices):
                time_label = f't = {t1[time_idx]:.2e} s' if ax == ax_phi else None
                ax.plot(x1, spec['data1'][time_idx, :],
                        color=colors[i], linestyle='-', linewidth=4, alpha=0.9,
                        label=time_label)
                ax.plot(x2, spec['data2'][time_idx, :],
                        color='red', linestyle='--', linewidth=2, alpha=0.65)
            ax.set_title(spec['title'], fontsize=14)
            ax.set_ylabel(spec['ylabel'])
            ax.set_xlabel('Position x')
            ax.grid(True, linestyle=':', alpha=0.7)

        ax_c.tick_params(labelleft=False)

        # Build combined legend
        handles, labels = ax_phi.get_legend_handles_labels()
        overlay_handle = Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'{label2} (Overlay)')
        handles.append(overlay_handle)
        labels.append(f'{label2} (Overlay)')

        legend_kwargs = {'loc': 'best', 'fontsize': 10, 'title': label1}
        ax_phi.legend(handles=handles, labels=labels, **legend_kwargs)
        ax_c.legend(handles=handles, labels=labels, **legend_kwargs)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

    def _plot_1d_diagnostic(self, x_coords, sol1, sol2, abs_diff, t_fail,
                            var_name, var_symbol, label1, label2, log_scale):
        """
        Internal helper: plot 1D comparison and absolute error at a single time.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'Discrepancy in {var_name} ({var_symbol}) at t = {t_fail:.2e} s', fontsize=16)

        # Solutions
        ax1.plot(x_coords, sol1, 'b-', lw=4, alpha=0.7, label=label1)
        ax1.plot(x_coords, sol2, 'r--', lw=2, label=label2)
        ax1.set_ylabel(var_symbol)
        ax1.legend()
        ax1.grid(True, linestyle=':')

        # Absolute error
        ax2.plot(x_coords, abs_diff, 'k-')
        ax2.set_xlabel('Position x')
        ax2.set_ylabel(f'Absolute Error |Δ{var_symbol}|')
        ax2.grid(True, linestyle=':')
        if log_scale:
            ax2.set_yscale('log')
            ax2.set_ylabel(f'Abs Error |Δ{var_symbol}| (log)')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def check2(self, results1, results2, label1="first input", label2="second input",
               log_scale=False, rtol=1e-6, atol=1e-8):
        """
        Comprehensive quantitative verification between two solver results.

        - Computes max absolute differences
        - Checks tolerance compliance
        - Always plots 1D diagnostics at max-error times
        - Always plots 2D error heatmaps

        Args:
            results1 (dict): Reference results.
            results2 (dict): Comparison results.
            label1 (str): Label for first solver.
            label2 (str): Label for second solver.
            log_scale (bool): Whether to use log scale for error plots.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.
        """
        print("--- Starting Comprehensive Solver Verification ---")
        all_passed = True

        x_coords = results1["x"]
        abs_diff_phi = jnp.abs(results1["phi"] - results2["phi"])
        abs_diff_c = jnp.abs(results1["c"] - results2["c"])

        # Check both phi and c
        variables_to_check = [
            {"name": "Phase Field", "symbol": "φ", "data1": results1["phi"], "data2": results2["phi"], "diff": abs_diff_phi},
            {"name": "Concentration", "symbol": "c", "data1": results1["c"], "data2": results2["c"], "diff": abs_diff_c},
        ]

        for i, var in enumerate(variables_to_check):
            print(f"\n[{i+1}] Verifying {var['name']} solution '{var['symbol']}'...")
            max_abs_diff = jnp.max(var["diff"])
            t_idx, x_idx = jnp.unravel_index(jnp.argmax(var["diff"]), var["diff"].shape)
            print(f"   - Max absolute difference |Δ{var['symbol']}|: {max_abs_diff:.2e}")
            print(f"   - Location: time_index={t_idx}, space_index={x_idx}")

            if jnp.allclose(var["data1"], var["data2"], rtol=rtol, atol=atol):
                print("   - SUCCESS: Solutions are within tolerance.")
            else:
                all_passed = False
                print("   - FAILURE: Solutions exceed tolerance.")

            # Always generate 1D diagnostic
            t_fail = results1["t"][t_idx]
            print(f"   - Generating 1D diagnostic plot at t = {t_fail:.2e} s...")
            self._plot_1d_diagnostic(
                x_coords, var["data1"][t_idx, :], var["data2"][t_idx, :],
                var["diff"][t_idx, :], t_fail,
                var["name"], var["symbol"], label1, label2, log_scale
            )

        # Generate 2D error heatmaps
        print("\n[3] Generating 2D absolute error heatmaps...")
        fig_hm, (ax_phi_hm, ax_c_hm) = plt.subplots(1, 2, figsize=(14, 6))
        fig_hm.suptitle(f'Absolute Error Heatmaps |{label1} - {label2}|', fontsize=16)
        extent = [x_coords[0], x_coords[-1], results1["t"][0], results1["t"][-1]]

        norm_phi = LogNorm(vmin=jnp.min(abs_diff_phi[abs_diff_phi > 0]),
                           vmax=jnp.max(abs_diff_phi)) if log_scale and jnp.any(abs_diff_phi > 0) else None
        norm_c = LogNorm(vmin=jnp.min(abs_diff_c[abs_diff_c > 0]),
                         vmax=jnp.max(abs_diff_c)) if log_scale and jnp.any(abs_diff_c > 0) else None

        cbar_label_phi = 'Absolute Error |Δφ|' + (' (log scale)' if norm_phi else '')
        cbar_label_c = 'Absolute Error |Δc|' + (' (log scale)' if norm_c else '')

        im1 = ax_phi_hm.imshow(abs_diff_phi, aspect='auto', origin='lower', extent=extent, cmap='magma', norm=norm_phi)
        ax_phi_hm.set_title('Absolute Error in φ')
        ax_phi_hm.set_xlabel('Position x'); ax_phi_hm.set_ylabel('Time (s)')
        fig_hm.colorbar(im1, ax=ax_phi_hm, label=cbar_label_phi)

        im2 = ax_c_hm.imshow(abs_diff_c, aspect='auto', origin='lower', extent=extent, cmap='magma', norm=norm_c)
        ax_c_hm.set_title('Absolute Error in c')
        ax_c_hm.set_xlabel('Position x'); ax_c_hm.set_ylabel('Time (s)')
        fig_hm.colorbar(im2, ax=ax_c_hm, label=cbar_label_c)

        plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()

        # Summary
        print("\n--- Verification Complete ---")
        if all_passed:
            print("✅ Final Result: All checks passed within tolerance.")
        else:
            print("❌ Final Result: Some checks failed. See diagnostic plots.")
        print(f"\n[Summary] Maximum absolute errors:")
        print(f"   - |Δφ|_max = {jnp.max(abs_diff_phi):.2e}")
        print(f"   - |Δc|_max = {jnp.max(abs_diff_c):.2e}")

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import astropy.units as u
from astropy.table import Table
from astropy.modeling import models
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
from specutils import Spectrum
from specutils.fitting import fit_lines
from sklearn.metrics import r2_score

import warnings
from astropy.utils.exceptions import AstropyUserWarning

warnings.simplefilter('ignore', category=AstropyUserWarning)

coadd_dir = "/home/holdennix/Research/G165/MMT/pypeit/all_coadded"
PATTERN = "*.fits"
coadd_files = sorted(glob.glob(os.path.join(coadd_dir, PATTERN)))


def load_spectrum(filepath, smooth=False):
    '''
    This function is simply a way for me to easly retrieve all the
    data needed from the fits files.

    smooth: input whether or not you want a smoothed array. The stddev is hard
            coded in.
    '''
    spec_table = Table.read(filepath)
    spec_table = spec_table[spec_table['wave'] != 0]

    wave = np.asarray(spec_table["wave"]).astype(float).ravel()
    flux = np.asarray(spec_table["flux"]).astype(float).ravel()
    sigma = np.asarray(spec_table["sigma"]).astype(float).ravel()
    try:
        sky = np.asarray(spec_table["sky_flux"]).astype(float).ravel()
    except KeyError:
        sky = np.zeros(len(wave))

    if smooth:
        kernel = Gaussian1DKernel(stddev=3)
        smoothed_flux = convolve(flux, kernel)
        smoothed_sky = convolve(sky, kernel)

        kernel_squared_weights = kernel.array**2
        variance_convolved = convolve(sigma**2, kernel_squared_weights)
        smoothed_sigma = np.sqrt(variance_convolved)

        return wave, flux, sigma, sky, smoothed_flux, smoothed_sigma, smoothed_sky
    else:
        return wave, flux, sigma, sky
    

sky_lines = np.loadtxt('/home/holdennix/Research/G165/MMT/pipeline/skylines.txt', usecols=(0,))
window = 15
degree = 2
ncols = 4
n_lines = len(sky_lines)
nrows = int(np.ceil(n_lines / ncols))

with PdfPages("/home/holdennix/Research/G165/MMT/figures/sky_calibration_summary.pdf") as pdf:

    for k, filepath in enumerate(coadd_files):
        print(f"\r{k}/{len(coadd_files)-1}", end="", flush=True)

        # ---- Load spectrum ----
        wave, flux, err, sky, smooth_flux, smooth_err, smoothed_sky = load_spectrum(filepath, smooth=True)
        sky_spec = Spectrum(spectral_axis=wave * u.Angstrom, flux=sky * u.count / u.second)

        # ---- Set up one BIG figure per file, with 4 stacked sections ----
        fig = plt.figure(figsize=(4 * ncols, 4 + 3 * nrows + 8 + 3 * nrows))
        outer = fig.add_gridspec(
            4, 1,
            height_ratios=[4, 3 * nrows, 6, 3 * nrows]
        )

        # ============= Section 1: raw sky overview =============
        ax_sky = fig.add_subplot(outer[0])
        ax_sky.plot(wave, smoothed_sky, color="black")
        ax_sky.vlines(sky_lines, ymin=0, ymax=smoothed_sky.max(), color="gray", ls="--", lw=1)
        ax_sky.set_xlim(wave.min(), wave.max())
        ax_sky.set_title(f"Sky spectrum overview: {os.path.basename(filepath)}")

        # ============= Section 2: pre-calibration fit grid =============
        gs_pre = outer[1].subgridspec(nrows, ncols, hspace=0.6, wspace=0.3)
        true_l, fit_l = [], []
        i = 0
        for l in sky_lines:
            sub_region = (sky_spec.spectral_axis > l*u.AA - window*u.AA) & \
                         (sky_spec.spectral_axis < l*u.AA + window*u.AA)
            sub_wave = sky_spec.spectral_axis[sub_region]
            sub_flux = sky_spec.flux[sub_region]
            sub_spec = Spectrum(flux=sub_flux, spectral_axis=sub_wave)

            try:
                g_init = models.Gaussian1D(amplitude=sub_flux.max(), mean=l, stddev=4*u.AA) \
                        + models.Const1D(amplitude=sub_flux.min())
                g_fit = fit_lines(sub_spec, g_init, get_fit_info=True)
            except ValueError:
                continue

            y_fit = g_fit(sub_spec.spectral_axis)

            if len(sub_flux) < 2:
                continue  # not enough data points to evaluate the fit meaningfully
            r2 = r2_score(sub_flux.value, y_fit.value)

            plot_wave_range = np.linspace(sub_spec.spectral_axis.min(), sub_spec.spectral_axis.max(), 1000)
            plot_fit = g_fit(plot_wave_range)

            if r2 > 0.75:
                true_l.append(l)
                fit_l.append(g_fit[0].mean.value)

            ax = fig.add_subplot(gs_pre[i // ncols, i % ncols])
            ax.plot(sub_wave, sub_flux, color="black")
            ax.axvline(l, color="gray", ls="--")
            ax.plot(plot_wave_range, plot_fit)
            ax.set_title(f"{l:.1f} $\\AA$, R$^2$={r2:.2f}", fontsize=8)
            i += 1

        # ============= Section 3: calibration fit + residuals =============
        coeffs = np.polyfit(fit_l, true_l, degree)
        p = np.poly1d(coeffs)
        x = np.linspace(min(fit_l), max(fit_l), 100)
        y = p(x)
        residuals = np.array(true_l) - p(np.array(fit_l))

        gs_cal = outer[2].subgridspec(2, 1, height_ratios=[3, 1], hspace=0)
        ax1 = fig.add_subplot(gs_cal[0])
        ax2 = fig.add_subplot(gs_cal[1], sharex=ax1)

        ax1.plot(x, y, color="black", lw=1)
        ax1.scatter(fit_l, true_l, zorder=10)
        ax1.set_ylabel("True λ")
        ax1.set_title(f"{coeffs[0]:.1e}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}")

        ax2.axhline(0, color="black", lw=1, ls="--")
        ax2.scatter(fit_l, residuals, zorder=10, color="tab:red")
        ax2.set_xlabel("Centroid")
        ax2.set_ylabel("Resid.")

        # ============= Section 4: post-calibration fit grid =============
        corrected_wave = p(sky_spec.spectral_axis.value) * sky_spec.spectral_axis.unit
        corrected_spec = Spectrum(spectral_axis=corrected_wave, flux=sky_spec.flux)

        gs_post = outer[3].subgridspec(nrows, ncols, hspace=0.6, wspace=0.3)
        i = 0
        for l in sky_lines:
            sub_region = (corrected_spec.spectral_axis > l*u.AA - window*u.AA) & \
                         (corrected_spec.spectral_axis < l*u.AA + window*u.AA)
            sub_wave = corrected_spec.spectral_axis[sub_region]
            sub_flux = corrected_spec.flux[sub_region]
            sub_spec = Spectrum(flux=sub_flux, spectral_axis=sub_wave)

            try:
                g_init = models.Gaussian1D(amplitude=sub_flux.max(), mean=l, stddev=4*u.AA) \
                        + models.Const1D(amplitude=sub_flux.min())
                g_fit = fit_lines(sub_spec, g_init, get_fit_info=True)
            except ValueError:
                continue

            y_fit = g_fit(sub_spec.spectral_axis)

            if len(sub_flux) < 2:
                continue  # not enough data points to evaluate the fit meaningfully
            r2 = r2_score(sub_flux.value, y_fit.value)

            plot_wave_range = np.linspace(sub_spec.spectral_axis.min(), sub_spec.spectral_axis.max(), 1000)
            plot_fit = g_fit(plot_wave_range)

            ax = fig.add_subplot(gs_post[i // ncols, i % ncols])
            ax.plot(sub_wave, sub_flux, color="black")
            ax.axvline(l, color="gray", ls="--")
            ax.plot(plot_wave_range, plot_fit)
            ax.set_title(f"{l:.1f} $\\AA$, R$^2$={r2:.2f}", fontsize=8)
            i += 1

        fig.suptitle(os.path.basename(filepath), fontsize=14, y=1.0)

        # ---- Save this ENTIRE figure as ONE page ----
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # write wave_cal column to the FITS table ----
        with fits.open(filepath, mode="update") as hdul:
            table_hdu = hdul[1]
            orig_data = table_hdu.data
            orig_cols = orig_data.columns

            wave_cal = p(orig_data["wave"])

            # remove wave_cal column if it already exists (re-running the script)
            try:
                orig_cols.del_col("wave_cal")
            except KeyError:
                pass

            new_col = fits.Column(name="wave_cal", format="D", array=wave_cal)

            new_hdu = fits.BinTableHDU.from_columns(orig_cols + new_col,
                                                    header=table_hdu.header,
                                                    name=table_hdu.name)
            hdul[1] = new_hdu
            hdul.flush()
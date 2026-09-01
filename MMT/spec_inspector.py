import glob
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from astropy.table import Table
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.coordinates import SkyCoord
import astropy.units as u

# from specutils.utils.wcs_utils import vac_to_air, air_to_vac

# ------------
# Load in all coadded spectra
# ------------
DIR1 = "/home/holdennix/Research/G165/MMT/pypeit/RDXDIR_faint/mmt_binospec_A/Coadded"
DIR2 = "/home/holdennix/Research/G165/MMT/pypeit/RDXDIR_short/mmt_binospec_A/Coadded"
PATTERN = "*.fits"

# Gather file lists from both directories
files1 = sorted(glob.glob(os.path.join(DIR1, PATTERN)))
files2 = sorted(glob.glob(os.path.join(DIR2, PATTERN)))

all_files = files1 + files2

print(f"Found {len(files1)} files in RDXDIR_faint")
print(f"Found {len(files2)} files in RDXDIR_short")
print(f"Total: {len(all_files)} files")

if len(all_files) == 0:
    raise FileNotFoundError(
        "No FITS files found in either directory. Check DIR1, DIR2, and PATTERN."
    )

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


def get_coords(f):
    '''
    Converts the JHMS+DMS coordinate format in the filename to
    an Astropy SkyCoord object.
    '''

    relevant = f.split("/")[-1]
    ra = relevant[1:10]
    ra_fmt = f"{ra[0:2]}h{ra[2:4]}m{ra[4:]}s"

    dec = relevant[10:19]
    dec_fmt = f"{dec[0:3]}d{dec[3:5]}m{dec[5:]}s"

    coords = SkyCoord(ra_fmt, dec_fmt, frame="icrs")

    return coords


z_cat_file = "MMT/data/mmt_spec_pipe_z_cat.txt"
z_cat = np.loadtxt(z_cat_file, usecols=[1,2,3])
all_ra = [row[0] for row in z_cat]
all_dec = [row[1] for row in z_cat]
all_z = [row[2] for row in z_cat]

def get_z_from_cat(f):
    '''
    The redshift catalog provided does not have the same names as the objects
    from PypeIt. However, it does provide the RA, Dec coordinates which I
    use to match galaxies.

    It uses the seperation() function from Astropy to find an object within
    2 arcseconds. If there it can't find one, it returns 0 as a null value.
    '''

    closest_i = 0
    coords = get_coords(f)

    candidates = SkyCoord(ra=all_ra*u.deg, dec=all_dec*u.deg)

    sep = coords.separation(candidates)
    idx = np.argmin(sep)

    # print(idx, sep[idx].arcsec)
    # print(f"z={all_z[idx]}")
    if sep[idx].arcsec > 2:
        print("Seperation greater than 2 arcseconds.")
        return 0, coords
    
    return all_z[idx], coords


# Collection of common emission/absorption lines
lwave = [912,1026,1215.67,1240,1260,1296.3,1323.9,1302,1304,1335,\
         1343.354,1394,1403,1417.237,1427.85,1501.76,1527,1548,1550,1608,\
         1640,1671,1855,1863,1909,2326,2344,2374,2424,2587,\
         2600,2796,2799,3346,3426,3727,3798,3835,3889,3933,3968,\
         3970,4102,4304,4340,4861,4959,5007,5167,5173,5184,\
         5876,5889,5896,6548,6563,6583,6716,6730,6875,7040,\
         7680,8190,8520]
lname = ['Lylim','Lyb','Lya','NV','SiII','CIII/SiIII','CII/NIII','SiII/OI',' ','CII',\
         'OIV','SiIV',' ','SiIII','CIII','SV','SiII','CIV',' ','FeII', \
         'HeII','AlII','AlIII','.','CIII','CII','FeII','FeII','NeIV','FeII',\
         '.','MgII',' ','NeV','NeV','[OII]','Hth','Heta','Hz','K','H,Hep',\
         ' ','Hd','Gb','Hg','Hb','[OIII]','[OIII]',' ','MgI',' ',\
         'HeI,NaD','.','.','NII','Ha','.','SII','.','Bb','TiO',\
         'KI','Na','Cs']

# Want to output all figures into scrollable PDF file
with PdfPages("mmt_spec_inspect.pdf") as pdf:
    for i, f in enumerate(all_files):
        wave, flux, err, sky, smooth_flux, smoooth_err, smoothed_sky = load_spectrum(f, smooth=True)
        z, coords = get_z_from_cat(f)

        # Don't care too much yet about bad redshift fits as
        # we have to redo it on PypeIt spectra
        if z==-1 or z==0:
            print("Bad redshift... skipping")
            continue

        # Lots of plot setup to get the outline wanted
        fig = plt.figure(figsize=(15, 11))
        outer_gs = GridSpec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.3)  # top block, zoom row 1, zoom row 2

        top_gs = outer_gs[0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
        ax0 = fig.add_subplot(top_gs[0])
        ax1 = fig.add_subplot(top_gs[1], sharex=ax0)

        zoom_gs = outer_gs[1].subgridspec(1, 3, wspace=0.3)
        ax_zoom = [fig.add_subplot(zoom_gs[i]) for i in range(3)]

        zoom_gs2 = outer_gs[2].subgridspec(1, 3, wspace=0.3)
        ax_zoom2 = [fig.add_subplot(zoom_gs2[i]) for i in range(3)]

 
        if err is not None:
            ax0.fill_between(wave, flux - err, flux + err, color="steelblue", alpha=0.2, label="1$\\sigma$ error")

        ax0.plot(wave, smooth_flux, label="Smoothed Flux ($\\sigma=3$)", color="firebrick", lw=1)
        ax0.set_ylabel("Flux [Counts]")
        ax0.set_title(f"{coords.ra.deg:.6f}, {coords.dec.deg:.6f} | z={z}")
        ax0.legend(loc="upper left")
        ax0.set_ylim(2 * smooth_flux.min(), 2 * smooth_flux.max())

        yann = 1.5 * smooth_flux.max()
        xoffs = 5
        zed = z
        for i in range(len(lwave)):
            tmpwl = lwave[i] * (1 + zed)
            ax0.axvline(x=tmpwl, color='green', lw=1, ls=':')
            ax0.annotate(lname[i], (tmpwl, yann), xytext=(tmpwl - xoffs, yann), rotation=90, clip_on=True)

        ax1.plot(wave, smoothed_sky, color="black", lw=1, label="Smoothed Sky Flux ($\\sigma=3$)")
        ax1.set_xlabel("Wavelength [Angstrom]")
        ax1.legend(loc="upper left")
        ax1.set_xlim(wave.min(), wave.max())

        # Plot emission/absorption lines
        zoom_lines = [35, 45, 46]        # indices of lwave/lname
        zoom_halfwidth = 30              # angstrom width of zoomed windows
        for ax_z, li in zip(ax_zoom, zoom_lines):
            center = lwave[li] * (1 + zed)

            mask = (wave > center - zoom_halfwidth) & (wave < center + zoom_halfwidth)

            ax_z.plot(wave[mask], smooth_flux[mask], color="firebrick", lw=1)
            if err is not None:
                ax_z.fill_between(wave[mask], (flux - err)[mask], (flux + err)[mask],
                                color="steelblue", alpha=0.2)
            ax_z.axvline(x=center, color='green', lw=1, ls=':')
            ax_z.set_title(lname[li], fontsize=10)
            ax_z.set_xlabel("Wavelength [Å]")

        # Need to do this again for bottom row
        zoom_lines = [47, 54, 55]  
        for ax_z, li in zip(ax_zoom2, zoom_lines):
            center = lwave[li] * (1 + zed)

            mask = (wave > center - zoom_halfwidth) & (wave < center + zoom_halfwidth)

            ax_z.plot(wave[mask], smooth_flux[mask], color="firebrick", lw=1)
            if err is not None:
                ax_z.fill_between(wave[mask], (flux - err)[mask], (flux + err)[mask],
                                color="steelblue", alpha=0.2)
            ax_z.axvline(x=center, color='green', lw=1, ls=':')
            ax_z.set_title(lname[li], fontsize=10)
            ax_z.set_xlabel("Wavelength [Å]")

        ax_zoom[0].set_ylabel("Flux [Counts]")
        ax_zoom2[0].set_ylabel("Flux [Counts]")

        pdf.savefig(fig)
        plt.close(fig)


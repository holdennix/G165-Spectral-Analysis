""""
This is a custom file to coadd sky spectra from the spec1D files produced by PypeIt. For some reason PypeIt does not do this
by default. If you point the folders to where your spec1d files live, it should work.

Issues:
 - There are 4 files that it fails on (can't find coadded files): r13, r202, r206, SDSSJ112738.41+421717.7
"""
import numpy as np
import glob
import os
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from pypeit import specobjs

import warnings

# Collect spec1d files
coadd_dir = "/home/holdennix/Research/G165/MMT/pypeit/all_coadded"

PATTERN = "spec1d*.fits"

spec1d_files = sorted(glob.glob(os.path.join("/home/holdennix/Research/G165/MMT/pypeit/all_spec1d", PATTERN)))
print(f"{len(spec1d_files)} files")

if len(spec1d_files) == 0:
    raise FileNotFoundError(
        "No FITS files found."
    )


sky_stack_dic = {}
# I found it easier to loop through the files instead of the objects, so this is the hierarchy I went with.
# To keep track of the objects, I just use a dictionary.
for f in spec1d_files:
    # Load a list of spectrum objects
    sobjs = specobjs.SpecObjs.from_fitsfile(f)

    for sobj in sobjs:
        sobj = sobjs[sobjs.NAME == sobj.NAME]
        sobj_name = sobj.MASKDEF_OBJNAME[0]

        # PypeIt ignores serendipidous objects when running its coadd function, so I do too
        if sobj_name == "SERENDIP": continue

        try:
            coadd = fits.open(f"{coadd_dir}/{sobj_name}.fits")[1].data
        except FileNotFoundError:
            print(f"Can not find {sobj_name}")
            continue

        # we use the wave array in coadd and interpolate to this so that we can plot the sky
        # over the real spectrum later
        wave_ref = coadd["wave"]

        wave_i = sobj.OPT_WAVE
        sky_i  = sobj.OPT_COUNTS_SKY
        gpm_i  = sobj.OPT_MASK

        # interpolate onto the common grid; mask edges/bad pixels as NaN
        sky_interp = np.interp(wave_ref, wave_i[gpm_i], sky_i[gpm_i],
                                left=np.nan, right=np.nan)

        # initialize the object in the dictionary if it does not exist already
        if sobj_name not in sky_stack_dic:
            sky_stack_dic[sobj_name] = []

        sky_stack_dic[sobj_name].append(sky_interp)

# Now I take the dictionary and loop over each object. There is prob a more efficient way to do this, but
# this seems to work and is quick for the number of objects I have.
print(f"Saving sky spectra to coadded fits files.")

for obj_name in sky_stack_dic.keys():
    # I am just taking the mean to coadd the sky spectra. Maybe take the median instead?
    sky_stack = np.array(sky_stack_dic[obj_name])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        sky_coadd = np.nanmean(sky_stack, axis=0)

    # Save the coadded sky spectrum as a new column in the coadded fits tables.
    coadd_file = f"{coadd_dir}/{obj_name}.fits"
    with fits.open(coadd_file, mode="update") as hdul:
        table_hdu = hdul[1]          # the OneSpec table extension
        orig_data = table_hdu.data
        orig_cols = orig_data.columns

        # sanity check: sky array must match the length of the existing table
        assert len(sky_coadd) == len(orig_data), \
            f"length mismatch: sky={len(sky_coadd)}, table={len(orig_data)},\n{obj_name}, {coadd_file}"

        # remove sky_flux column if it already exists
        try:
            orig_cols.del_col("sky_flux")
        except KeyError:
            pass

        new_col = fits.Column(name="sky_flux", format="D", array=sky_coadd)

        new_hdu = fits.BinTableHDU.from_columns(orig_cols + new_col,
                                                header=table_hdu.header,
                                                name=table_hdu.name)
        hdul[1] = new_hdu
        hdul.flush()


print("Done")
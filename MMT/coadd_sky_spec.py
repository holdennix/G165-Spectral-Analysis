""""
This is a custom file to coadd sky spectra from the spec1D files produced by PypeIt. For some reason PypeIt does not do this
by default. If you point the folders to where your spec1d files live, it should work.

Issues:
 - There are 3 files that it fails on (can't find a match from spec1d RA, Dec to Coadded filename RA, Dec)
"""



import numpy as np
import glob
import os
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from pypeit import specobjs

import warnings


# ---------------
# Collect all spec1d files
# This script will basically run twice, once for each set of spec1d files.
# ---------------
DIR1 = "/home/holdennix/Research/G165/MMT/pypeit/RDXDIR_faint/mmt_binospec_A/Science"
DIR2 = "/home/holdennix/Research/G165/MMT/pypeit/RDXDIR_short/mmt_binospec_A/Science"
PATTERN = "spec1d*.fits"

faint_files = sorted(glob.glob(os.path.join(DIR1, PATTERN)))
short_files = sorted(glob.glob(os.path.join(DIR2, PATTERN)))

spec1d_files = faint_files + short_files

print(f"Found {len(faint_files)} files in RDXDIR_faint")
print(f"Found {len(short_files)} files in RDXDIR_short")
print(f"Total: {len(spec1d_files)} files")

if len(spec1d_files) == 0:
    raise FileNotFoundError(
        "No FITS files found in either directory. Check RDXDIR_faint, RDXDIR_short, and PATTERN."
    )


for file_set in [faint_files, short_files]:
    if "faint" in file_set[0]: files_type = "faint"
    elif "short" in file_set[0]: files_type = "short"

    print(f"Coadding '{files_type}' sky spectra.")

    # ---------------
    # I found it easier to loop through the files instead of the objects, so this is the hierarchy I went with.
    # To keep track of the objects, I just use a dictionary. This all runs fairly fast for about 160 objects.
    # ---------------
    
    sky_stack_dic = {}
    for f in file_set:
        # Load a list of spectrum objects
        sobjs = specobjs.SpecObjs.from_fitsfile(f)

        for sobj in sobjs:
            sobj = sobjs[sobjs.NAME == sobj.NAME]

            # PypeIt ignores serendipidous objects when running its coadd function, so I do too
            if sobj.MASKDEF_OBJNAME == "SERENDIP": continue

            # Find RA, Dec in JHMS+DMS to find matching coadd file
            ra, dec = sobj.RA, sobj.DEC   # in decimal degrees
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            name = 'J' + coord.to_string('hmsdms', sep='', precision=2)[0]
            name = name.replace(' ', '')  # to_string puts a space between RA and Dec

            pattern = f"/home/holdennix/Research/G165/MMT/pypeit/RDXDIR_{files_type}/mmt_binospec_A/Coadded/{name}*.fits"
            matches = glob.glob(pattern)

            if not matches:
                print(f"No coadd file found matching pattern: {pattern}")
                continue
            elif len(matches) > 1:
                print(f"Warning: multiple matches found, using first: {matches}")

            coadd = fits.open(matches[0])[1].data

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
            if name not in sky_stack_dic:
                sky_stack_dic[name] = []

            sky_stack_dic[name].append(sky_interp)


    # ---------------
    # Now I take the dictionary and loop over each object. There is prob a more efficient way to do this, but
    # this seems to work and is quick for the number of objects I have.
    # ---------------
    print(f"Saving {files_type} sky spectra to coadded fits files.")
    for obj_name in sky_stack_dic.keys():
        # ---------------
        # I am just taking the mean to coadd the sky spectra. Maybe take the median instead?
        # ---------------
        sky_stack = np.array(sky_stack_dic[obj_name])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            sky_coadd = np.nanmean(sky_stack, axis=0)

        pattern = f"/home/holdennix/Research/G165/MMT/pypeit/RDXDIR_{files_type}/mmt_binospec_A/Coadded/{obj_name}*.fits"
        matches = glob.glob(pattern)

        if len(matches) > 1: 
            print(False)
            break

        coadd_file = matches[0]
        # ---------------
        # Save the coadded sky spectrum as a new column in the coadded fits tables.
        # ---------------
        with fits.open(coadd_file, mode="update") as hdul:
            table_hdu = hdul[1]          # the OneSpec table extension
            orig_data = table_hdu.data
            orig_cols = orig_data.columns

            # sanity check: sky array must match the length of the existing table
            assert len(sky_coadd) == len(orig_data), \
                f"length mismatch: sky={len(sky_coadd)}, table={len(orig_data)},\n{obj_name}, {coadd_file}"

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

        # print("Added 'sky_flux' column to", coadd_file)


print("Done")
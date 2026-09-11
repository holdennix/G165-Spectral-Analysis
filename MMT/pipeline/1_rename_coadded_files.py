'''
This script simply renames the coadded files to their object name.
This just makes life a little easier later.

Replacing the coadd_dir with your directory to all the coadded files should
make this work easily.
'''


from astropy.table import Table
from astropy.io import fits
import numpy as np
import os


coadd_dir = "/home/holdennix/Research/G165/MMT/pypeit/all_coadded"

report_table = Table.read(f"{coadd_dir}/collate_report.dat", format="ascii.ipac")
keep_cols = ["filename", "maskdef_objname", "maskdef_id", "pypeit_name",
             "objra", "objdec", "s2n", "wave_rms", "spec1d_filename", "exptime"]
report_table.keep_columns(keep_cols)
rt_groups = report_table.group_by("filename")

# choose "objname" for pypeit objname as file convention
# choose "default" for pypeit default J[RA][Dec]... convention
filename_type = "objname"


all_objnames = []
for group in rt_groups.groups:
    coadd_filename = group["filename"].data[0]
    coadd_objname = group["maskdef_objname"].data[0]

    ra = group["objra"].data[0]
    dec = group["objdec"].data[0]
    s2n = group["s2n"].data[0]
    wave_rms = group["wave_rms"].data[0]

    # changing from og_filename to filename_type specified above
    if filename_type == "default":
        og_filename, new_filename = f"{coadd_objname}.fits", coadd_filename
    elif filename_type == "objname":
        og_filename, new_filename = coadd_filename, f"{coadd_objname}.fits"

    if coadd_objname == "SERENDIP":
        try: os.remove(f"{coadd_dir}/{og_filename}")
        except FileNotFoundError: pass
        continue

    try:
        os.rename(f"{coadd_dir}/{og_filename}", f"{coadd_dir}/{new_filename}")
    except FileNotFoundError:
        print(f"Can not find file: {coadd_dir}/{og_filename}")

    # Used in redshift error estimation later
    with fits.open(f"{coadd_dir}/{new_filename}", mode="update") as hdul:
        hdul[0].header["wave_rms"] = wave_rms


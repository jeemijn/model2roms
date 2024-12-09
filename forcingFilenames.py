from datetime import datetime

from netCDF4 import MFDataset, date2index, num2date


# Main function called from model2roms
def get_filename(confM2R, year, month, day, defaultvar):
    """Function returning the filename. 
    All ocean_indata_types should appear here.
    The defaultvar is used in get_time() and in convert_MODEL2ROMS() for grid initialization."""
    if confM2R.ocean_indata_type == "SODA3":
        if defaultvar is None:
            defaultvar = "salinity"
        filenamein = getSODA3filename(confM2R, year, month, day, defaultvar)
    if confM2R.ocean_indata_type == "SODA3_5DAY":
        if defaultvar is None:
            defaultvar = "salinity"
        filenamein = getSODA3_5DAYfilename(confM2R, year, month, day, defaultvar)
    if confM2R.ocean_indata_type == "NORESM":
        if defaultvar is None:
            defaultvar = "grid"
        filenamein = getNORESMfilename(confM2R, year, month, defaultvar)
    if confM2R.ocean_indata_type == "GLORYS":
        if defaultvar is None:
            defaultvar = "So"
        filenamein = get_GLORYS_filename(confM2R, year, month, defaultvar)
    return filenamein


# private functions called from within module
def get_GLORYS_filename(confM2R, year, month, varname):
    if confM2R.use_zarr:
        return f"{confM2R.ocean_forcing_path}{varname.lower()}"
    return "{}{}/CMEMS_{}_monthly_MERCATOR_{}-{}.nc".format(
        confM2R.ocean_forcing_path,
        varname.lower(),
        varname.capitalize(),
        year,
        str(month).zfill(2),
    )


def getNORESMfilename(confM2R, year, month, myvar):
    if myvar == "grid":
        filename = confM2R.ocean_forcing_path + "grid_gx1v6.nc"
    else:
        if myvar in ["iage", "uvel", "vvel", "aice", "hi", "hs"]:
            filename = (
                confM2R.ocean_forcing_path
                + "ICE/NRCP45AERCN_f19_g16_CLE_01.cice.h."
                + str(year)
                + ".nc"
            )

        elif myvar in ["dissic", "talk", "po4", "no3", "si", "o2"]:
            filename = (
                confM2R.ocean_forcing_path
                + "BCG_NRCP85BPRPEX_01.micom.hbgcmlvl.2006-2050.nc"
            )

        elif myvar in ["templvl", "salnlvl", "sealv", "uvellvl", "vvellvl"]:
            if myvar in ["salnlvl", "templvl"]:
                filename = (
                    confM2R.ocean_forcing_path + "TS_NRCP85BPRPEX_01.micom.2006-2100.nc"
                )
            else:
                filename = (
                    confM2R.ocean_forcing_path
                    + "VEL_NRCP85BPRPEX_01.micom.hmlvl.2006-2100.nc"
                )

        elif myvar in ["TAUX", "TAUY", "U10"]:    # atmospheric variables
            filename = (
                confM2R.atmospheric_forcing_path
                + f"NRCP45AERCN_f19_g16_CLE_01.cam2.h5.{year}-{month:02d}-{day:02d}-00000.nc"
            )
    return filename


def getSODA3filename(confM2R, year, month, day, myvar):
    if myvar in ["cn", "hi", "hs"]:
        return "{}soda{}_mn_ice_reg_{}.nc".format(
            confM2R.ocean_forcing_path, confM2R.soda_version, str(year)
        )
    else:
        return "{}soda{}_mn_ocean_reg_{}.nc".format(
            confM2R.ocean_forcing_path, confM2R.soda_version, str(year)
        )


def getSODA3_5DAYfilename(confM2R, year, month, day, myvar):
    if len(confM2R.time_object) == 0:
        mcdf = MFDataset(confM2R.ocean_forcing_path + "*.nc")
        confM2R.time_object = mcdf.variables["time"]

    index = date2index(
        datetime(year, month, day, 0, 0),
        confM2R.time_object,
        calendar=confM2R.time_object.calendar,
        select="nearest",
    )
    seldate = num2date(
        confM2R.time_object[index],
        units=confM2R.time_object.units,
        calendar=confM2R.time_object.calendar,
    )

    if myvar in ["cn", "hi", "hs"]:
        return "{}soda{}_5dy_ocean_ice_{:04}_{:02}_{:02}.nc".format(
            confM2R.ocean_forcing_path,
            confM2R.soda_version,
            seldate.year,
            seldate.month,
            seldate.day,
        )
    else:
        return "{}soda{}_5dy_ocean_reg_{:04}_{:02}_{:02}.nc".format(
            confM2R.ocean_forcing_path,
            confM2R.soda_version,
            seldate.year,
            seldate.month,
            seldate.day,
        )

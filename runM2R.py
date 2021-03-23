import time
from datetime import datetime, timedelta
import configM2R
import model2roms
import IOstation
import clim2bry
import decimateGrid
import atmosForcing

__author__ = 'Trond Kristiansen'
__email__ = 'trond.kristiansen@niva.no'
__created__ = datetime(2009, 1, 30)
__modified__ = datetime(2019, 2, 8)
__version__ = "1.5"
__status__ = "Development"

"""
    Main method for running model2roms
    Start: python runM2R.py 
"""

def run():
    print("Started model2roms")
    confM2R = configM2R.Model2romsConfig()

    if confM2R.create_atmos_forcing or confM2R.create_ocean_forcing:

        if confM2R.create_ocean_forcing:
            model2roms.convertMODEL2ROMS(confM2R)

            clim2bry.writebry(confM2R)

      #  if confM2R.createAtmosForcing:
      #      atmosForcing.createAtmosFileUV(confM2R)

    if confM2R.decimategridfile:
        decimateGrid.createGrid(confM2R.grdROMS, "/Users/trondkr/Projects/KINO/GRID/kino_1600m_18072015.nc",
                                "/Users/trondkr/Projects/KINO/GRID/kino_1600m_18072015v2.nc", 2)

    if confM2R.extract_stations:
        print("Running in station mode and extracting pre-defined station locations")
        IOstation.getStationData(confM2R)

    print('Finished ' + time.ctime(time.time()))

run()

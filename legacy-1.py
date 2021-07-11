import fastf1
import fastf1.plotting
import numpy as np
import matplotlib.pyplot as plt

fastf1.plotting.setup_mpl()
# fastf1.Cache.enable_cache("path/to/cache")

session = fastf1.get_session(2020, 'Italy', 'R')
laps = session.load_laps(with_telemetry=True)

DRIVER = 'VER'  # which driver; need to specify number and abbreviation
DRIVER_NUMBER = '33'
LAP_N = 10  # which lap number to plot

drv_laps = laps.pick_driver(DRIVER)
drv_lap = drv_laps[(drv_laps['LapNumber'] == LAP_N)]  # select the lap

# create a matplotlib figure
fig = plt.figure()
ax = fig.add_subplot()

# ############### new
df_new = drv_lap.get_car_data().add_driver_ahead()
ax.plot(df_new['Time'], df_new['DistanceToDriverAhead'], label='new')

# ############### legacy
df_legacy = fastf1.legacy.inject_driver_ahead(session)[DRIVER_NUMBER].slice_by_lap(drv_lap)
ax.plot(df_legacy['Time'], df_legacy['DistanceToDriverAhead'], label='legacy')

plt.legend()
plt.show()
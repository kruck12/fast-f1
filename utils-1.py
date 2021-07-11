import fastf1 as ff1
from fastf1 import plotting
from fastf1 import utils
from matplotlib import pyplot as plt

plotting.setup_mpl()

quali = ff1.get_session(2021, 'Emilia Romagna', 'Q')
laps = quali.load_laps(with_telemetry=True)
lec = laps.pick_driver('LEC').pick_fastest()
ham = laps.pick_driver('HAM').pick_fastest()

delta_time, ref_tel, compare_tel = utils.delta_time(ham, lec)
# ham is reference, lec is compared

fig, ax = plt.subplots()
# use telemetry returned by .delta_time for best accuracy,
# this ensure the same applied interpolation and resampling
ax.plot(ref_tel['Distance'], ref_tel['Speed'],
        color=plotting.TEAM_COLORS[ham['Team']])
ax.plot(compare_tel['Distance'], compare_tel['Speed'],
        color=plotting.TEAM_COLORS[lec['Team']])

twin = ax.twinx()
twin.plot(ref_tel['Distance'], delta_time, '--', color='white')
twin.set_ylabel("<-- Lec ahead | Ham ahead -->")
plt.show()
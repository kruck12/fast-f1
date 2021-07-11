from matplotlib import pyplot as plt
import fastf1 as ff1
from fastf1 import plotting

plotting.setup_mpl()

monza_quali = ff1.get_session(2019, 'Monza', 'Q')

laps = monza_quali.load_laps(with_telemetry=True)
fast_leclerc = laps.pick_driver('LEC').pick_fastest()
lec_car_data = fast_leclerc.get_car_data()
t = lec_car_data['Time']
vCar = lec_car_data['Speed']

# The rest is just plotting
fig, ax = plt.subplots()
ax.plot(t, vCar, label='Fast')
ax.set_xlabel('Time')
ax.set_ylabel('Speed [Km/h]')
ax.set_title('Leclerc is')
ax.legend()
plt.show()
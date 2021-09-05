"""
:mod:`fastf1.legacy` - Provides access to legacy functionality
==============================================================

This module contains the legacy implementation for calculating distance to driver ahead.

:func:`inject_driver_ahead` adds 'DriverAhead' and 'DistanceToDriverAhead' to the position data for all laps of all
drivers. This functionality has been replaced with :func:`fastf1.core.Telemetry.add_driver_ahead`.
The new implementation provides smoother and more accurate results. Additionally, it can be applied to arbitrary slices
of data. But it suffers from integration error when used over multiple laps. The legacy implementation has no
integration error issues.

It is recommended to use the new version. If necessary, it should be applied lap
by lap and the lap data should be concatenated afterwards. Still, the old version can be used if so desired.

The following is an example comparison plot of the legacy version and the new version. It also shows how the two
versions can be used.

.. plot::
    :include-source:

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
"""

import numpy as np
import pandas as pd
import scipy.spatial
import logging
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message="Using slow pure-python SequenceMatcher")
    # suppress that warning, it's confusing at best here, we don't need fast sequence matching
    # and the installation (on windows) some effort
    from fuzzywuzzy import fuzz

from fastf1 import ergast, core


REFERENCE_LAP_RESOLUTION = 0.667
"""A distance in meters which indicates the resolution of the reference
lap. This reference is used to project car positions and calculate
things like distance between cars.
"""


def _get_reference_lap(session):
    """Find a reference lap for creating the track map in `_make_trajectory`."""
    times = session.laps['LapTime'].copy()
    times = times.sort_values()

    for i in range(len(session.laps)):
        lap = session.laps.loc[times.index[i]]
        car_data = lap.get_car_data()  # TODO interpolate_edges ?
        if np.all(car_data['Speed'] > 0):  # check for valid telemetry
            break
    else:
        return None

    return lap


def _make_trajectory(session, ref_lap):
    """Create telemetry Distance
    """
    telemetry = ref_lap.telemetry

    if telemetry.size != 0:
        x = telemetry['X'].values
        y = telemetry['Y'].values
        z = telemetry['Z'].values
        s = telemetry['Distance'].values

        # Assuming constant speed in the last tenth
        dt0_ = (ref_lap['LapTime'] - telemetry['Time'].iloc[-1]).total_seconds()
        ds0_ = (telemetry['Speed'].iloc[-1] / 3.6) * dt0_
        total_s = s[-1] + ds0_

        # To prolong start and finish and have a correct linear interpolation
        full_s = np.concatenate([s - total_s, s, s + total_s])
        full_x = np.concatenate([x, x, x])
        full_y = np.concatenate([y, y, y])
        full_z = np.concatenate([z, z, z])

        reference_s = np.arange(0, total_s, REFERENCE_LAP_RESOLUTION)

        reference_x = np.interp(reference_s, full_s, full_x)
        reference_y = np.interp(reference_s, full_s, full_y)
        reference_z = np.interp(reference_s, full_s, full_z)

        ssize = len(reference_s)

        """Build track map and project driver position to one trajectory
        """

        def fix_suzuka(projection_index, _s):
            """Yes, suzuka is bad
            """

            # For tracks like suzuka (therefore only suzuka) we have
            # a beautiful crossing point. So, FOR F**K SAKE, sometimes
            # shortest distance may fall below the bridge or viceversa
            # gotta do some monotony sort of check. Not the cleanest
            # solution.
            def moving_average(a, n=3):
                ret = np.cumsum(a, dtype=float)
                ret[n:] = ret[n:] - ret[:-n]
                ma = ret[n - 1:] / n

                return np.concatenate([ma[0:n // 2], ma, ma[-n // 2:-1]])

            ma_projection = moving_average(_s[projection_index], n=3)
            spikes = np.absolute(_s[projection_index] - ma_projection)
            # 1000 and 3000, very suzuka specific. Damn magic numbers
            sel_bridge = np.logical_and(spikes > 1000, spikes < 3000)
            unexpected = np.where(sel_bridge)[0]
            max_length = _s[-1]

            for p in unexpected:
                # Just assuming linearity for this 2 or 3 samples
                last_value = _s[projection_index[p - 1]]
                last_step = last_value - _s[projection_index[p - 2]]

                if (last_value + last_step) > max_length:
                    # Over the finish line
                    corrected_distance = -max_length + last_step + last_value
                else:
                    corrected_distance = last_value + last_step

                corrected_index = np.argmin(np.abs(_s - corrected_distance))
                projection_index[p] = corrected_index

            return projection_index

        track = np.empty((ssize, 3))
        track[:, 0] = reference_x
        track[:, 1] = reference_y
        track[:, 2] = reference_z

        track_tree = scipy.spatial.cKDTree(track)
        drivers_list = np.array(list(session.drivers))
        stream_length = len(session.pos_data[drivers_list[0]])
        dmap = np.empty((stream_length, len(drivers_list)), dtype=int)

        fast_query = {'n_jobs': 2, 'distance_upper_bound': 500}
        # fast_query < Increases speed
        for index, drv in enumerate(drivers_list):
            if drv not in session.pos_data.keys():
                logging.warning(f"Driver {drv: >2}: No position data. (_make_trajectory)")
                continue
            trajectory = session.pos_data[drv][['X', 'Y', 'Z']].values
            projection_index = track_tree.query(trajectory, **fast_query)[1]
            # When tree cannot solve super far points means there is some
            # pit shit shutdown. We can replace these index with 0
            projection_index[projection_index == len(reference_s)] = 0
            dmap[:, index] = fix_suzuka(projection_index.copy(), reference_s)

        """Create transform matrix to change distance point of reference
        """
        t_matrix = np.empty((ssize, ssize))
        for index in range(ssize):
            rref = reference_s - reference_s[index]
            rref[rref <= 0] = total_s + rref[rref <= 0]
            t_matrix[index, :] = rref

        """Create mask to remove distance elements when car is on track
        """
        time = session.pos_data[drivers_list[0]]['Time']
        pit_mask = np.zeros((stream_length, len(drivers_list)), dtype=bool)
        for driver_index, driver_number in enumerate(drivers_list):
            laps = session.laps.pick_driver(driver_number)
            in_pit = True
            times = [[], []]
            for lap_index in laps.index:
                lap = laps.loc[lap_index]
                if not pd.isnull(lap['PitInTime']) and not in_pit:
                    times[1].append(lap['PitInTime'])
                    in_pit = True
                if not pd.isnull(lap['PitOutTime']) and in_pit:
                    times[0].append(lap['PitOutTime'])
                    in_pit = False

            if not in_pit:
                # Car crashed, we put a time and 'Status' will take care
                times[1].append(lap['Time'])
            times = np.transpose(np.array(times))
            for inout in times:
                out_of_pit = np.logical_and(time >= inout[0], time < inout[1])
                pit_mask[:, driver_index] |= out_of_pit
            on_track = (session.pos_data[driver_number]['Status'] == 'OnTrack')
            pit_mask[:, driver_index] &= on_track.values

        """Calculate relative distances using transform matrix
        """
        driver_ahead = {}
        stream_axis = np.arange(stream_length)
        for my_di, my_d in enumerate(drivers_list):
            rel_distance = np.empty(np.shape(dmap))

            for his_di, his_d in enumerate(drivers_list):
                my_pos_i = dmap[:, my_di]
                his_pos_i = dmap[:, his_di]
                rel_distance[:, his_di] = t_matrix[my_pos_i, his_pos_i]

            his_in_pit = ~pit_mask.copy()
            his_in_pit[:, my_di] = False
            my_in_pit = ~pit_mask[:, drivers_list == my_d][:, 0]
            rel_distance[his_in_pit] = np.nan

            closest_index = np.nanargmin(rel_distance, axis=1)
            closest_distance = rel_distance[stream_axis, closest_index]
            closest_driver = drivers_list[closest_index].astype(object)
            closest_distance[my_in_pit] = np.nan
            closest_driver[my_in_pit] = None

            data = {'DistanceToDriverAhead': closest_distance,
                    'DriverAhead': closest_driver}
            driver_ahead[my_d] = session.pos_data[my_d].join(pd.DataFrame(data), how='outer')

    else:
        # no data to base calculations on; create empty results
        driver_ahead = dict()
        for drv in session.drivers:
            data = {'DistanceToDriverAhead': (), 'DriverAhead': ()}
            driver_ahead[drv] = session.pos_data[drv].join(pd.DataFrame(data), how='outer')

    return driver_ahead


def inject_driver_ahead(session):
    """Add 'DistanceToDriverAhead' and 'DriverAhead' column to position data of all drivers in session.

    Args:
        session: :class:`fastf1.core.Session`

    Returns:
        A dictionary containing :class:`fastf1.core.Telemetry` for each driver. The telemetry is the same as
        :any:`fastf1.core.Session.pos_data` but with 'DriverAhead' and 'DistanceToDriverAhead' columns added
        to each drivers telemetry.
    """

    ref_lap = _get_reference_lap(session)

    if ref_lap is not None:
        driver_ahead = _make_trajectory(session, ref_lap)

    else:
        raise ValueError("No valid telemetry for calculating distance to driver ahead!")

    return driver_ahead


class LegacyScheduleBackend:
    TESTING_LOOKUP = {'2020': [['2020-02-19', '2020-02-20', '2020-02-21'],
                               ['2020-02-26', '2020-02-27', '2020-02-28']],
                      '2021': [['2021-03-12', '2021-03-13', '2021-03-14']]}

    @classmethod
    def get_session(cls, year, gp, event=None):
        """Create a :class:`Session` or :class:`Weekend` object based on year,
        event name and session name.
        This function will take care of crafting an object
        corresponding to the requested session.
        If no session is specified, the full weekend is returned.

        Examples:

            Get the second free practice of the first race of 2021::

                get_session(2021, 1, 'FP2')

            Get the qualifying of the 2020 Austrian Grand Prix::

                get_session(2020, 'Austria', 'Q')

            Get the second day of pre-season testing of 2021::

                get_session(2021, 'testing', 2)


        Args:
            year (number): Session year
            gp (number or string): Name or weekend number (1: Australia,
                                   ..., 21: Abu Dhabi). If gp is a string,
                                   a fuzzy match will be performed on the
                                   season rounds and the most likely will be
                                   selected.

                                   Some examples that will be correctly
                                   interpreted: 'bahrain', 'australia',
                                   'abudabi', 'monza'.

                                   Pass 'testing' to fetch Barcelona winter
                                   tests.

            event (=None): may be 'FP1', 'FP2', 'FP3', 'Q' or 'R', if not
                           specified you get the full :class:`Weekend`.
                           If gp is 'testing' event is the test day (1 to 6)

        Returns:
            :class:`Weekend` or :class:`Session`

        """
        if type(gp) is str and gp == 'testing':
            pre_season_week, event = cls._get_testing_week_event(year, event)
            weekend = core.Weekend(year, pre_season_week)
            return core.Session(weekend, event)

        if type(gp) is str:
            gp = cls.get_round(year, gp)
        weekend = core.Weekend(year, gp)
        if event == 'R':
            return core.Session(weekend, 'Race')
        if event == 'Q':
            return core.Session(weekend, 'Qualifying')
        if event == 'FP3':
            return core.Session(weekend, 'Practice 3')
        if event == 'FP2':
            return core.Session(weekend, 'Practice 2')
        if event == 'FP1':
            return core.Session(weekend, 'Practice 1')
        return weekend

    @classmethod
    def get_round(cls, year, match):
        """Get event number by year and (partial) event name

        A fuzzy match is performed to find the most likely event for the provided name.

        Args:
            year (int): Year of the event
            match (string): Name of the race or gp (e.g. 'Bahrain')

        Returns:
            The round number. (2019, 'Bahrain') -> 2
        """

        def build_string(d):
            r = len('https://en.wikipedia.org/wiki/')  # TODO what the hell is this
            c, l = d['Circuit'], d['Circuit']['Location']  # noqa: E741 (for now...)
            return (f"{d['url'][r:]} {d['raceName']} {c['circuitId']} "
                    + f"{c['url'][r:]} {c['circuitName']} {l['locality']} "
                    + f"{l['country']}")

        races = ergast.fetch_season(year)
        to_match = [build_string(block) for block in races]
        ratios = np.array([fuzz.partial_ratio(match, ref) for ref in to_match])

        return int(races[np.argmax(ratios)]['round'])

    @classmethod
    def _get_testing_week_event(cls, year, day):
        """Get the correct weekend and event for testing from the
        year and day of the test. (where day is 1, 2, 3, ...)
        """
        if year == 2020:
            try:
                day = int(day)
                week = 1 if day < 4 else 2
            except:  # noqa: E722 TODO: improve
                raise core.InvalidSessionError
            week_day = ((day - 1) % 3) + 1
            pre_season_week = f'Pre-Season Test {week}'
            event = f'Practice {week_day}'
        elif year == 2021 and int(day) in (1, 2, 3):
            pre_season_week = 'Pre-Season Test'
            event = f'Practice {day}'
        else:
            raise core.InvalidSessionError

        return pre_season_week, event

    @classmethod
    def get_session_date(cls, session):
        """Session date formatted as '%Y-%m-%d' (e.g. '2019-03-12')"""
        if session.weekend.is_testing():
            if (year := str(session.weekend.year)) == '2020':
                week_index = int(session.weekend.name[-1]) - 1
                day_index = int(session.name[-1]) - 1
                date = cls.TESTING_LOOKUP[year][week_index][day_index]
            elif year == '2021':
                day_index = int(session.name[-1]) - 1
                date = cls.TESTING_LOOKUP[year][0][day_index]

        elif session.name in ('Qualifying', 'Practice 3'):
            # Assuming that quali was one day before race which is not always correct
            # TODO Should check if also formula1 makes this assumption
            offset_date = pd.to_datetime(session.weekend.date) + pd.DateOffset(-1)
            date = offset_date.strftime('%Y-%m-%d')
        elif session.name in ('Practice 1', 'Practice 2'):
            # Again, assuming that practice 1/2 are the day before quali (except Monaco)
            _ = -3 if session.weekend.name == 'Monaco Grand Prix' else -2
            offset_date = pd.to_datetime(session.weekend.date) + pd.DateOffset(_)
            date = offset_date.strftime('%Y-%m-%d')
        else:  # Race
            date = session.weekend.date

        return date

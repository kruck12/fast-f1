import datetime

import pandas as pd
import pytest

from fastf1 import events
from fastf1 import Cache


Cache.enable_cache('tmp_cache')


def _get_event_schedule():
    return events.get_event_schedule()


def test_pick_weekend_by_number():
    schedule = _get_event_schedule()

    weekend = schedule.pick_weekend_by_number(10)
    assert isinstance(weekend, events.EventSchedule)
    assert len(weekend) == 5
    assert 'BRITISH GRAND PRIX' in str(weekend['Name'].iloc[0])


@pytest.mark.parametrize(
    "date", ['2021-07-18', '7/18/2021', datetime.date(2021, 7, 18),
             pd.Timestamp(year=2021, month=7, day=18)])
def test_pick_weekend_by_date(date):
    schedule = _get_event_schedule()

    weekend = schedule.pick_weekend_by_date(date)
    assert isinstance(weekend, events.EventSchedule)
    assert len(weekend) == 5
    assert 'BRITISH GRAND PRIX' in str(weekend['Name'].iloc[0])


@pytest.mark.xfail
def test_pick_session_types():
    schedule = _get_event_schedule()
    for _type in ['R', 'r', 'Race', 'race']:
        q = schedule.pick_session_type(_type).pick_confirmed()
        assert isinstance(q, events.EventSchedule)
        assert len(q) == 21
        assert q['Session'].unique() == ['Race', ]


@pytest.mark.xfail
def test_pick_session_by_number():
    schedule = _get_event_schedule()

    for session_type in ('R', 'r', 'Race', 'race'):  # TODO allow make full name case insensitive
        session = schedule.pick_session_by_number(session_type, 10)
        assert isinstance(session, events.EventSchedule)
        assert 'BRITISH GRAND PRIX' in str(session['Name'])
        assert session['Session'].iloc[0] == 'Race'


@pytest.mark.xfail
@pytest.mark.parametrize(
    "date", ['2021-08-01', '8/1/2021', datetime.date(2021, 8, 1),
             pd.Timestamp(year=2021, month=8, day=1)])  # TODO not only closest but limit to same day
def test_pick_session_by_date(date):
    schedule = _get_event_schedule()

    session = schedule.pick_session_by_date(date)
    assert isinstance(session, events.EventSchedule)
    assert 'FORMULA 1 MAGYAR' in str(session['Name'])
    assert session['Session'] == 'Race'


def test_pick_sessions_by_date_same_day():
    pass  # TODO tbd


def test_pick_confirmed():
    schedule = _get_event_schedule()

    assert len(schedule['Status'].unique()) > 1
    confirmed = schedule.pick_confirmed()
    assert isinstance(confirmed, events.EventSchedule)
    assert len(confirmed['Status'].unique()) == 1


def test_pick_race_weekends():
    schedule = _get_event_schedule()

    assert any(['TESTING' in str(name) for name in schedule['Name']])
    race_weekends = schedule.pick_race_weekends()
    assert not any(['TESTING' in str(name) for name in race_weekends['Name']])

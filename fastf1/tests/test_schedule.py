import datetime

import pandas as pd
import pytest

from fastf1 import events, core, Cache


# create an EventSchedule class from a local downloaded copy
with open('fastf1/testing/reference_data/calendar.ics',
          encoding='utf-8') as icsfile:
    content = icsfile.read()


class MockResponse:
    def __init__(self, text):
        self.encoding = None
        self.text = text


response = MockResponse(content)
schedule = events._parse_ics_event_schedule(response)


def test_pick_weekend_by_number():
    weekend = schedule.pick_weekend_by_number(10)
    assert isinstance(weekend, events.EventSchedule)
    assert len(weekend) == 5
    assert 'BRITISH GRAND PRIX' in str(weekend['Name'].iloc[0])


@pytest.mark.parametrize(
    "date", ['2021-07-18', '7/18/2021', datetime.date(2021, 7, 18),
             pd.Timestamp(year=2021, month=7, day=18)])
def test_pick_weekend_by_date(date):
    weekend = schedule.pick_weekend_by_date(date)
    assert isinstance(weekend, events.EventSchedule)
    assert len(weekend) == 5
    assert 'BRITISH GRAND PRIX' in str(weekend['Name'].iloc[0])


def test_pick_session_types():
    for _type in ['R', 'r', 'Race', 'race']:
        q = schedule.pick_session_type(_type).pick_confirmed()
        assert isinstance(q, events.EventSchedule)
        assert len(q) == 21
        assert q['Session'].unique() == ['Race', ]

    # ensure every combination for names with multiple words
    for _type in ['SQ', 'sq', 'Sprint Qualifying', 'sPrInT qualiFying']:
        q = schedule.pick_session_type(_type).pick_confirmed()
        assert isinstance(q, events.EventSchedule)
        assert len(q) == 2
        assert q['Session'].unique() == ['Sprint Qualifying', ]


def test_pick_session_by_number():
    for session_type in ('R', 'r', 'Race', 'race'):  # TODO allow make full name case insensitive
        session = schedule.pick_session_by_number(10, session_type)
        assert isinstance(session, events.EventSchedule)
        assert 'BRITISH GRAND PRIX' in str(session['Name'])
        assert session['Session'].iloc[0] == 'Race'


@pytest.mark.parametrize(
    "date", ['2021-08-01', '8/1/2021', datetime.date(2021, 8, 1),
             pd.Timestamp(year=2021, month=8, day=1)])  # TODO not only closest but limit to same day
def test_pick_session_by_date(date):
    session = schedule.pick_session_by_date(date)
    assert isinstance(session, events.Event)
    assert 'FORMULA 1 ROLEX MAGYAR' in str(session['Name'])
    assert session['Session'] == 'Race'


def test_pick_session_by_date_same_day():
    fp3 = schedule.pick_session_by_date('2021-07-31 10:00')
    assert isinstance(fp3, events.Event)
    assert fp3['Session'] == 'Practice 3'

    q = schedule.pick_session_by_date('2021-07-31 14:00')
    assert isinstance(q, events.Event)
    assert q['Session'] == 'Qualifying'


def test_pick_confirmed():
    assert len(schedule['Status'].unique()) > 1
    confirmed = schedule.pick_confirmed()
    assert isinstance(confirmed, events.EventSchedule)
    assert len(confirmed['Status'].unique()) == 1


def test_pick_race_weekends():
    assert any(['TESTING' in str(name) for name in schedule['Name']])
    race_weekends = schedule.pick_race_weekends()
    assert not any(['TESTING' in str(name) for name in race_weekends['Name']])


def test_pick_weekend_by_name():
    weekend = schedule.pick_weekend_by_name('Belgian')
    assert isinstance(weekend, events.EventSchedule)
    assert weekend['Name'].unique() == \
           ['FORMULA 1 ROLEX BELGIAN GRAND PRIX 2021', ]


def test_pick_session_by_name():
    session = schedule.pick_session_by_name('Belgian', 'Q')
    assert isinstance(session, events.Event)
    assert session['Name'] == 'FORMULA 1 ROLEX BELGIAN GRAND PRIX 2021'
    assert session['Session'] == 'Qualifying'


@pytest.mark.f1telapi
def test_to_weekend():
    Cache.enable_cache('test_cache')
    w = schedule.pick_weekend_by_number(11)
    weekend = w.to_weekend()
    assert isinstance(weekend, core.Weekend)
    session = weekend.get_quali()
    assert isinstance(session, core.Session)
    assert session.name == 'Qualifying'
    assert session.load_laps() is not None


@pytest.mark.f1telapi
def test_to_session():
    Cache.enable_cache('test_cache')
    r = schedule.pick_session_by_date('2021-08-01')
    session = r.to_session()
    assert isinstance(session, core.Session)
    assert session.load_laps() is not None


def test_normalize_session_name():
    test_values = (
        (('fp1', 'FP1', 'fP1', 'practice 1', 'Practice 1', 'PracTicE 1',
          'free practice 1', 'Free Practice 1', 'fReE PracTIce 1'),
         'Practice 1'),
        (('q', 'Q', 'qualifying', 'Qualifying', 'quALifYing'), 'Qualifying'),
        (('sq', 'SQ', 'sQ', 'sprint qualifying', 'Sprint Qualifying',
          'sPriNT quAlIfYING'), 'Sprint Qualifying'),
        (('r', 'R', 'race', 'Race', 'rAcE'), 'Race')
    )
    for session_names, expected in test_values:
        for name in session_names:
            assert events.normalize_session_name(name) == expected

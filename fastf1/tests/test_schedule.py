import datetime

import pandas as pd
import pytest

from fastf1 import events


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
        session = schedule.pick_session_by_number(session_type, 10)
        assert isinstance(session, events.EventSchedule)
        assert 'BRITISH GRAND PRIX' in str(session['Name'])
        assert session['Session'].iloc[0] == 'Race'


@pytest.mark.parametrize(
    "date", ['2021-08-01', '8/1/2021', datetime.date(2021, 8, 1),
             pd.Timestamp(year=2021, month=8, day=1)])  # TODO not only closest but limit to same day
def test_pick_session_by_date(date):
    session = schedule.pick_session_by_date(date)
    assert isinstance(session, pd.Series)
    assert 'FORMULA 1 ROLEX MAGYAR' in str(session['Name'])
    assert session['Session'] == 'Race'


def test_pick_session_by_date_same_day():
    fp3 = schedule.pick_session_by_date('2021-07-31 10:00')
    assert isinstance(fp3, pd.Series)
    assert fp3['Session'] == 'Practice 3'

    q = schedule.pick_session_by_date('2021-07-31 14:00')
    assert isinstance(q, pd.Series)
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

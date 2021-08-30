import logging

import pandas as pd
import ics

from fastf1.api import Cache
from fastf1.core import Weekend, Session


def get_event_schedule():
    schedule = Cache.request_and_parse_modified(
        'https://www.formula1.com/calendar/Formula_1_Official_Calendar.ics',
        _parse_ics_event_schedule,
        name='event schedule', cache_name='event_schedule',
        skip_update_check=True
    )

    return schedule


def _parse_ics_event_schedule(response):
    response.encoding = 'uft-8'  # set proper encoding
    text = response.text.replace('\r', '')
    lines = text.split('\n')
    new_lines = list()
    # strip all but first 'METHOD'
    method_match = False
    for line in lines:
        if 'METHOD' in line and not method_match:
            method_match = True
            new_lines.append(line)
        elif 'METHOD' in line:
            continue
        else:
            new_lines.append(line)

    cal_text = '\r\n'.join(new_lines)
    calendar = ics.Calendar(cal_text)

    data = {'Name': list(), 'Session': list(), 'Location': list(),
            'Begin': list(), 'End': list(), 'Duration': list(),
            'Status': list(), 'EventType': list()}

    for ev in calendar.events:
        if ' - ' in ev.name:
            split_char = ' - '
        elif ' – ' in ev.name:
            split_char = ' – '
        else:
            logging.warning(
                f"Failed to parse event with name '{ev.name}'. "
                f"Event will be skipped."
            )
            continue

        split_name = ev.name.split(split_char)
        # strip the cancelled info out of the name; redundant info
        for i in range(len(split_name)-1, -1, -1):
            if 'cancelled' in split_name[i].lower():
                split_name.pop(i)

        data['Name'].append(split_name[0])

        if 'pre-season' in ev.name.lower():
            # Pre-season testing is called Practice 1/2/3 which is confusing
            # change it to Pre-Season Testing 1/2/3
            session = split_name[1]
            n = [c for c in session if c.isnumeric()][0]
            session = f'Pre-Season Testing {n}'
            data['Session'].append(session)
            data['EventType'].append('testing')

        else:
            data['Session'].append(split_name[1])
            data['EventType'].append('race_weekend')

        data['Location'].append(ev.location.rstrip(' '))
        # remove trailing whitespace from location

        data['Begin'].append(ev.begin.datetime)
        data['End'].append(ev.end.datetime)
        data['Duration'].append(ev.duration)
        data['Status'].append(ev.status.lower())

    schedule = EventSchedule(data)\
        .sort_values(by='Begin').reset_index(drop=True)

    # make date timezone-naive
    schedule['Begin'] = schedule['Begin'].dt.tz_localize(None)
    schedule['End'] = schedule['End'].dt.tz_localize(None)

    # group schedule by event (name) and add event count
    schedule['EventNumber'] = int(0)  # set to zero as default for all first
    grouped_schedule = schedule.groupby('Name', sort=False)
    evn = 1
    for event in grouped_schedule:
        # if the race is confirmed, count it as a valid weekend
        if (event[1].iloc[-1]['Status'] == 'confirmed' and
                'testing' not in event[1].iloc[-1]['Session'].lower()):
            schedule.loc[schedule['Name'] == event[0], 'EventNumber'] = evn
            evn += 1

    return schedule


class EventSchedule(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return EventSchedule

    @property
    def base_class_view(self):
        """For a nicer debugging experience; can now view as
        dataframe in various IDEs"""
        return pd.DataFrame(self)

    def pick_weekend_by_number(self, number):
        return self[self['EventNumber'] == number]

    def pick_weekend_by_date(self, date):
        date = pd.to_datetime(date)
        tdelta = (self['Begin'].round('D') - date.round('D')).abs()
        if tdelta.min() > pd.Timedelta(1, 'days'):
            raise ValueError("No matches the given date.")
        name = self.iloc[tdelta.idxmin()]['Name']
        return self[self['Name'] == name]

    def pick_session_type(self, session):
        translate = {'fp1': 'Practice 1', 'fp2': 'Practice 2',
                     'fp3': 'Practice 3', 'q': 'Qualifying',
                     'r': 'Race', 'sq': 'Sprint Qualifying'}
        session = translate.get(session.lower(), session)
        session = ' '.join(word.capitalize() for word in session.split(' '))
        return self[self['Session'] == session]

    def pick_session_by_number(self, session, number):
        sessions = self.pick_session_type(session)
        return sessions[sessions['EventNumber'] == number]

    def pick_session_by_date(self, timestamp):
        date = pd.to_datetime(timestamp)
        same_day = self.loc[(self['Begin'].dt.year == date.year) &
                            (self['Begin'].dt.month == date.month) &
                            (self['Begin'].dt.day == date.day)]
        tdelta = (same_day['Begin'] - date).abs()
        return self.iloc[tdelta.idxmin()]

    def pick_confirmed(self):
        return self[self['Status'] == 'confirmed']

    def pick_race_weekends(self):
        return self[self['EventType'] == 'race_weekend']

    def to_weekend(self):
        if (n := len(self['Name'].unique())) > 1:
            raise ValueError(
                "Cannot create `.core.Weekend` from `.events.Schedule` if the"
                "schedule contains multiple race weekends!"
            )
        elif n == 0:
            return None

        session = self.iloc[0]
        return Weekend(session['Begin'].year, session['EventNumber'])

    def to_session(self):
        if (n := len(self)) > 1:
            raise ValueError(
                "Cannot create `.core.Weekend` from `.events.Schedule` if the"
                "schedule contains multiple race weekends!"
            )
        elif n == 0:
            return None

        weekend = self.to_weekend()
        return Session(weekend, self['Session'].squeeze())

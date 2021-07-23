import pandas as pd
import ics

from fastf1.api import Cache


def get_event_schedule():
    schedule = Cache.request_and_parse_modified(
        'https://www.formula1.com/calendar/Formula_1_Official_Calendar.ics',
        _parse_ics_event_schedule,
        name='event schedule', cache_name='event_schedule',
        skip_update_check=True
    )

    return schedule


def _parse_ics_event_schedule(response_text):
    lines = response_text.split('\r\n')
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
            'Status': list()}

    for ev in calendar.events:
        data['Name'].append(ev.name.split(' -')[0])
        # remove session type info from event name
        if 'pre-season' in ev.name.lower():
            session = ev.name.split('- ')[1]
            session = ''.join([c for c in session if c.isnumeric()])
            session = 'Pre-Season Testing ' + session
            data['Session'].append(session)
        else:
            data['Session'].append(ev.name.split('- ')[1])

        # session type can only be properly determined from name
        data['Location'].append(ev.location.rstrip(' '))
        # remove trailing whitespace from location
        data['Begin'].append(ev.begin.datetime)
        data['End'].append(ev.end.datetime)
        data['Duration'].append(ev.duration)
        data['Status'].append(ev.status)

    schedule = EventSchedule(data)\
        .sort_values(by='Begin').reset_index(drop=True)

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

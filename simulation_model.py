# In simulation_model.py

import simpy
import random
import statistics
import numpy as np

ORDER_BOOK_SIZE = 10
# NEW: Time constants in minutes
MINS_IN_HOUR = 60
MINS_IN_DAY = 24 * MINS_IN_HOUR
MINS_IN_WEEK = 7 * MINS_IN_DAY
# Factory schedule
DAY_START_MINS = 8 * MINS_IN_HOUR    # 8 AM
LUNCH_START_MINS = 12 * MINS_IN_HOUR   # 12 PM
LUNCH_END_MINS = 13 * MINS_IN_HOUR     # 1 PM
NORMAL_DAY_END_MINS = 18.5 * MINS_IN_HOUR # 6:30 PM
OVERTIME_DAY_END_MINS = 20.5 * MINS_IN_HOUR # 8:30 PM

class AssemblyLineSim:
    def __init__(self, part_mix=None, priority_mix=None, fail_rate=None):
        self.BUFFER_CAPACITY = 10
        self.REPAIR_TIME = 30
        self.FAIL_RATE = fail_rate if fail_rate is not None else 0.08
        self.PART_MIX = part_mix if part_mix is not None else {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15}
        self.PRIORITY_MIX = priority_mix if priority_mix is not None else {'HIGH': 0.2, 'LOW': 0.8}
        self.PART_CONFIGS = {
            'Type_A': {'s1_time': 9, 's2_time': 20, 's3_time': 8, 'type_id': 0},
            'Type_B': {'s1_time': 10, 's2_time': 25, 's3_time': 7, 'type_id': 1},
            'Type_C': {'s1_time': 8,  's2_time': 18, 's3_time': 9, 'type_id': 2},
        }
        self.PART_TYPES = list(self.PART_MIX.keys())
        self.PART_PROBABILITIES = list(self.PART_MIX.values())
        self.PRIORITY_MAP = {'HIGH': 1, 'LOW': 2}
        self.setup_simulation()

    def setup_simulation(self):
        self.env = simpy.Environment()
        self.stations = {
            'station1': simpy.PriorityResource(self.env, capacity=1),
            'station2': simpy.PriorityResource(self.env, capacity=1),
            'station3': simpy.PriorityResource(self.env, capacity=1),
            'repair_station': simpy.PriorityResource(self.env, capacity=1)
        }
        self.buffers = {
            'buffer12': simpy.Store(self.env, capacity=self.BUFFER_CAPACITY),
            'buffer23': simpy.Store(self.env, capacity=self.BUFFER_CAPACITY)
        }
        self.completed_parts = []
        self.source_halted = False
        self.station_busy_time = {name: 0 for name in self.stations.keys()}
        self.parts_processed_per_station = {'station1': 0, 'station2': 0, 'station3': 0}
        self.order_book = []
        self.part_id_counter = 0
        self.event_log = []
        
        # NEW: A dictionary to track which part is in which station
        self.parts_in_stations = {name: None for name in self.stations.keys()}
        
        self._fill_order_book()
        self.overtime_active_today = False
        self.env.process(self._master_schedule_process())

    def _fill_order_book(self):
        while len(self.order_book) < ORDER_BOOK_SIZE:
            self.part_id_counter += 1
            part_type = np.random.choice(self.PART_TYPES, p=self.PART_PROBABILITIES)
            priority_name = np.random.choice(list(self.PRIORITY_MIX.keys()), p=list(self.PRIORITY_MIX.values()))
            priority = self.PRIORITY_MAP[priority_name]
            due_date = self.env.now + (240 if priority == 1 else 720)
            part = {
                "id": self.part_id_counter, "type": part_type, "config": self.PART_CONFIGS[part_type],
                "priority": priority, "arrival_time": self.env.now, "due_date": due_date
            }
            self.order_book.append(part)

    def release_part(self, order_index):
         # Check if factory is open before releasing
        time_of_day = self.env.now % MINS_IN_DAY
        day_of_week = (self.env.now // MINS_IN_DAY) % 7
        end_of_day = OVERTIME_DAY_END_MINS if self.overtime_active_today else NORMAL_DAY_END_MINS
        is_work_hours = (day_of_week < 6) and (DAY_START_MINS <= time_of_day < end_of_day)
        is_lunch_break = (LUNCH_START_MINS <= time_of_day < LUNCH_END_MINS)

        if is_work_hours and not is_lunch_break and not self.source_halted and self.order_book and 0 <= order_index < len(self.order_book):
            part_to_release = self.order_book.pop(order_index)
            # MODIFIED: Add timestamp to the event log
            event = (self.env.now, f"Part-{part_to_release['id']} ({'HIGH' if part_to_release['priority']==1 else 'LOW'}) released into Station 1.")
            self.event_log.append(event)
            self.env.process(self._part_process(part_to_release))
            self._fill_order_book()

    def _part_process(self, part):
        part_id_str = f"Part-{part['id']}"
        request_priority = part['priority']

        with self.stations['station1'].request(priority=request_priority) as req:
            yield req
            self.parts_in_stations['station1'] = part # Part enters station
            self.event_log.append((self.env.now, f"{part_id_str} started at Station 1."))
            yield self.env.timeout(part['config']['s1_time'])
        self.parts_in_stations['station1'] = None # Part leaves station
        
        yield self.buffers['buffer12'].put(part)
        self.event_log.append((self.env.now, f"{part_id_str} entered Buffer 1->2."))
        
        tested_successfully = False
        while not tested_successfully:
            part_from_buffer = yield self.buffers['buffer12'].get()
            self.event_log.append((self.env.now, f"Part-{part_from_buffer['id']} left Buffer 1->2 for Station 2."))
            with self.stations['station2'].request(priority=part_from_buffer['priority']) as req:
                yield req
                self.parts_in_stations['station2'] = part_from_buffer
                self.event_log.append((self.env.now, f"Part-{part_from_buffer['id']} started at Station 2."))
                yield self.env.timeout(part_from_buffer['config']['s2_time'])
            self.parts_in_stations['station2'] = None
            
            if random.random() > self.FAIL_RATE:
                tested_successfully = True
                self.parts_processed_per_station['station2'] += 1
                yield self.buffers['buffer23'].put(part_from_buffer)
                self.event_log.append((self.env.now, f"Part-{part_from_buffer['id']} PASSED, entered Buffer 2->3."))
            else:
                self.event_log.append((self.env.now, f"Part-{part_from_buffer['id']} FAILED, moving to Repair."))
                with self.stations['repair_station'].request(priority=part_from_buffer['priority']) as repair_req:
                    yield repair_req
                    self.parts_in_stations['repair_station'] = part_from_buffer
                    yield self.env.timeout(self.REPAIR_TIME)
                self.parts_in_stations['repair_station'] = None
                yield self.buffers['buffer12'].put(part_from_buffer)
                self.event_log.append((self.env.now, f"Part-{part_from_buffer['id']} finished repair, re-entered Buffer 1->2."))
        
        part_from_buffer_2 = yield self.buffers['buffer23'].get()
        self.event_log.append((self.env.now, f"Part-{part_from_buffer_2['id']} left Buffer 2->3 for Station 3."))
        with self.stations['station3'].request(priority=part_from_buffer_2['priority']) as req:
            yield req
            self.parts_in_stations['station3'] = part_from_buffer_2
            self.event_log.append((self.env.now, f"Part-{part_from_buffer_2['id']} started at Station 3."))
            yield self.env.timeout(part_from_buffer_2['config']['s3_time'])
        self.parts_in_stations['station3'] = None
        
        part['finish_time'] = self.env.now
        part['cycle_time'] = part['finish_time'] - part['arrival_time']
        part['is_late'] = part['finish_time'] > part['due_date']
        self.completed_parts.append(part)
        self.event_log.append((self.env.now, f"✅ Part-{part['id']} COMPLETED the line!"))

    def get_kpis_and_state(self):
        obs = {
            "buffer_12_level": len(self.buffers['buffer12'].items),
            "buffer_23_level": len(self.buffers['buffer23'].items),
            "order_book": self.order_book,
            "parts_in_stations": self.parts_in_stations # Return parts in stations
        }
        results = { "newly_completed_parts": self.completed_parts, "events": self.event_log }
        self.completed_parts = []
        self.event_log = []
        return obs, results
    
    # Method for agent to set overtime for the current day
    def set_overtime_status(self, status):
        self.overtime_active_today = status

    def set_source_status(self, halt_status):
        self.source_halted = halt_status

        # NEW: Master scheduler to manage open/close times and breaks
    def _master_schedule_process(self):
        while True:
            # Determine current time and day
            now = self.env.now
            day_of_week = (now // MINS_IN_DAY) % 7
            
            # --- Handle Sunday ---
            if day_of_week == 6: # It's Sunday
                time_until_monday_8am = MINS_IN_DAY - (now % MINS_IN_DAY) + DAY_START_MINS
                requests = [s.request(priority=-1) for s in self.stations.values()]
                yield simpy.AllOf(self.env, requests)
                yield self.env.timeout(time_until_monday_8am)
                # CORRECTED LINE
                for req in requests: req.resource.release(req)
                continue

            # --- Handle Workday Breaks and Closing ---
            # Lunch Break
            time_until_lunch = LUNCH_START_MINS - (now % MINS_IN_DAY)
            if time_until_lunch > 0:
                yield self.env.timeout(time_until_lunch)
            
            requests = [s.request(priority=-1) for s in self.stations.values()]
            yield simpy.AllOf(self.env, requests)
            yield self.env.timeout(LUNCH_END_MINS - LUNCH_START_MINS)
            # CORRECTED LINE
            for req in requests: req.resource.release(req)

            # End of Day Closing
            end_time = OVERTIME_DAY_END_MINS if self.overtime_active_today else NORMAL_DAY_END_MINS
            time_until_eod = end_time - (self.env.now % MINS_IN_DAY)
            if time_until_eod > 0:
                yield self.env.timeout(time_until_eod)

            requests = [s.request(priority=-1) for s in self.stations.values()]
            yield simpy.AllOf(self.env, requests)
            
            time_until_next_day_8am = MINS_IN_DAY - (self.env.now % MINS_IN_DAY) + DAY_START_MINS
            yield self.env.timeout(time_until_next_day_8am)
            # CORRECTED LINE
            for req in requests: req.resource.release(req)
            self.overtime_active_today = False

    def run(self, duration):
        self.env.run(until=self.env.now + duration)


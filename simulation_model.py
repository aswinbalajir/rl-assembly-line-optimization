# simulation_model.py

import random
import statistics
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import simpy

# --- Constants ---

ORDER_BOOK_SIZE = 10

# Time Constants (Minutes)
MINS_IN_HOUR = 60
MINS_IN_DAY = 24 * MINS_IN_HOUR
MINS_IN_WEEK = 7 * MINS_IN_DAY

# Factory Schedule
DAY_START_MINS = 8 * MINS_IN_HOUR           # 8:00 AM
LUNCH_START_MINS = 12 * MINS_IN_HOUR        # 12:00 PM
LUNCH_END_MINS = 13 * MINS_IN_HOUR          # 1:00 PM
NORMAL_DAY_END_MINS = 18.5 * MINS_IN_HOUR   # 6:30 PM
OVERTIME_DAY_END_MINS = 22.5 * MINS_IN_HOUR # 8:30 PM


class AssemblyLineSim:
    """
    A SimPy-based discrete event simulation of an assembly line.
    
    This simulation models a 3-station assembly line with buffers, failure rates,
    repair logic, and a master schedule that handles shifts, breaks, and overtime.
    
    Attributes:
        env (simpy.Environment): The simulation environment.
        order_book (List[Dict]): Current list of pending orders.
        stations (Dict): Dictionary of simpy.PriorityResource objects.
    """

    def __init__(
        self, 
        part_mix: Optional[Dict[str, float]] = None, 
        priority_mix: Optional[Dict[str, float]] = None, 
        fail_rate: Optional[float] = None
    ):
        """
        Initialize the simulation parameters.

        Args:
            part_mix (dict, optional): Probability distribution of part types.
            priority_mix (dict, optional): Probability distribution of order priorities.
            fail_rate (float, optional): Probability of failure at the test station.
        """
        # Configuration
        self.BUFFER_CAPACITY = 10
        self.REPAIR_TIME = 30
        self.FAIL_RATE = fail_rate if fail_rate is not None else 0.08
        
        self.PART_MIX = part_mix if part_mix is not None else {
            'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15
        }
        self.PRIORITY_MIX = priority_mix if priority_mix is not None else {
            'HIGH': 0.2, 'LOW': 0.8
        }
        
        # Processing Times (Minutes)
        # self.PART_CONFIGS = {
        #     'Type_A': {'s1_time': 9,  's2_time': 20, 's3_time': 8, 'type_id': 0},
        #     'Type_B': {'s1_time': 10, 's2_time': 25, 's3_time': 7, 'type_id': 1},
        #     'Type_C': {'s1_time': 8,  's2_time': 18, 's3_time': 9, 'type_id': 2},
        # }
        # In simulation_model.py

        self.PART_CONFIGS = {
            # Tyepe A: The "Standard" Part (60% of mix)
            'Type_A': {'s1_time': 14, 's2_time': 18, 's3_time': 12, 'type_id': 0},

            # Type B: The "Complex" Part (25% of mix)
            'Type_B': {'s1_time': 18, 's2_time': 22, 's3_time': 15, 'type_id': 1},

            # Type C: The "Simple" Part (15% of mix)
            'Type_C': {'s1_time': 10, 's2_time': 14, 's3_time': 10, 'type_id': 2},
        }
        
        self.PART_TYPES = list(self.PART_MIX.keys())
        self.PART_PROBABILITIES = list(self.PART_MIX.values())
        self.PRIORITY_MAP = {'HIGH': 1, 'LOW': 2}
        
        # Internal State
        self.env: simpy.Environment = None
        self.stations: Dict[str, simpy.PriorityResource] = {}
        self.buffers: Dict[str, simpy.Store] = {}
        self.completed_parts: List[Dict] = []
        self.event_log: List[Tuple[float, str]] = []
        
        self.setup_simulation()

    def setup_simulation(self):
        """Resets the simulation environment and state variables."""
        self.env = simpy.Environment()
        
        # Resources (Priority 1 = High, 2 = Low, -1 = Master Schedule Interrupt)
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
        
        # Tracking
        self.completed_parts = []
        self.source_halted = False
        self.station_busy_time = {name: 0.0 for name in self.stations.keys()}
        self.parts_processed_per_station = {'station1': 0, 'station2': 0, 'station3': 0}
        self.parts_in_stations = {name: None for name in self.stations.keys()}
        
        self.order_book = []
        self.part_id_counter = 0
        self.event_log = []
        
        self._fill_order_book()
        self.overtime_active_today = False
        
        # Start Master Process
        self.env.process(self._master_schedule_process())

    def _fill_order_book(self):
        """Replenishes the order book with new random orders."""
        while len(self.order_book) < ORDER_BOOK_SIZE:
            self.part_id_counter += 1
            
            # Select Type and Priority
            part_type = np.random.choice(self.PART_TYPES, p=self.PART_PROBABILITIES)
            priority_name = np.random.choice(
                list(self.PRIORITY_MIX.keys()), 
                p=list(self.PRIORITY_MIX.values())
            )
            priority = self.PRIORITY_MAP[priority_name]
            
            # Due Date Logic (4 hours for High, 12 hours for Low)
            due_date = self.env.now + (600 if priority == 1 else 1440)
            
            part = {
                "id": self.part_id_counter,
                "type": part_type,
                "config": self.PART_CONFIGS[part_type],
                "priority": priority,
                "arrival_time": self.env.now,
                "due_date": due_date
            }
            self.order_book.append(part)

    def release_part(self, order_index: int):
        """
        Releases a part from the order book into the assembly line.

        Args:
            order_index (int): Index of the part in the order book to release.
        """
        # 1. Check Factory Status (Open/Closed/Break)
        time_of_day = self.env.now % MINS_IN_DAY
        day_of_week = (self.env.now // MINS_IN_DAY) % 7
        end_of_day = OVERTIME_DAY_END_MINS if self.overtime_active_today else NORMAL_DAY_END_MINS
        
        is_work_hours = (day_of_week < 6) and (DAY_START_MINS <= time_of_day < end_of_day)
        is_lunch_break = (LUNCH_START_MINS <= time_of_day < LUNCH_END_MINS)

        # 2. Release Validation
        if (is_work_hours and 
            not is_lunch_break and 
            not self.source_halted and 
            self.order_book and 
            0 <= order_index < len(self.order_book)):
            
            part_to_release = self.order_book.pop(order_index)
            
            # Dynamic Priority Promotion
            # If low priority but due within 4 hours (240 mins), bump to HIGH
            # try:
            #     due_in = part_to_release['due_date'] - self.env.now
            #     if part_to_release.get('priority', 2) != 1 and due_in <= 240:
            #         part_to_release['priority'] = 1
            #         self.event_log.append((
            #             self.env.now, 
            #             f"Part-{part_to_release['id']} PROMOTED to HIGH (due in {int(due_in)}m)."
            #         ))
            # except Exception:
            #     pass # Graceful fallback if keys missing

            # Log Release
            prio_label = 'HIGH' if part_to_release['priority'] == 1 else 'LOW'
            self.event_log.append((
                self.env.now, 
                f"Part-{part_to_release['id']} ({prio_label}) released into Station 1."
            ))
            
            # Start Process and Refill Book
            self.env.process(self._part_process(part_to_release))
            self._fill_order_book()

    def _part_process(self, part: Dict):
        """Simulates the lifecycle of a single part through the line."""
        part_id_str = f"Part-{part['id']}"
        request_priority = part['priority']

        # --- STATION 1 ---
        with self.stations['station1'].request(priority=request_priority) as req:
            yield req
            self.parts_in_stations['station1'] = part 
            self.event_log.append((self.env.now, f"{part_id_str} started at Station 1."))
            
            s1_t = part['config']['s1_time']
            yield self.env.timeout(s1_t)
            self.station_busy_time['station1'] += s1_t
            
        self.parts_in_stations['station1'] = None 
        
        yield self.buffers['buffer12'].put(part)
        self.event_log.append((self.env.now, f"{part_id_str} entered Buffer 1->2."))
        
        # --- STATION 2 (Bottleneck + Testing) ---
        tested_successfully = False
        while not tested_successfully:
            part_from_buffer = yield self.buffers['buffer12'].get()
            
            with self.stations['station2'].request(priority=part_from_buffer['priority']) as req:
                yield req
                self.parts_in_stations['station2'] = part_from_buffer
                
                s2_t = part_from_buffer['config']['s2_time']
                yield self.env.timeout(s2_t)
                self.station_busy_time['station2'] += s2_t

            self.parts_in_stations['station2'] = None
            
            # Quality Check
            if random.random() > self.FAIL_RATE:
                tested_successfully = True
                self.parts_processed_per_station['station2'] += 1
                yield self.buffers['buffer23'].put(part_from_buffer)
            else:
                # Failure -> Repair Loop
                with self.stations['repair_station'].request(priority=part_from_buffer['priority']) as repair_req:
                    yield repair_req
                    self.parts_in_stations['repair_station'] = part_from_buffer
                    
                    yield self.env.timeout(self.REPAIR_TIME)
                    self.station_busy_time['repair_station'] += self.REPAIR_TIME
                
                self.parts_in_stations['repair_station'] = None
                # Send back to Buffer 1->2 for re-test
                yield self.buffers['buffer12'].put(part_from_buffer)
        
        # --- STATION 3 ---
        part_from_buffer_2 = yield self.buffers['buffer23'].get()
        with self.stations['station3'].request(priority=part_from_buffer_2['priority']) as req:
            yield req
            self.parts_in_stations['station3'] = part_from_buffer_2
            
            s3_t = part_from_buffer_2['config']['s3_time']
            yield self.env.timeout(s3_t)
            self.station_busy_time['station3'] += s3_t

        self.parts_in_stations['station3'] = None
        
        # Finalize Metrics
        part['finish_time'] = self.env.now
        part['cycle_time'] = part['finish_time'] - part['arrival_time']
        part['is_late'] = part['finish_time'] > part['due_date']
        self.completed_parts.append(part)

    def get_kpis_and_state(self) -> Tuple[Dict, Dict]:
        """Returns current state snapshot and recent results."""
        obs = {
            "buffer_12_level": len(self.buffers['buffer12'].items),
            "buffer_23_level": len(self.buffers['buffer23'].items),
            "order_book": self.order_book,
            "parts_in_stations": self.parts_in_stations
        }
        results = { 
            "newly_completed_parts": self.completed_parts, 
            "events": self.event_log 
        }
        
        # Reset transient logs
        self.completed_parts = []
        self.event_log = []
        
        return obs, results
    
    def set_overtime_status(self, status: bool):
        """Updates overtime flag for the current day."""
        self.overtime_active_today = status

    def set_source_status(self, halt_status: bool):
        """Updates the source halt status."""
        self.source_halted = halt_status

    def _master_schedule_process(self):
        """
        Manages the factory clock, enforcing breaks and shifts.
        
        Uses priority -1 (highest possible in SimPy default is integer based, 
        smaller is higher priority, but standard SimPy PriorityResource uses 
        integers where smaller = higher priority. If using default, we must ensure 
        this preempts normal work.)
        
        Note: SimPy PriorityResource prioritizes smaller numbers. 
        Normal parts are 1 (High) or 2 (Low).
        We use -1 to ensure the scheduler preempts everything.
        """
        while True:
            now = self.env.now
            day_of_week = (now // MINS_IN_DAY) % 7
            
            # --- SUNDAY (Closed) ---
            if day_of_week == 6:
                time_until_monday = MINS_IN_DAY - (now % MINS_IN_DAY) + DAY_START_MINS
                
                # Lock all stations
                reqs = [s.request(priority=-1) for s in self.stations.values()]
                yield simpy.AllOf(self.env, reqs)
                
                yield self.env.timeout(time_until_monday)
                
                # Unlock
                for req in reqs:
                    req.resource.release(req)
                continue

            # --- WORKDAY ---
            
            # 1. Wait until Lunch
            time_until_lunch = LUNCH_START_MINS - (now % MINS_IN_DAY)
            if time_until_lunch > 0:
                yield self.env.timeout(time_until_lunch)
            
            # 2. Lunch Break (Lock Stations)
            reqs = [s.request(priority=-1) for s in self.stations.values()]
            yield simpy.AllOf(self.env, reqs)
            
            yield self.env.timeout(LUNCH_END_MINS - LUNCH_START_MINS)
            
            # Unlock after lunch
            for req in reqs:
                req.resource.release(req)

            # 3. Work until End of Day
            end_time = OVERTIME_DAY_END_MINS if self.overtime_active_today else NORMAL_DAY_END_MINS
            time_until_eod = end_time - (self.env.now % MINS_IN_DAY)
            
            if time_until_eod > 0:
                yield self.env.timeout(time_until_eod)

            # 4. End of Day Closing (Lock Stations)
            reqs = [s.request(priority=-1) for s in self.stations.values()]
            yield simpy.AllOf(self.env, reqs)
            
            time_until_next_start = MINS_IN_DAY - (self.env.now % MINS_IN_DAY) + DAY_START_MINS
            yield self.env.timeout(time_until_next_start)
            
            # Unlock for next day
            for req in reqs:
                req.resource.release(req)
            
            # Reset daily flags
            self.overtime_active_today = False

    def run(self, duration: int):
        """Advances the simulation by a specific duration."""
        self.env.run(until=self.env.now + duration)
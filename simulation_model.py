# In simulation_model.py

import simpy
import random
import statistics
import numpy as np

ORDER_BOOK_SIZE = 10

class AssemblyLineSim:
    def __init__(self):
        self.BUFFER_CAPACITY = 10
        self.INTER_ARRIVAL_TIME = 12
        self.FAIL_RATE = 0.08
        self.REPAIR_TIME = 30
        self.PART_CONFIGS = {
            'Type_A': {'s1_time': 10, 's2_time': 20, 's3_time': 8, 'type_id': 0},
            'Type_B': {'s1_time': 12, 's2_time': 25, 's3_time': 7, 'type_id': 1},
            'Type_C': {'s1_time': 8,  's2_time': 18, 's3_time': 9, 'type_id': 2},
        }
        self.PART_MIX = {'Type_A': 0.60, 'Type_B': 0.25, 'Type_C': 0.15}
        self.PART_TYPES = list(self.PART_MIX.keys())
        self.PART_PROBABILITIES = list(self.PART_MIX.values())
        self.PRIORITY_MIX = {'HIGH': 0.20, 'LOW': 0.80}
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
        self._fill_order_book()

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
        if not self.source_halted and self.order_book and 0 <= order_index < len(self.order_book):
            part_to_release = self.order_book.pop(order_index)
            self.env.process(self._part_process(part_to_release))
            self._fill_order_book()

# In simulation_model.py

    def _part_process(self, part):
        arrival_time = self.env.now
        request_priority = part['priority']

        # --- Station 1 ---
        with self.stations['station1'].request(priority=request_priority) as req:
            yield req
            start_proc_time = self.env.now
            yield self.env.timeout(part['config']['s1_time'])
            self.station_busy_time['station1'] += self.env.now - start_proc_time
        
        yield self.buffers['buffer12'].put(part)

        # --- Station 2 & Repair Loop ---
        tested_successfully = False
        while not tested_successfully:
            part_from_buffer = yield self.buffers['buffer12'].get()
            with self.stations['station2'].request(priority=request_priority) as req:
                yield req
                start_proc_time = self.env.now
                yield self.env.timeout(part_from_buffer['config']['s2_time'])
                self.station_busy_time['station2'] += self.env.now - start_proc_time

            # CRITICAL FIX: Re-introducing the failure check logic
            if random.random() > self.FAIL_RATE:
                tested_successfully = True
                self.parts_processed_per_station['station2'] += 1
                yield self.buffers['buffer23'].put(part_from_buffer)
            else:
                # Part failed, send to repair
                with self.stations['repair_station'].request(priority=request_priority) as repair_req:
                    yield repair_req
                    start_repair_time = self.env.now
                    yield self.env.timeout(self.REPAIR_TIME)
                    self.station_busy_time['repair_station'] += self.env.now - start_repair_time
                # After repair, it goes back into the queue for Station 2
                yield self.buffers['buffer12'].put(part_from_buffer)
        
        # --- Station 3 ---
        part_from_buffer_2 = yield self.buffers['buffer23'].get()
        with self.stations['station3'].request(priority=request_priority) as req:
            yield req
            start_proc_time = self.env.now
            yield self.env.timeout(part_from_buffer_2['config']['s3_time'])
            self.station_busy_time['station3'] += self.env.now - start_proc_time
        
        # --- Final Recording ---
        part['finish_time'] = self.env.now
        part['cycle_time'] = part['finish_time'] - part['arrival_time']
        part['is_late'] = part['finish_time'] > part['due_date']
        self.completed_parts.append(part)

    def get_kpis_and_state(self):
        obs = {
            "buffer_12_level": len(self.buffers['buffer12'].items),
            "buffer_23_level": len(self.buffers['buffer23'].items),
            "order_book": self.order_book
        }
        results = { "newly_completed_parts": self.completed_parts }
        self.completed_parts = []
        return obs, results

    def set_source_status(self, halt_status):
        self.source_halted = halt_status

    def run(self, duration):
        self.env.run(until=self.env.now + duration)
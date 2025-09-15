# In simulation_model.py

import simpy
import random
import statistics
import numpy as np

class AssemblyLineSim:
    def __init__(self):
        # --- Simulation Parameters ---
        self.BUFFER_CAPACITY = 10
        self.INTER_ARRIVAL_TIME = 12
        self.FAIL_RATE = 0.08
        self.REPAIR_TIME = 30

        self.PART_CONFIGS = {
            'Type_A': {'s1_time': 10, 's2_time': 20, 's3_time': 8},
            'Type_B': {'s1_time': 12, 's2_time': 25, 's3_time': 7},
            'Type_C': {'s1_time': 8,  's2_time': 18, 's3_time': 9},
        }
        self.PART_MIX = {'Type_A': 0.60, 'Type_B': 0.25, 'Type_C': 0.15}
        self.PART_TYPES = list(self.PART_MIX.keys())
        self.PART_PROBABILITIES = list(self.PART_MIX.values())
        
        self.PRIORITY_MIX = {'HIGH': 0.20, 'LOW': 0.80}
        self.PRIORITY_MAP = {'HIGH': 1, 'LOW': 2}
        
        self.setup_simulation()

    def setup_simulation(self):
        """Initializes or resets the simulation environment and data structures."""
        self.env = simpy.Environment()
        
        self.stations = {
            'station1': simpy.PriorityResource(self.env, capacity=1),
            'station2': simpy.PriorityResource(self.env, capacity=1),
            'station3': simpy.PriorityResource(self.env, capacity=1),
            'repair_station': simpy.PriorityResource(self.env, capacity=1)
        }
        self.buffers = {
            'buffer12': simpy.Container(self.env, capacity=self.BUFFER_CAPACITY, init=0),
            'buffer23': simpy.Container(self.env, capacity=self.BUFFER_CAPACITY, init=0)
        }
        
        # --- Data Collection ---
        self.parts_completed_total = 0
        self.station_busy_time = {name: 0 for name in self.stations.keys()}
        self.last_observation_time = 0

        # Start the simulation processes
        self.env.process(self._part_source())

    def _part_process(self, part_name, part_priority, part_config):
        # This is the same logic as before, just as a method of the class
        with self.stations['station1'].request(priority=part_priority) as req:
            yield req
            start_proc_time = self.env.now
            yield self.env.timeout(part_config['s1_time'])
            self.station_busy_time['station1'] += self.env.now - start_proc_time

        yield self.buffers['buffer12'].put(1)

        tested_successfully = False
        while not tested_successfully:
            with self.stations['station2'].request(priority=part_priority) as req:
                yield req
                yield self.buffers['buffer12'].get(1)
                start_proc_time = self.env.now
                yield self.env.timeout(part_config['s2_time'])
                self.station_busy_time['station2'] += self.env.now - start_proc_time

            if random.random() > self.FAIL_RATE:
                tested_successfully = True
                yield self.buffers['buffer23'].put(1)
            else:
                with self.stations['repair_station'].request(priority=part_priority) as repair_req:
                    yield repair_req
                    start_repair_time = self.env.now
                    yield self.env.timeout(self.REPAIR_TIME)
                    self.station_busy_time['repair_station'] += self.env.now - start_repair_time
                yield self.buffers['buffer12'].put(1)

        with self.stations['station3'].request(priority=part_priority) as req:
            yield req
            yield self.buffers['buffer23'].get(1)
            start_proc_time = self.env.now
            yield self.env.timeout(part_config['s3_time'])
            self.station_busy_time['station3'] += self.env.now - start_proc_time
        
        self.parts_completed_total += 1

    def _part_source(self):
        part_id = 0
        while True:
            yield self.env.timeout(random.expovariate(1.0 / self.INTER_ARRIVAL_TIME))
            part_id += 1
            part_type = np.random.choice(self.PART_TYPES, p=self.PART_PROBABILITIES)
            part_config = self.PART_CONFIGS[part_type]
            priority_name = np.random.choice(list(self.PRIORITY_MIX.keys()), p=list(self.PRIORITY_MIX.values()))
            part_priority = self.PRIORITY_MAP[priority_name]
            part_name = f"Part-{part_id}({part_type}, P{part_priority})"
            self.env.process(self._part_process(part_name, part_priority, part_config))

    def get_kpis(self):
        """Calculates and returns the current KPIs of the system."""
        # This is a simplified KPI snapshot for the RL state
        return {
            "buffer_12_level": self.buffers['buffer12'].level,
            "buffer_23_level": self.buffers['buffer23'].level,
            "station_2_utilization": (self.station_busy_time['station2'] / (self.env.now + 1e-6)),
            "repair_station_utilization": (self.station_busy_time['repair_station'] / (self.env.now + 1e-6)),
            "parts_completed_total": self.parts_completed_total
        }

    def run(self, duration):
        """Runs the simulation for a given duration."""
        self.env.run(until=self.env.now + duration)
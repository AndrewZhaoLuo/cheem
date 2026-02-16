from problem import SLOT_LIMITS, Engine
from typing import List, Dict, Tuple, Set, Deque
from collections import defaultdict, deque
from queue import PriorityQueue


def get_dests(slot: tuple) -> Set[int]:
    engine, inst = slot
    if engine in ["valu", "alu", "load", "store", "hint"]:
        return {inst[1]}

    if engine in ["flow"]:
        assert inst[0] in ["select", "vselect"]
        return {inst[1]}

    raise AssertionError(f"Unknown slot {slot}")


def get_srcs(slot: tuple) -> Set[int]:
    engine, inst = slot
    if engine in ["valu", "alu", "load", "store", "hint"]:
        if inst[0] in ["const"]:
            return set()

        return set(inst[2:])

    if engine in ["flow"]:
        assert inst[0] in ["select", "vselect"]
        return set(inst[2:])

    raise AssertionError(f"Unknown slot {slot}")


class Scheduler:
    def __init__(self, builder: "KernelBuilder", slots: list[tuple[Engine, tuple]]) -> None:
        self.builder = builder
        self.slots = [s for s in slots if s[0] != "debug"]

        # map of instruction slot to dependencies before slot can run
        # also have reverse
        self.dependency_map: List[Set[int]] = [set() for i in range(len(self.slots))]
        self.free_map: List[Set[int]] = [set() for i in range(len(self.slots))]
        self.calculate_dependencies()

    # add a dependency that instruction index ai --> bi
    def add_dependence(self, ai, bi):
        assert isinstance(ai, int)
        assert isinstance(bi, int)
        if isinstance(bi, set):
            for bii in bi:
                self.add_dependence(ai, bii)
        else:
            self.dependency_map[ai].add(bi)
            self.free_map[bi].add(ai)

    def calculate_dependencies(self):
        # map of addr --> last slot where it was written
        last_write: Dict[int, int] = {}

        # map of resource --> when it was last read in current state
        reads: Dict[int, Set[int]] = defaultdict(set)
        for i, slot in enumerate(self.slots):
            dests = get_dests(slot)
            srcs = get_srcs(slot)
            # print(i, slot, " --- ", dests, srcs)

            # all reads of previous state must be done
            for dest in dests:
                for prev_read_inst in reads[dest]:
                    self.add_dependence(i, prev_read_inst)

            # all srcs must have been written to
            for src in srcs:
                if src not in last_write:
                    continue
                last_instr = last_write[src]
                self.add_dependence(i, last_instr)

            # writes to a destination must be properly ordered
            for dest in dests:
                if dest not in last_write:
                    continue
                last_instr = last_write[dest]
                self.add_dependence(i, last_instr)

            # update last_write
            for dest in dests:
                last_write[dest] = i
                reads[dest].clear()

            # update reads
            for src in srcs:
                reads[src].add(i)

    def schedule_greedy(self) -> List[Dict[Engine, List[Tuple]]]:
        ready_slots: Dict[Engine, Deque[int]] = {
            "valu": deque(),
            "alu": deque(),
            "load": deque(),
            "store": deque(),
            "flow": deque(),
            "debug": deque(),
        }

        num_scheduled_slots = 0
        dep_count: List[int] = [len(self.dependency_map[i]) for i in range(len(self.slots))]

        def add_slot(i):
            slot = self.slots[i]
            engine, _ = slot
            ready_slots[engine].append(i)

        def free_dep(i):
            dep_count[i] -= 1
            assert dep_count[i] >= 0
            if dep_count[i] == 0:
                add_slot(i)

        def schedule_greedy() -> Dict[Engine, List[Tuple]]:
            nonlocal num_scheduled_slots

            bundle: Dict[Engine, List[Tuple]] = {"valu": [], "alu": [], "load": [], "store": [], "flow": []}
            for engine, limit in SLOT_LIMITS.items():
                while len(ready_slots[engine]) > 0 and len(bundle[engine]) < limit:
                    i = ready_slots[engine].popleft()
                    bundle[engine].append(i)

            # canonicalize things
            for k, v in bundle.items():
                for e, i in enumerate(v):
                    for j in self.free_map[i]:
                        free_dep(j)
                    v[e] = self.slots[i]
                    num_scheduled_slots += 1

            # resolve hints immediately
            while len(ready_slots["hint"]) > 0:
                next_hint_i = ready_slots["hint"].popleft()
                for j in self.free_map[next_hint_i]:
                    free_dep(j)
                num_scheduled_slots += 1

            return {k: [vv[1] for vv in v] for k, v in bundle.items() if len(v) > 0}

        for i, cnt in enumerate(dep_count):
            if cnt == 0:
                add_slot(i)

        answer = []
        while num_scheduled_slots < len(self.slots):
            bundle = schedule_greedy()
            # print(bundle)
            answer.append(bundle)
        return answer

    def schedule_critical_path(self) -> List[Dict[Engine, List[Tuple]]]:
        ready_slots: Dict[Engine, PriorityQueue] = {
            "valu": PriorityQueue(),
            "alu": PriorityQueue(),
            "load": PriorityQueue(),
            "store": PriorityQueue(),
            "flow": PriorityQueue(),
            "debug": PriorityQueue(),
            "hint": PriorityQueue(),
        }

        num_scheduled_slots = 0
        dep_count: List[int] = [len(self.dependency_map[i]) for i in range(len(self.slots))]

        # map of instruction index --> length / "priority"
        # the priority is negative because PriorityQueue takes the smallest values first
        path_lengths: Dict[int, int] = {}
        for i in range(len(self.slots) - 1, -1, -1):
            if i not in path_lengths:
                path_lengths[i] = 0
            for dep in self.dependency_map[i]:
                if dep not in path_lengths:
                    path_lengths[dep] = 0
                path_lengths[dep] = min(path_lengths[dep], path_lengths[i] - 1)

        def add_slot(i):
            slot = self.slots[i]
            engine, _ = slot
            priority = path_lengths[i]

            # We want to prioritize the load at all cost, do lookahead
            additional_priority_for_load = priority
            ready_slots[engine].put((i, additional_priority_for_load))

        def free_dep(i):
            dep_count[i] -= 1
            assert dep_count[i] >= 0
            if dep_count[i] == 0:
                add_slot(i)

        def schedule() -> Dict[Engine, List[Tuple]]:
            nonlocal num_scheduled_slots

            bundle: Dict[Engine, List[Tuple]] = {"valu": [], "alu": [], "load": [], "store": [], "flow": []}
            for engine, limit in SLOT_LIMITS.items():
                while not ready_slots[engine].empty() and len(bundle[engine]) < limit:
                    i, _priority = ready_slots[engine].get()
                    bundle[engine].append(i)

            # canonicalize things
            for k, v in bundle.items():
                for e, i in enumerate(v):
                    for j in self.free_map[i]:
                        free_dep(j)
                    v[e] = self.slots[i]
                    num_scheduled_slots += 1

            # resolve hints immediately
            while not ready_slots["hint"].empty():
                next_hint_i, _priority = ready_slots["hint"].get()
                for j in self.free_map[next_hint_i]:
                    free_dep(j)
                num_scheduled_slots += 1

            return {k: [vv[1] for vv in v] for k, v in bundle.items() if len(v) > 0}

        for i, cnt in enumerate(dep_count):
            if cnt == 0:
                add_slot(i)

        answer = []
        while num_scheduled_slots < len(self.slots):
            bundle = schedule()
            # print(num_scheduled_slots, '/', len(self.slots))
            # print(bundle)
            answer.append(bundle)
        return answer

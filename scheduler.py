from problem import SLOT_LIMITS, Engine, VLEN
from typing import List, Dict, Tuple, Set, Deque
from collections import defaultdict, deque
from queue import PriorityQueue


def get_dests(slot: tuple) -> Set[int]:
    engine, inst = slot
    op = inst[0]

    if engine == "valu" or (engine == "load" and op == "vload") or (engine == "flow" and op == "vselect"):
        # Vector destinations
        return set(range(inst[1], inst[1] + VLEN))
    elif engine in ["store", "debug"] or (engine == "flow" and op not in ["select", "vselect", "coreid", "add_imm"]):
        # No destination
        return set()
    else:
        # Scalar destination (alu, load, const, flow(select/coreid/add_imm), hint)
        return {inst[1]}


def get_srcs(slot: tuple) -> Set[int]:
    engine, inst = slot
    op = inst[0]

    if engine == "alu":  # op, dest, a1, a2
        return set(inst[2:])
    if engine == "load":
        if op == "const":
            return set()
        return {inst[2]}  # (v)load, dest, addr -> addr is scalar
    if engine == "store":
        if op == "store":  # store, addr, src
            return {inst[1], inst[2]}
        if op == "vstore":  # vstore, addr, src
            return {inst[1]} | set(range(inst[2], inst[2] + VLEN))
    if engine == "valu":
        if op == "vbroadcast":  # vbroadcast, dest, src
            return {inst[2]}
        # Other VALU ops: op, dest, src1, src2...
        srcs = set()
        for i in range(2, len(inst)):
            srcs.update(range(inst[i], inst[i] + VLEN))
        return srcs
    if engine == "flow":
        if op in ["select"]:  # select, dest, cond, a, b
            return set(inst[2:])
        if op == "vselect":  # vselect, dest, cond, a, b
            srcs = set()
            for i in range(2, len(inst)):
                srcs.update(range(inst[i], inst[i] + VLEN))
            return srcs
        if op in ["cond_jump", "cond_jump_rel", "jump_indirect", "trace_write"]:
            return {inst[1]}
        if op == "add_imm":  # add_imm, dest, a, imm
            return {inst[2]}
        return set()  # halt, pause, jump, coreid
    if engine == "debug":
        return set()

    raise AssertionError(f"get_srcs: Unknown slot {slot}")


class Scheduler:
    def __init__(self, builder: "KernelBuilder", slots: list[tuple[Engine, tuple]], use_debug: bool) -> None:
        self.builder = builder
        self.slots = slots if use_debug else [s for s in slots if s[0] != "debug"]

        # map of instruction slot to dependencies before slot can run
        # also have reverse
        self.dependency_map: List[Set[int]] = [set() for i in range(len(self.slots))]
        self.free_map: List[Set[int]] = [set() for i in range(len(self.slots))]
        self.calculate_dependencies()

    # add a dependency that instruction index ai --> bi
    def add_dependence(self, ai, bi):
        assert isinstance(ai, int)
        assert isinstance(bi, int)
        self.dependency_map[ai].add(bi)
        self.free_map[bi].add(ai)

    def calculate_dependencies(self):
        # map of addr --> last slot where it was written
        last_write: Dict[int, int] = {}

        # map of resource --> when it was last read in current state
        reads: Dict[int, Set[int]] = defaultdict(set)
        for i, slot in enumerate(self.slots):
            engine = slot[0]

            # debug must always be run in-pr
            if engine == "debug":
                for j in range(i):
                    self.add_dependence(i, j)
                continue

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

            bundle: Dict[Engine, List[Tuple]] = {
                "valu": [],
                "alu": [],
                "load": [],
                "store": [],
                "flow": [],
                "debug": [],
            }
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

            return {k: [vv[1] for vv in v] for k, v in bundle.items() if len(v) > 0}

        for i, cnt in enumerate(dep_count):
            if cnt == 0:
                add_slot(i)

        answer = []
        while num_scheduled_slots < len(self.slots):
            bundle = schedule_greedy()
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

            # We want to prioritize instructions on the critical path.
            # The priority is the path length (more negative is more critical).
            # The instruction index `i` is used as a tie-breaker.
            ready_slots[engine].put((priority, i))

        def free_dep(i):
            dep_count[i] -= 1
            assert dep_count[i] >= 0
            if dep_count[i] == 0:
                add_slot(i)

        def schedule() -> Dict[Engine, List[Tuple]]:
            nonlocal num_scheduled_slots

            bundle: Dict[Engine, List[Tuple]] = {
                "valu": [],
                "alu": [],
                "load": [],
                "store": [],
                "flow": [],
                "debug": [],
            }
            for engine, limit in SLOT_LIMITS.items():
                while not ready_slots[engine].empty() and len(bundle[engine]) < limit:
                    _priority, i = ready_slots[engine].get()
                    bundle[engine].append(i)

            # canonicalize things
            for k, v in bundle.items():
                for e, i in enumerate(v):
                    for j in self.free_map[i]:
                        free_dep(j)
                    v[e] = self.slots[i]
                    num_scheduled_slots += 1

            return {k: [vv[1] for vv in v] for k, v in bundle.items() if len(v) > 0}

        for i, cnt in enumerate(dep_count):
            if cnt == 0:
                add_slot(i)

        answer = []
        while num_scheduled_slots < len(self.slots):
            bundle = schedule()
            answer.append(bundle)
        return answer

    def schedule_by_index(self) -> List[Dict[Engine, List[Tuple]]]:
        """Schedules using a priority queue, but prioritizes by original instruction index.
        This reproduces the old buggy behavior for comparison."""
        ready_slots: Dict[Engine, PriorityQueue] = {
            "valu": PriorityQueue(),
            "alu": PriorityQueue(),
            "load": PriorityQueue(),
            "store": PriorityQueue(),
            "flow": PriorityQueue(),
            "debug": PriorityQueue(),
        }

        num_scheduled_slots = 0
        dep_count: List[int] = [len(self.dependency_map[i]) for i in range(len(self.slots))]

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

            # The old buggy behavior: prioritize by instruction index `i`
            ready_slots[engine].put((i, priority))

        def free_dep(i):
            dep_count[i] -= 1
            assert dep_count[i] >= 0
            if dep_count[i] == 0:
                add_slot(i)

        def schedule() -> Dict[Engine, List[Tuple]]:
            nonlocal num_scheduled_slots

            bundle: Dict[Engine, List[Tuple]] = {k: [] for k in SLOT_LIMITS.keys()}
            for engine, limit in SLOT_LIMITS.items():
                while not ready_slots[engine].empty() and len(bundle[engine]) < limit:
                    i, _priority = ready_slots[engine].get()
                    bundle[engine].append(i)

            for k, v in bundle.items():
                for e, i in enumerate(v):
                    for j in self.free_map[i]:
                        free_dep(j)
                    v[e] = self.slots[i]
                    num_scheduled_slots += 1

            return {k: [vv[1] for vv in v] for k, v in bundle.items() if len(v) > 0}

        for i, cnt in enumerate(dep_count):
            if cnt == 0:
                add_slot(i)

        answer = []
        while num_scheduled_slots < len(self.slots):
            bundle = schedule()
            if any(bundle.values()):
                answer.append(bundle)
        return answer

"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

from scheduler import Scheduler
from config import SCALAR_SCHEDULE, SCHEDULER, USE_OPTIMIZED_HASH, USE_SIMPLE_WORKLOAD


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.const_map_vector = {}
        self.const_map_scalar = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]]):
        # Simple slot packing that just uses one slot per instruction bundle
        match SCHEDULER:
            case "greedy":
                return Scheduler(self, slots).schedule_greedy()
            case "critical":
                return Scheduler(self, slots).schedule_critical_path()
            case "none":
                instrs = []
                for engine, slot in slots:
                    instrs.append({engine: [slot]})
                return instrs
            case _:
                raise NotImplementedError(f"Unknown scheduler {SCHEDULER}")

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1, cached=True):
        addr = self.scratch_ptr
        if name is not None:
            if cached and name in self.scratch:
                return self.scratch[name]
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
            for i in range(1, length):
                self.scratch_debug[addr + i] = (f"{name}_offset_{i}", 1)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, body):
        if val not in self.const_map:
            addr = self.alloc_scratch(f"const_{val}")
            body.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_const_vector(self, val, body):
        if val in self.const_map_vector:
            return self.const_map_vector[val]
        scalar = self.scratch_const(val, body)
        scratch = self.alloc_scratch(f"const_vec_{val}", length=VLEN)
        body.add("valu", ("vbroadcast", scratch, scalar))
        return scratch

    def scratch_const_vector_from_scalar(self, scalar, body):
        # Takes in addresses in scratch mem
        if scalar in self.const_map_scalar:
            return self.const_map_vector[scalar]
        scratch = self.alloc_scratch(f"const_vec_from_scalar_addr_{scalar}", length=VLEN)
        body.add("valu", ("vbroadcast", scratch, scalar))
        return scratch

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i, body):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1, body))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3, body))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(
                (
                    "debug",
                    ("compare", val_hash_addr, (round, i, "hash_stage", hi)),
                )
            )

        return slots

    def build_hash_vectorized(self, body: "Appender", values_v, extra_slot):
        for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # body.append("debug", (f"print_scratch_v", f"before {i}", values_v))
            if op3 == "<<" and op1 == "+" and op2 == "+" and USE_OPTIMIZED_HASH:
                body.append(
                    "valu",
                    (
                        "multiply_add",
                        values_v,
                        values_v,
                        self.scratch_const_vector(2**val3 + 1, body),
                        self.scratch_const_vector(val1, body),
                    ),
                )
            else:
                body.append("valu", (op3, extra_slot, values_v, self.scratch_const_vector(val3, body)))
                body.append("valu", (op1, values_v, values_v, self.scratch_const_vector(val1, body)))
                body.append("valu", (op2, values_v, values_v, extra_slot))
            # body.append("debug", (f"print_scratch_v", f"after {i}", values_v))

    def build_hash_scalar(self, body: "Appender", values, extra_slot):
        for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # body.append("debug", (f"print_scratch_v", f"before {i}", values_v))
            if op3 == "<<" and op1 == "+" and op2 == "+" and USE_OPTIMIZED_HASH:
                body.append("alu", ("*", values, values, self.scratch_const(2**val3 + 1, body)))
                body.append("alu", ("+", values, values, self.scratch_const(val1, body)))
            else:
                body.append("alu", (op3, extra_slot, values, self.scratch_const(val3, body)))
                body.append("alu", (op1, values, values, self.scratch_const(val1, body)))
                body.append("alu", (op2, values, values, extra_slot))
            # body.append("debug", (f"print_scratch_v", f"after {i}", values_v))

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        # tmp1 = self.alloc_scratch("tmp1")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            tmp = self.alloc_scratch(f"tmp{i}")
            self.add("load", ("const", tmp, i))
            self.add("load", ("load", self.scratch[v], tmp))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        class Appender:
            def __init__(self) -> None:
                self.body = []  # array of slots, we do scheduling later

            def append(self, engine, slot):
                self.body.append((engine, slot))

            def add(self, engine, slot):
                self.body.append((engine, slot))

            def get(self):
                return self.body

        body = Appender()
        assert batch_size % VLEN == 0

        cur_levels = [0] * (batch_size // VLEN)
        forest_values_p_v = self.scratch_const_vector_from_scalar(self.scratch["forest_values_p"], body)
        for round in range(rounds):
            body.append("debug", ("print", f"=============ROUND {round}=============="))
            for i in range(batch_size // VLEN):

                def load_tree(use_vselect: bool, forest_v, forest_addr_v):
                    if use_vselect:
                        if round % (forest_height + 1) == 0:
                            addr = self.scratch["forest_values_p"]
                            body.append("load", ("vload", forest_v, addr))
                        else:
                            raise NotImplementedError("Error")
                    else:
                        # Load data from nodes
                        for load_i in range(VLEN):
                            body.append("hint", ("join_dst", forest_addr_v + load_i, forest_addr_v))
                            body.append("load", ("load", forest_v + load_i, forest_addr_v + load_i))

                        join_dst = ["join_dst", forest_v]
                        for load_i in range(VLEN):
                            join_dst.append(forest_v + load_i)
                        body.append("hint", tuple(join_dst))

                def schedule_loop_vector(use_vselect: bool = False):
                    addr_indices = self.alloc_scratch(f"addr_indices_batch_{i}")
                    addr_values = self.alloc_scratch(f"addr_values_batch_{i}")

                    # Load indices
                    indices_v = self.alloc_scratch(f"indices_batch_{i}_v", VLEN)

                    if round == 0:  # prologue
                        body.append(
                            "alu", ("+", addr_indices, self.scratch["inp_indices_p"], self.scratch_const(i * 8, body))
                        )
                        body.append("load", ("vload", indices_v, addr_indices))

                    # Load values
                    body.append(
                        "alu",
                        (
                            "+",
                            addr_values,
                            self.scratch["inp_values_p"],
                            self.scratch_const(i * 8, body),
                        ),
                    )
                    values_v = self.alloc_scratch(f"values_batch_{i}_v", VLEN)

                    if round == 0:  # prologue
                        body.append("load", ("vload", values_v, addr_values))

                    # calculates loads
                    forest_addr_v = self.alloc_scratch(f"addr_forest_batch_{i}_v", VLEN)
                    body.append("valu", ("+", forest_addr_v, forest_values_p_v, indices_v))
                    forest_v = self.alloc_scratch(f"forest_batch_{i}_v", VLEN)

                    load_tree(use_vselect, forest_v, forest_addr_v)

                    # forest_v --> the bintree values
                    # values_v --> the values in our array
                    # indices_v --> the indices in our array
                    # body.append("debug", ("print", f"before work batch {i}"))
                    # body.append("debug", ("print_scratch_v", "values_v", values_v))
                    # body.append("debug", ("print_scratch_v", "forest_v", forest_v))
                    # body.append("debug", ("print_scratch_v", "indices_v", indices_v))

                    body.append("valu", ("^", values_v, values_v, forest_v))

                    # body.append("debug", ("print", f"before hash batch {i}"))
                    # body.append("debug", ("print_scratch_v", "values_v", values_v))
                    # body.append("debug", ("print_scratch_v", "forest_v", forest_v))
                    # body.append("debug", ("print_scratch_v", "indices_v", indices_v))

                    # At end, values_v has the correct values
                    self.build_hash_vectorized(body, values_v, forest_v)

                    # body.append("debug", ("print", f"after hash batch {i}"))
                    # body.append("debug", ("print_scratch_v", "values_v", values_v))
                    # body.append("debug", ("print_scratch_v", "forest_v", forest_v))
                    # body.append("debug", ("print_scratch_v", "indices_v", indices_v))

                    modulo = forest_addr_v
                    body.append(
                        "valu",
                        ("%", modulo, values_v, self.scratch_const_vector(2, body)),
                    )
                    body.append("valu", ("+", modulo, modulo, self.scratch_const_vector(1, body)))

                    # Update the indices
                    cur_levels[i] += 1
                    if cur_levels[i] <= forest_height:
                        body.append(
                            "valu", ("multiply_add", indices_v, indices_v, self.scratch_const_vector(2, body), modulo)
                        )
                    else:
                        body.append("valu", ("^", indices_v, indices_v, indices_v))
                        cur_levels[i] = 0

                    # body.append("debug", ("print", f"after indices batch {i}"))
                    # body.append("debug", ("print_scratch_v", "values_v", values_v))
                    # body.append("debug", ("print_scratch_v", "forest_v", forest_v))
                    # body.append("debug", ("print_scratch_v", "indices_v", indices_v))

                    # Write batch back to memory
                    if round == rounds - 1:
                        body.append("store", ("vstore", addr_indices, indices_v))
                        body.append("store", ("vstore", addr_values, values_v))

                def schedule_loop_scalar(use_vselect: bool = False):
                    addr_indices = self.alloc_scratch(f"addr_indices_batch_{i}")
                    addr_values = self.alloc_scratch(f"addr_values_batch_{i}")

                    # Load indices
                    indices_v = self.alloc_scratch(f"indices_batch_{i}_v", VLEN)

                    if round == 0:  # prologue
                        body.append(
                            "alu", ("+", addr_indices, self.scratch["inp_indices_p"], self.scratch_const(i * 8, body))
                        )
                        body.append("load", ("vload", indices_v, addr_indices))
                        for vi in range(VLEN):
                            body.append("hint", ("join_dst", indices_v + vi, indices_v))

                    # Load values
                    body.append(
                        "alu",
                        (
                            "+",
                            addr_values,
                            self.scratch["inp_values_p"],
                            self.scratch_const(i * 8, body),
                        ),
                    )
                    values_v = self.alloc_scratch(f"values_batch_{i}_v", VLEN)

                    if round == 0:  # prologue
                        body.append("load", ("vload", values_v, addr_values))
                        for vi in range(VLEN):
                            body.append("hint", ("join_dst", values_v + vi, values_v))

                    # calculates loads
                    forest_addr_v = self.alloc_scratch(f"addr_forest_batch_{i}_v", VLEN)
                    for vi in range(VLEN):
                        body.append("alu", ("+", forest_addr_v + vi, forest_values_p_v + vi, indices_v + vi))
                    forest_v = self.alloc_scratch(f"forest_batch_{i}_v", VLEN)

                    # Load data from nodes
                    load_tree(use_vselect, forest_v, forest_addr_v)

                    modulo = forest_addr_v
                    for vi in range(VLEN):
                        body.append("alu", ("^", values_v + vi, values_v + vi, forest_v + vi))
                        self.build_hash_scalar(body, values_v + vi, forest_v + vi)

                        body.append(
                            "alu",
                            ("%", modulo + vi, values_v + vi, self.scratch_const(2, body)),
                        )
                        body.append("alu", ("+", modulo + vi, modulo + vi, self.scratch_const(1, body)))

                    cur_levels[i] += 1

                    if cur_levels[i] <= forest_height:
                        for vi in range(VLEN):
                            body.append("alu", ("*", indices_v + vi, indices_v + vi, self.scratch_const(2, body)))
                            body.append("alu", ("+", indices_v + vi, indices_v + vi, modulo + vi))
                    else:
                        for vi in range(VLEN):
                            body.append("alu", ("^", indices_v + vi, indices_v + vi, indices_v + vi))
                        cur_levels[i] = 0

                    hints_indices_v = ["join_dst", indices_v]
                    hints_values_v = ["join_dst", values_v]
                    for vi in range(VLEN):
                        hints_indices_v.append(indices_v + vi)
                        hints_values_v.append(values_v + vi)
                    body.append("hint", tuple(hints_indices_v))
                    body.append("hint", tuple(hints_values_v))

                    # Write batch back to memory
                    if round == rounds - 1:
                        body.append("store", ("vstore", addr_indices, indices_v))
                        body.append("store", ("vstore", addr_values, values_v))

                # breakpoint()
                if i in SCALAR_SCHEDULE:
                    schedule_loop_scalar()
                else:
                    schedule_loop_vector()

        body_instrs = self.build(body.get())
        print("TOTAL SCRATCH SPACE:", self.scratch_ptr)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        if (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            != ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ):
            print("actual:", machine.mem)
            print("ref   :", ref_mem)
            raise AssertionError(f"Incorrect result on round {i}")
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        if USE_SIMPLE_WORKLOAD:
            do_kernel_test(0, 1, 8, trace=True, prints=False)
        else:
            do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        # do_kernel_test(10, 16, 256)
        # do_kernel_test(1, 1, 16, trace=True, prints=False)
        pass


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()

"""
Things i did:
1. basic vectorized version
2. simple scheduler
3. make scheduler better by adding hints to deal with certain instructions (for ordering)

Things i need to do:
4. non-greedy scheduling 
5. optimization of top of tree using vselect
6. alternative alu version scalar
7. alternative alu version using control flow
8. other misc. optimization
"""

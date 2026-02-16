# batch ids that should go ahead and be scheduled!
SCALAR_SCHEDULE = [0, 4, 7, 11, 15, 20, 24, 28, 31]

# "critical": uses critical path heuristic
# "greedy": simple greedy scheduler
# "none": don't do vliw bundle creation
SCHEDULER = "critical"

# Use optimized hashes for certain steps
USE_OPTIMIZED_HASH = True

# Whether to run one round on a simple tree with batch size 8
USE_SIMPLE_WORKLOAD = False

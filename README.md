# hydropower-mpc-stress-testing
The closed-loop dispatch script (`dispatch_AE.py`) depends on an external module named `sphdrostation.py`, which implements the hydropower station simulator. This file is not included in the repository because it contains **confidential station-specific hydraulic characteristic curves and operational parameters**, and therefore cannot be publicly released.


The public repository provides the complete workflow for:
- data preprocessing,
- model training,
- residual generation,
- dispatch optimization,
- and closed-loop evaluation,


but the plant-specific simulator must be supplied locally by authorized users or replaced by a compatible mock simulator with the same interface.

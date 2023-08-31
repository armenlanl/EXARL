Notes:

Debug mode is added in the const env and will drastically slow down run
Provides actor learning rate, critics learning rates in file with same name as normal with _agent_ appended to end
See logging function for more information
Analysis files input the logging function from agent as well as logging file from env
- Same format with more data introduced

Attached below are the run commands utilized for testing and running

GraphTD3-v2 and ExaExaaltGraph-v2 correspond to unconstrained task list complete
GraphTD3-v3 and ExaExaaltGraph-v3 correspond to the Unconstrained no database implmentations
GraphTD3-v4 and ExaExaaltGraph-v4 correspond to the Constrained implmentations

python exarl/driver --workflow sync --model_type AC --agent GraphTD3-v4 --env ExaExaaltGraph-v4 --n_episodes 100 --n_steps 100 --experiment_id Exaalt

python exarl/driver --workflow sync --model_type AC --agent TD3-v2 --env ExaPendulum-v1 --n_episodes 50 --n_steps 200 --environment_id pend

What to check before running:

AC.json
Learning rates

Agents:
Hozion value is applicable
Update frequency
Noise
Debug mode (True = On)
Architecture of networks
- Use of CNNs
- Kernel size

Environment:
State Depth
Node count and Number of states (need to be equal)
Number of workers
Initial state selection
Action space low/high
Reward function

Make sure in the reset the values are the same as instantiation
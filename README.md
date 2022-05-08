# 2AMC15
Project that implements Policy Iteration and Value Iteration to guide the moves of a vaccum robot-cleaner

## How to run
Required version of python is 3.6+

1. Install dependencies ```pip install -r requirements.txt #pip3 if python3 is installed```
2. Go to ```/Discrete-Simulations/```
for GUI webapp:
3. Run ```python app.py #optionally python3``` 

for testing
3. Run ```python headless.py #optionally python3``` 

### Settings
To test different parameters (and recreate experiments) the following settings can be changed.

The boards used in the experiments can be found in grid_configs as: example-random-house-grid-0.grid, example-random-level.grid, Smaller_empty.grid

The policy iteration-based robot can be found in policy_iter.py and value iteration-based in value_iter.py

#### Parameter settings
policy_iter.py
* Discount value: gamma (line 14)
* Policy Evaluation max iteration: max_iter (line 17)
* Policy Evaluation threshold: if np.max(abs(values1-values2)) < .0001 (line 62)
* Max steps: for step in range(3000): (line 209)

value_iter.py
* Discount value: gamma (line 14)
* Policy Evaluation max iteration: max_iter (line 74)
* Policy Evaluation threshold: if np.max(abs(values1-values2)) < .0001 (line 95)


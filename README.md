# controls_challenge

Hi! This is my submission to the comma.ai controls challenge.

**Final Score: 41.71 on test, 39.94 on train**

## Approach: A 20-Parameter Feedback Controller Evolved with CMA-ES

### Starting Point

The provided PID controller scores about 80.72 on 100 segments. It's purely reactive — it only responds to the current error between target and actual lateral acceleration. I knew I could do better by using the future plan data the simulator provides.

### Failed Attempts

Before landing on the final approach, I tried a few things that didn't work:

- **MPC with the ONNX model**: I tried loading the physics model inside the controller and evaluating candidate actions by simulating forward. This never worked because the simulator uses stochastic sampling (temperature 0.8) while my internal model used expected values, causing the predictions to diverge from reality. Scores ranged from 116 to 105,000. Completely unusable.
- **Simple Nelder-Mead PID tuning**: Optimizing just 5 PID parameters with scipy got me to about 73. Better than baseline but nowhere near competitive.

### The Final Solution

The winning approach was designing a rich controller structure with 20 tunable parameters, then letting CMA-ES (Covariance Matrix Adaptation Evolution Strategy) find the optimal values.

The controller goes well beyond basic PID. It includes:

- **PID with integral decay** — standard proportional, integral, and derivative terms, plus a second derivative term for anticipating error acceleration. The integral term has a per-step decay factor to prevent windup.
- **Multi-horizon future plan feedforward** — instead of just reacting to the current error, the controller looks ahead at the planned trajectory at 5 different time horizons (1, 5, 10, 20, and 50 steps ahead). Each horizon has its own learned gain.
- **Target rate-of-change feedforward** — separate gains for how fast the target is changing at 1-step and 5-step horizons.
- **Road roll compensation** — a direct feedforward term from the road's roll-induced lateral acceleration.
- **Velocity-adaptive gains** — the error gain and overall action scale both adjust based on vehicle speed.
- **Nonlinear error correction** — a cubic error term for more aggressive correction on large errors, plus a separate linear boost that kicks in when the error exceeds 0.5.
- **Action smoothing** — an exponential moving average on the output action to reduce jerk.

### Optimization

I ran CMA-ES with a population of 20 candidates, evaluating each on 100 data segments. Each generation takes about 10 minutes. The optimization ran for 164 generations total (~27 hours). The convergence was pretty smooth:

| Generation | Best Score |
|---|---|
| 1 | 60.52 |
| 9 | 50.45 |
| 22 | 42.73 |
| 50 | 41.73 |
| 89 | 40.36 |
| 139 | 39.94 |

The optimizer plateaued around 40 after generation ~89. The final validation on 100 segments came out to 41.71, slightly higher than the training score due to the usual generalization gap.

### What I Learned

The biggest insight was how much the future plan data matters. The basic PID controller completely ignores it. My controller's strongest parameters ended up being the future feedforward gains — the 5-step average feedforward and the road roll compensation had the largest magnitudes after optimization. The P and I gains actually shrank compared to the baseline PID, suggesting the controller shifted from reactive correction to predictive anticipation.

Getting below 40 with a pure feedback controller seems to be near the theoretical limit. The top scores on the leaderboard use model-predictive control or reinforcement learning which can plan multiple steps ahead using the actual physics model — my controller can't do that, it only uses hand-designed features of the future.

---


<div align="center">
<h1>comma Controls Challenge v2</h1>


<h3>
  <a href="https://comma.ai/leaderboard">Leaderboard</a>
  <span> · </span>
  <a href="https://comma.ai/jobs">comma.ai/jobs</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
  <span> · </span>
  <a href="https://x.com/comma_ai">X</a>
</h3>

</div>

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.

## Getting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual car and road states from [openpilot](https://github.com/commaai/openpilot) users.

```
# install required packages
# recommended python==3.11
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid
```

There are some other scripts to help you get aggregate metrics:
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. Its inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`), and a steer input (`steer_action`), then it predicts the resultant lateral acceleration of the car.

## Controllers
Your controller should implement a new [controller](https://github.com/commaai/controls_challenge/tree/master/controllers). This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.

## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(\mathrm{actual{\textunderscore}lat{\textunderscore}accel} - \mathrm{target{\textunderscore}lat{\textunderscore}accel})^2}{\text{steps}} * 100$
- `jerk_cost`: $\dfrac{(\Sigma( \mathrm{actual{\textunderscore}lat{\textunderscore}accel_t} - \mathrm{actual{\textunderscore}lat{\textunderscore}accel_{t-1}}) / \Delta \mathrm{t} )^{2}}{\text{steps} - 1} * 100$

It is important to minimize both costs. `total_cost`: $(\mathrm{lat{\textunderscore}accel{\textunderscore}cost} * 50) + \mathrm{jerk{\textunderscore}cost}$

## Submission
Run the following command, then submit `report.html` and your code to [this form](https://forms.gle/US88Hg7UR6bBuW3BA).

Competitive scores (`total_cost<100`) will be added to the leaderboard

```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller pid
```

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma

Like this sort of stuff? You might want to work at comma!
[comma.ai/jobs](https://comma.ai/jobs)

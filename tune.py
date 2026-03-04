import numpy as np
import cma
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

MODEL_PATH = "./models/tinyphysics.onnx"
DATA_PATH = Path("./data")
NUM_SEGS = 100
files = sorted(DATA_PATH.iterdir())[:NUM_SEGS]


class InlineController:
    def __init__(self, params):
        self._p = list(params)
        self._error_integral = 0.0
        self._prev_error = 0.0
        self._prev_prev_error = 0.0
        self._prev_action = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        p = self._p
        error = target_lataccel - current_lataccel
        decay = np.clip(1.0 - abs(p[19]), 0.9, 1.0)
        self._error_integral = self._error_integral * decay + error
        self._error_integral = np.clip(self._error_integral, -abs(p[3]), abs(p[3]))
        error_deriv = error - self._prev_error
        error_deriv2 = error_deriv - (self._prev_error - self._prev_prev_error)
        self._prev_prev_error = self._prev_error
        self._prev_error = error

        action = p[0] * error + p[1] * self._error_integral + p[2] * error_deriv + p[4] * error_deriv2
        action += p[5] * target_lataccel

        if future_plan is not None and len(future_plan.lataccel) > 0:
            fl = future_plan.lataccel
            if len(fl) >= 1:
                action += p[6] * (fl[0] - current_lataccel)
            if len(fl) >= 5:
                action += p[7] * (np.mean(fl[:5]) - current_lataccel)
            if len(fl) >= 10:
                action += p[8] * (np.mean(fl[5:10]) - current_lataccel)
            if len(fl) >= 20:
                action += p[9] * (np.mean(fl[10:20]) - current_lataccel)
            if len(fl) >= 50:
                action += p[10] * (np.mean(fl[20:50]) - current_lataccel)
            if len(fl) >= 1:
                action += p[11] * (fl[0] - target_lataccel)
            if len(fl) >= 5:
                action += p[12] * (np.mean(fl[:5]) - target_lataccel)

        action += p[13] * state.roll_lataccel
        action += p[14] * (state.v_ego / 30.0 - 1.0) * error
        action *= (1.0 + p[15] * (state.v_ego / 30.0 - 1.0))
        action += p[16] * error ** 3
        if abs(error) > 0.5:
            action += p[17] * error
        action = action * (1.0 - abs(p[18])) + self._prev_action * abs(p[18])
        self._prev_action = action
        return action


def eval_file(args):
    data_file, params = args
    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    ctrl = InlineController(params)
    sim = TinyPhysicsSimulator(model, str(data_file), controller=ctrl, debug=False)
    return sim.rollout()['total_cost']


def objective(params):
    args = [(f, params) for f in files]
    with ThreadPoolExecutor(max_workers=8) as executor:
        costs = list(executor.map(eval_file, args))
    return np.mean(costs)


x0 = [0.15114, 0.13662, -0.00993, 4.94185, -0.00948, 0.08555, 0.06345, 0.00553, -0.01914, -0.06787, -0.01662, -0.00127, 0.149, -0.01298, 0.00059, -0.01987, -0.03218, 0.00719, 0.03303, 0.00251]

opts = {
    'maxiter': 300,
    'popsize': 20,
    'seed': 123,
    'tolfun': 0.01,
    'CMA_diagonal': True,
}

print(f"CMA-ES: {len(x0)} params, {NUM_SEGS} segs, pop={opts['popsize']}")
es = cma.CMAEvolutionStrategy(x0, 0.02, opts)

gen = 0
while not es.stop():
    solutions = es.ask()
    fitnesses = [objective(s) for s in solutions]
    es.tell(solutions, fitnesses)
    gen += 1
    print(f"Gen {gen:3d} | best_gen={min(fitnesses):.2f} | best_ever={es.result.fbest:.2f}")
    if gen % 10 == 0:
        print(f"  Params: {list(np.round(es.result.xbest, 5))}")

best = es.result.xbest
print(f"\nFinal best cost: {es.result.fbest:.3f}")
names = ['p_gain', 'i_gain', 'd_gain', 'integral_limit', 'd2_gain',
         'ff_target', 'ff_future_1', 'ff_future_5', 'ff_future_10',
         'ff_future_20', 'ff_future_50', 'ff_rate_1', 'ff_rate_5',
         'roll_comp', 'vel_err_scale', 'vel_act_scale',
         'error_cubic', 'large_err_boost', 'action_smooth', 'integral_decay']
for name, val in zip(names, best):
    print(f"  {name:20s}: {val:.6f}")

param_list = [round(float(v), 8) for v in best]
code = f'''from . import BaseController
import numpy as np

DEFAULT_PARAMS = {param_list}

class Controller(BaseController):
    def __init__(self, params=None):
        self._p = params if params is not None else list(DEFAULT_PARAMS)
        self._error_integral = 0.0
        self._prev_error = 0.0
        self._prev_prev_error = 0.0
        self._prev_action = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        p = self._p
        error = target_lataccel - current_lataccel
        decay = np.clip(1.0 - abs(p[19]), 0.9, 1.0)
        self._error_integral = self._error_integral * decay + error
        self._error_integral = np.clip(self._error_integral, -abs(p[3]), abs(p[3]))
        error_deriv = error - self._prev_error
        error_deriv2 = error_deriv - (self._prev_error - self._prev_prev_error)
        self._prev_prev_error = self._prev_error
        self._prev_error = error

        action = p[0] * error + p[1] * self._error_integral + p[2] * error_deriv + p[4] * error_deriv2
        action += p[5] * target_lataccel

        if future_plan is not None and len(future_plan.lataccel) > 0:
            fl = future_plan.lataccel
            if len(fl) >= 1:
                action += p[6] * (fl[0] - current_lataccel)
            if len(fl) >= 5:
                action += p[7] * (np.mean(fl[:5]) - current_lataccel)
            if len(fl) >= 10:
                action += p[8] * (np.mean(fl[5:10]) - current_lataccel)
            if len(fl) >= 20:
                action += p[9] * (np.mean(fl[10:20]) - current_lataccel)
            if len(fl) >= 50:
                action += p[10] * (np.mean(fl[20:50]) - current_lataccel)
            if len(fl) >= 1:
                action += p[11] * (fl[0] - target_lataccel)
            if len(fl) >= 5:
                action += p[12] * (np.mean(fl[:5]) - target_lataccel)

        action += p[13] * state.roll_lataccel
        action += p[14] * (state.v_ego / 30.0 - 1.0) * error
        action *= (1.0 + p[15] * (state.v_ego / 30.0 - 1.0))
        action += p[16] * error ** 3
        if abs(error) > 0.5:
            action += p[17] * error
        action = action * (1.0 - abs(p[18])) + self._prev_action * abs(p[18])
        self._prev_action = action
        return action
'''
with open("controllers/advanced.py", "w") as f:
    f.write(code)
print("Best controller saved to controllers/advanced.py")

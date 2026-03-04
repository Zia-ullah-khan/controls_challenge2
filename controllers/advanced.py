from . import BaseController
import numpy as np

DEFAULT_PARAMS = [0.07725, 0.1294, 0.02505, 4.70456, -0.00834, 0.35337, 0.13495, -0.04235, 0.05979, -0.07058, 0.10604, -0.02491, 0.45488, -0.49372, 0.03593, -0.29039, 0.05573, -0.12085, 0.13979, -0.00819]


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

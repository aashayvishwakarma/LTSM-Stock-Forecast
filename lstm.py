"""
NumPy-only LSTM (single cell), linear readout, BPTT, and Adam updates.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    """d sigmoid(x) / dx."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_deriv(x: np.ndarray) -> np.ndarray:
    """d tanh(x) / dx."""
    t = np.tanh(x)
    return 1.0 - t ** 2


def clip_grad(g: np.ndarray, limit: float = 5.0) -> np.ndarray:
    return np.clip(g, -limit, limit)


class AdamState:
    """Per-parameter Adam buffers."""

    def __init__(self, shape: tuple[int, ...]):
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def step(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        clip: float = 5.0,
    ) -> None:
        g = clip_grad(np.asarray(grad, dtype=np.float64), clip)
        self.t += 1
        self.m = beta1 * self.m + (1.0 - beta1) * g
        self.v = beta2 * self.v + (1.0 - beta2) * (g ** 2)
        m_hat = self.m / (1.0 - beta1 ** self.t)
        v_hat = self.v / (1.0 - beta2 ** self.t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


class LSTMCell:
    """
    One LSTM layer with concatenated weights:
      z_t = [h_{t-1}; x_t]  shape (H+I, 1)
      a_t = W z_t + b       shape (4H, 1)
    Gates: input/forget/output on sigmoid; candidate on tanh.
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int | None = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.H = hidden_size
        self.I = input_size
        rng = np.random.default_rng(seed)
        fan_in = self.H + self.I
        fan_out = 4 * self.H
        limit = 0.5 * np.sqrt(6.0 / (fan_in + fan_out))
        self.W = rng.uniform(-limit, limit, size=(4 * self.H, fan_in)).astype(np.float64)
        self.b = np.zeros((4 * self.H, 1), dtype=np.float64)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.adam_W = AdamState(self.W.shape)
        self.adam_b = AdamState(self.b.shape)

    def zero_grad(self) -> None:
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """x: (I,1), h_prev,c_prev: (H,1). Returns h, c, cache."""
        z = np.vstack([h_prev, x])
        a = np.dot(self.W, z) + self.b
        H = self.H
        ai = a[0:H]
        af = a[H : 2 * H]
        ag = a[2 * H : 3 * H]
        ao = a[3 * H : 4 * H]

        i_gate = sigmoid(ai)
        f_gate = sigmoid(af)
        c_tilde = np.tanh(ag)
        o_gate = sigmoid(ao)

        c = f_gate * c_prev + i_gate * c_tilde
        tanh_c = np.tanh(c)
        h = o_gate * tanh_c

        cache = {
            "z": z,
            "a": a,
            "ai": ai,
            "af": af,
            "ag": ag,
            "ao": ao,
            "i_gate": i_gate,
            "f_gate": f_gate,
            "c_tilde": c_tilde,
            "o_gate": o_gate,
            "c": c,
            "tanh_c": tanh_c,
            "c_prev": c_prev,
            "h_prev": h_prev,
            "x": x,
        }
        return h, c, cache

    def backward(
        self,
        cache: dict,
        dh_up: np.ndarray,
        dc_up: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        dh_up: dL/dh_t (total from above time step + loss path)
        dc_up: dL/dc_{t+1} carried backward through time
        """
        H = self.H
        i_gate = cache["i_gate"]
        f_gate = cache["f_gate"]
        c_tilde = cache["c_tilde"]
        o_gate = cache["o_gate"]
        c = cache["c"]
        tanh_c = cache["tanh_c"]
        c_prev = cache["c_prev"]
        z = cache["z"]
        ai = cache["ai"]
        af = cache["af"]
        ag = cache["ag"]
        ao = cache["ao"]

        # h = o * tanh(c)
        do = dh_up * tanh_c
        d_tanh_c = dh_up * o_gate
        dc = dc_up + d_tanh_c * tanh_deriv(c)

        # c = f * c_prev + i * c_tilde
        df = dc * c_prev
        di = dc * c_tilde
        dc_tilde = dc * i_gate
        dc_prev = dc * f_gate

        dao = do * sigmoid_deriv(ao)
        dai = di * sigmoid_deriv(ai)
        daf = df * sigmoid_deriv(af)
        dag = dc_tilde * tanh_deriv(ag)

        da = np.vstack([dai, daf, dag, dao])

        self.dW += np.dot(da, z.T)
        self.db += da

        dz = np.dot(self.W.T, da)
        dh_prev = dz[0:H]

        return dh_prev, dc_prev

    def update(self, lr: float) -> None:
        self.adam_W.step(self.W, self.dW, lr)
        self.adam_b.step(self.b, self.db, lr)


class LinearLayer:
    """y = W @ x + b with W: (out, in), x: (in, 1)."""

    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        limit = 0.5 * np.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-limit, limit, size=(out_features, in_features)).astype(np.float64)
        self.b = np.zeros((out_features, 1), dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.adam_W = AdamState(self.W.shape)
        self.adam_b = AdamState(self.b.shape)

    def zero_grad(self) -> None:
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        y = np.dot(self.W, x) + self.b
        cache = {"x": x}
        return y, cache

    def backward(self, cache: dict, dy: np.ndarray) -> np.ndarray:
        x = cache["x"]
        self.dW += np.dot(dy, x.T)
        self.db += dy
        dx = np.dot(self.W.T, dy)
        return dx

    def update(self, lr: float) -> None:
        self.adam_W.step(self.W, self.dW, lr)
        self.adam_b.step(self.b, self.db, lr)


class LSTMModel:
    """
    Sequence of inputs x_t each (input_size, 1).
    Final h_T mapped with a linear layer to `horizon` steps (multi-step forecast).
    """

    def __init__(self, input_size: int, hidden_size: int, horizon: int, seed: int | None = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.cell = LSTMCell(input_size, hidden_size, seed=seed)
        self.head = LinearLayer(hidden_size, horizon, seed=seed)

    def zero_grad(self) -> None:
        self.cell.zero_grad()
        self.head.zero_grad()

    def forward(self, inputs: list[np.ndarray]) -> tuple[np.ndarray, dict]:
        """
        inputs: length T, each array (input_size, 1).
        Returns (prediction (horizon, 1), caches bundle).
        """
        h = np.zeros((self.hidden_size, 1), dtype=np.float64)
        c = np.zeros((self.hidden_size, 1), dtype=np.float64)
        caches_seq: list[dict] = []
        for x in inputs:
            h, c, cache = self.cell.forward(np.asarray(x, dtype=np.float64), h, c)
            caches_seq.append(cache)

        y, lin_cache = self.head.forward(h)
        bundle = {"seq": caches_seq, "linear": lin_cache}
        return y, bundle

    def backward(self, pred: np.ndarray, target: np.ndarray, caches: dict) -> float:
        """
        Mean squared error loss and BPTT (gradients stored on cell and head).
        Call apply_updates(lr) after to apply Adam.
        """
        pred = np.asarray(pred, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        err = pred - target
        n = pred.size
        loss = float(np.mean(err ** 2))
        dy = (2.0 / n) * err

        self.zero_grad()
        dh = self.head.backward(caches["linear"], dy)

        seq = caches["seq"]
        dc = np.zeros((self.hidden_size, 1), dtype=np.float64)

        for t in reversed(range(len(seq))):
            cache_t = seq[t]
            dh, dc = self.cell.backward(cache_t, dh, dc)

        return loss

    def apply_updates(self, lr: float) -> None:
        self.cell.update(lr)
        self.head.update(lr)

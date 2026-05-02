import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_deriv(x):
    t = np.tanh(x)
    return 1.0 - t ** 2


class LSTMCell:
    def __init__(self, input_size, hidden_size, seed=None):
        rng = np.random.default_rng(seed)
        self.H = hidden_size
        self.I = input_size
        fan = self.H + self.I
        lim = 0.5 * np.sqrt(6.0 / (fan + 4 * self.H))
        self.W = rng.uniform(-lim, lim, size=(4 * self.H, fan)).astype(np.float64)
        self.b = np.zeros((4 * self.H, 1), dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._adam_w = [np.zeros_like(self.W), np.zeros_like(self.W), 0]
        self._adam_b = [np.zeros_like(self.b), np.zeros_like(self.b), 0]

    def zero_grad(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def forward(self, x, h_prev, c_prev):
        z = np.vstack([h_prev, x])
        a = np.dot(self.W, z) + self.b
        H = self.H
        ai, af, ag, ao = a[0:H], a[H : 2 * H], a[2 * H : 3 * H], a[3 * H : 4 * H]
        i_gate = sigmoid(ai)
        f_gate = sigmoid(af)
        c_tilde = np.tanh(ag)
        o_gate = sigmoid(ao)
        c = f_gate * c_prev + i_gate * c_tilde
        tanh_c = np.tanh(c)
        h = o_gate * tanh_c
        cache = {
            "z": z,
            "i_gate": i_gate,
            "f_gate": f_gate,
            "c_tilde": c_tilde,
            "o_gate": o_gate,
            "c": c,
            "tanh_c": tanh_c,
            "c_prev": c_prev,
            "ai": ai,
            "af": af,
            "ag": ag,
            "ao": ao,
        }
        return h, c, cache

    def backward(self, cache, dh_up, dc_up):
        H = self.H
        i_gate = cache["i_gate"]
        f_gate = cache["f_gate"]
        c_tilde = cache["c_tilde"]
        o_gate = cache["o_gate"]
        c = cache["c"]
        tanh_c = cache["tanh_c"]
        c_prev = cache["c_prev"]
        z = cache["z"]
        ai, af, ag, ao = cache["ai"], cache["af"], cache["ag"], cache["ao"]

        do = dh_up * tanh_c
        d_tanh_c = dh_up * o_gate
        dc = dc_up + d_tanh_c * tanh_deriv(c)
        df = dc * c_prev
        di = dc * c_tilde
        dc_tilde = dc * i_gate
        dc_prev = dc * f_gate

        da = np.vstack(
            [
                di * sigmoid_deriv(ai),
                df * sigmoid_deriv(af),
                dc_tilde * tanh_deriv(ag),
                do * sigmoid_deriv(ao),
            ]
        )
        self.dW += np.dot(da, z.T)
        self.db += da
        dz = np.dot(self.W.T, da)
        return dz[0:H], dc_prev

    def update(self, lr):
        for P, G, slot in (
            (self.W, self.dW, self._adam_w),
            (self.b, self.db, self._adam_b),
        ):
            g = np.clip(G, -5.0, 5.0)
            slot[2] += 1
            t = slot[2]
            slot[0][:] = 0.9 * slot[0] + 0.1 * g
            slot[1][:] = 0.999 * slot[1] + 0.001 * (g ** 2)
            m_hat = slot[0] / (1.0 - 0.9**t)
            v_hat = slot[1] / (1.0 - 0.999**t)
            P -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class LinearLayer:
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        lim = 0.5 * np.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-lim, lim, size=(out_features, in_features)).astype(np.float64)
        self.b = np.zeros((out_features, 1), dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._mw = np.zeros_like(self.W)
        self._vw = np.zeros_like(self.W)
        self._tw = 0
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)
        self._tb = 0

    def zero_grad(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def forward(self, x):
        y = np.dot(self.W, x) + self.b
        return y, {"x": x}

    def backward(self, cache, dy):
        x = cache["x"]
        self.dW += np.dot(dy, x.T)
        self.db += dy
        return np.dot(self.W.T, dy)

    def update(self, lr):
        g = np.clip(self.dW, -5.0, 5.0)
        self._tw += 1
        self._mw[:] = 0.9 * self._mw + 0.1 * g
        self._vw[:] = 0.999 * self._vw + 0.001 * (g ** 2)
        m_hat = self._mw / (1.0 - 0.9**self._tw)
        v_hat = self._vw / (1.0 - 0.999**self._tw)
        self.W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        g = np.clip(self.db, -5.0, 5.0)
        self._tb += 1
        self._mb[:] = 0.9 * self._mb + 0.1 * g
        self._vb[:] = 0.999 * self._vb + 0.001 * (g ** 2)
        m_hat = self._mb / (1.0 - 0.9**self._tb)
        v_hat = self._vb / (1.0 - 0.999**self._tb)
        self.b -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class LSTMModel:
    def __init__(self, input_size, hidden_size, horizon, seed=None):
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size, seed=seed)
        self.head = LinearLayer(hidden_size, horizon, seed=seed)

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        caches_seq = []
        for x in inputs:
            h, c, cache = self.cell.forward(x, h, c)
            caches_seq.append(cache)
        y, lin_cache = self.head.forward(h)
        return y, {"seq": caches_seq, "linear": lin_cache}

    def backward(self, pred, target, caches):
        err = pred - target
        n = pred.size
        loss = float(np.mean(err ** 2))
        dy = (2.0 / n) * err

        self.cell.zero_grad()
        self.head.zero_grad()
        dh = self.head.backward(caches["linear"], dy)
        dc = np.zeros((self.hidden_size, 1))
        for t in reversed(range(len(caches["seq"]))):
            dh, dc = self.cell.backward(caches["seq"][t], dh, dc)
        return loss

    def apply_updates(self, lr):
        self.cell.update(lr)
        self.head.update(lr)

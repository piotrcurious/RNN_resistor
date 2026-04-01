import numpy as np

class RCSimulator:
    def __init__(self, platform='esp32', R=10000, C=1e-6, V_supply=3.3):
        self.platform = platform
        self.R = R; self.C = C; self.V_supply = V_supply
        self.ADC_bits = 12 if platform == 'esp32' else 10
        self.ADC_ref = 3.3 if platform == 'esp32' else 5.0
        self.tau = R * C

    def time_to_reach(self, V_threshold_raw, V_initial):
        v_s = self.V_supply + np.random.normal(0, 0.05)
        v_target_adc = V_threshold_raw / (2**self.ADC_bits - 1) * self.ADC_ref
        # Systematic error (non-linearity/offset)
        v_phys = v_target_adc * (1.05 if self.platform == 'esp32' else 1.01)
        val = (v_phys - v_s) / (V_initial - v_s)
        if val <= 0: return 1.0
        return -self.tau * np.log(val)

def simulate_measurement(platform, R_true, C=1e-6, V_supply=None):
    if V_supply is None: V_supply = 3.3 if platform == 'esp32' else 5.0
    sim = RCSimulator(platform, R_true, C, V_supply)
    threshold_adc = 2588 if platform == 'esp32' else 648
    pulses = []; residuals = []
    for _ in range(10):
        v_res = np.random.uniform(0.0, 0.05)
        residuals.append(v_res * (2**sim.ADC_bits - 1) / sim.ADC_ref)
        t = sim.time_to_reach(threshold_adc, v_res)
        jitter = np.random.normal(0, 10e-6)
        pulses.append(max(1, int((t + jitter) * 1e6)))
    return pulses, residuals, V_supply

class RNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.in_s = input_size; self.h_s = hidden_size; self.out_s = output_size; self.lr = lr
        sc = np.sqrt(2.0 / (input_size + 2*hidden_size))
        self.Wxh = np.random.randn(hidden_size, input_size) * sc
        self.Whh = np.random.randn(hidden_size, hidden_size) * sc
        self.Why = np.random.randn(output_size, hidden_size) * sc
        self.bh = np.zeros((hidden_size, 1)); self.by = np.zeros((output_size, 1))
        self.sWxh = np.zeros_like(self.Wxh); self.sWhh = np.zeros_like(self.Whh); self.sWhy = np.zeros_like(self.Why)
        self.sbh = np.zeros_like(self.bh); self.sby = np.zeros_like(self.by)

    def reset(self): self.h = np.zeros((self.h_s, 1)); self.h_hist = [self.h.copy()]; self.x_hist = []

    def forward(self, x):
        self.x_hist.append(x)
        self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h) + self.bh)
        self.h_hist.append(self.h.copy())
        self.y = np.dot(self.Why, self.h) + self.by
        return self.y

    def backward(self, target):
        dy = self.y - target; steps = len(self.x_hist)
        dWhy = np.dot(dy, self.h_hist[-1].T); dby = dy
        dWxh = np.zeros_like(self.Wxh); dWhh = np.zeros_like(self.Whh); dbh = np.zeros_like(self.bh)
        dh_next = np.zeros((self.h_s, 1))
        for t in reversed(range(steps)):
            dh = dh_next + np.dot(self.Why.T, dy if t == steps-1 else 0)
            dh *= (1 - self.h_hist[t+1]**2)
            dbh += dh; dWxh += np.dot(dh, self.x_hist[t].T); dWhh += np.dot(dh, self.h_hist[t].T)
            dh_next = np.dot(self.Whh.T, dh)

        def upd(w, dw, s):
            dw = np.clip(dw, -1, 1) + 0.0001 * w
            s[:] = 0.9 * s + 0.1 * dw**2
            w -= self.lr * dw / (np.sqrt(s) + 1e-8)
        upd(self.Why, dWhy, self.sWhy); upd(self.by, dby, self.sby)
        upd(self.Wxh, dWxh, self.sWxh); upd(self.Whh, dWhh, self.sWhh); upd(self.bh, dbh, self.sbh)

if __name__ == "__main__":
    import sys
    platform = 'esp32' if len(sys.argv) < 2 else sys.argv[1]
    print(f"Simulating {platform}...")
    model = RNN(4, 16 if platform == 'esp32' else 6, 1)
    LOG_R_MIN = 2.0; LOG_R_MAX = 6.0; LOG_T_MIN = 0.0; LOG_T_MAX = 6.0
    V_REF = 3.3 if platform == 'esp32' else 5.0
    ADC_MAX = 4095.0 if platform == 'esp32' else 1023.0

    for i in range(10000):
        R = 10**np.random.uniform(LOG_R_MIN, LOG_R_MAX)
        pulses, residuals, v_s = simulate_measurement(platform, R)
        model.reset()
        for idx, p in enumerate(pulses):
            x = np.array([[(np.log10(p+1)-LOG_T_MIN)/6.0], [idx/10.0], [v_s/V_REF], [residuals[idx]/ADC_MAX]])
            model.forward(x)
        model.backward(np.array([[(np.log10(R)-LOG_R_MIN)/4.0]]))
        if i % 2000 == 0:
            pred = 10**(model.y[0,0]*4.0 + 2.0)
            print(f"Iter {i}: True {R:.0f}, Pred {pred:.0f}")

    errs = []
    for _ in range(500):
        R = 10**np.random.uniform(LOG_R_MIN, LOG_R_MAX)
        pulses, residuals, v_s = simulate_measurement(platform, R)
        model.reset()
        for idx, p in enumerate(pulses):
            x = np.array([[(np.log10(p+1)-LOG_T_MIN)/6.0], [idx/10.0], [v_s/V_REF], [residuals[idx]/ADC_MAX]])
            model.forward(x)
        pred = 10**(model.y[0,0]*4.0 + 2.0)
        errs.append(abs(pred - R) / R)
    print(f"Average Relative Error: {np.mean(errs)*100:.2f}%")

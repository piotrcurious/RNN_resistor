import numpy as np
import sys

class Simulator:
    def __init__(self, platform='esp32', R=10000, C1=1e-6, C2=1e-8, V_supply=3.3):
        self.platform = platform
        self.R = R; self.C1 = C1; self.C2 = C2; self.V_supply = V_supply
        self.ADC_bits = 12 if platform == 'esp32' else 10
        self.ADC_ref = 3.3 if platform == 'esp32' else 5.0

    def time_to_reach(self, V_threshold_raw, V_initial, temp, range_idx):
        C = self.C1 if range_idx == 0 else self.C2
        tau = self.R * (1 + 0.001 * (temp - 25.0)) * C
        v_s = self.V_supply + np.random.normal(0, 0.02)
        v_target_adc = V_threshold_raw / (2**self.ADC_bits - 1) * self.ADC_ref
        v_phys = v_target_adc * (1.05 if self.platform == 'esp32' else 1.01)
        val = (v_phys - v_s) / (V_initial - v_s)
        return -tau * np.log(max(1e-9, val))

def simulate_data(platform, R_true, temp):
    range_idx = 0 if R_true * 1e-6 > 0.001 else 1
    sim = Simulator(platform, R_true)
    thr = 2588 if platform == 'esp32' else 648
    pulses, residuals = [], []
    for _ in range(10):
        v_res = np.random.uniform(0.0, 0.02)
        residuals.append(v_res * (2**sim.ADC_bits - 1) / sim.ADC_ref)
        t = sim.time_to_reach(thr, v_res, temp, range_idx)
        pulses.append(max(1, int((t + np.random.normal(0, 5e-6)) * 1e6)))
    # For simulation, temp is already normalized 0-50 to match thermistor ADC range roughly
    t_adc = temp / 50.0 * (4095.0 if platform == 'esp32' else 1023.0)
    return pulses, residuals, sim.V_supply, t_adc, range_idx

class BaseML:
    def __init__(self, in_s, h_s, out_s, lr=0.001):
        self.in_s, self.h_s, self.out_s, self.lr = in_s, h_s, out_s, lr
        self.params, self.sq_grads = {}, {}

    def sigmoid(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def upd(self, name, dw):
        dw = np.clip(dw, -1, 1) + 0.0001 * self.params[name]
        self.sq_grads[name] = 0.9 * self.sq_grads[name] + 0.1 * dw**2
        self.params[name] -= self.lr * dw / (np.sqrt(self.sq_grads[name]) + 1e-8)

class RNN(BaseML):
    def __init__(self, in_s, h_s, out_s):
        super().__init__(in_s, h_s, out_s)
        sc = np.sqrt(2.0 / (in_s + 2*h_s))
        self.params = {'Wxh': np.random.randn(h_s, in_s)*sc, 'Whh': np.random.randn(h_s, h_s)*sc, 'Why': np.random.randn(out_s, h_s)*sc, 'bh': np.zeros((h_s, 1)), 'by': np.zeros((out_s, 1))}
        self.sq_grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def reset(self): self.h = np.zeros((self.h_s, 1)); self.h_hist = [self.h.copy()]; self.x_hist = []

    def forward(self, x):
        self.x_hist.append(x)
        self.h = np.tanh(np.dot(self.params['Wxh'], x) + np.dot(self.params['Whh'], self.h) + self.params['bh'])
        self.h_hist.append(self.h.copy()); self.y = np.dot(self.params['Why'], self.h) + self.params['by']
        return self.y

    def backward(self, target):
        dy = self.y - target; steps = len(self.x_hist); dWhy = np.dot(dy, self.h_hist[-1].T); dby = dy
        dWxh, dWhh, dbh, dh_next = np.zeros_like(self.params['Wxh']), np.zeros_like(self.params['Whh']), np.zeros_like(self.params['bh']), np.zeros((self.h_s, 1))
        for t in reversed(range(steps)):
            dh = (dh_next + np.dot(self.params['Why'].T, dy if t == steps-1 else 0)) * (1 - self.h_hist[t+1]**2)
            dbh += dh; dWxh += np.dot(dh, self.x_hist[t].T); dWhh += np.dot(dh, self.h_hist[t].T)
            dh_next = np.dot(self.params['Whh'].T, dh)
        for n, dw in [('Why', dWhy), ('by', dby), ('Wxh', dWxh), ('Whh', dWhh), ('bh', dbh)]: self.upd(n, dw)
        return float(np.sum(dy**2))

    def export(self, filename):
        with open(filename, "w") as f:
            for k, v in self.params.items(): f.write(f"const float pt_{k}[] = {{{', '.join(map(lambda x: f'{x:.8f}f', v.flatten()))}}};\n")

class GRU(BaseML):
    def __init__(self, in_s, h_s, out_s):
        super().__init__(in_s, h_s, out_s)
        sc = np.sqrt(2.0 / (in_s + 2*h_s))
        self.params = {'Wz': np.random.randn(h_s, in_s+h_s)*sc, 'Wr': np.random.randn(h_s, in_s+h_s)*sc, 'Wh': np.random.randn(h_s, in_s+h_s)*sc, 'bz': np.zeros((h_s,1)), 'br': np.zeros((h_s,1)), 'bh': np.zeros((h_s,1)), 'Why': np.random.randn(out_s, h_s)*sc, 'by': np.zeros((out_s,1))}
        self.sq_grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def reset(self): self.h = np.zeros((self.h_s, 1)); self.h_hist = [self.h.copy()]; self.x_hist = []; self.z_h = []; self.r_h = []; self.ht_h = []

    def forward(self, x):
        self.x_hist.append(x); c = np.vstack((x, self.h))
        z = self.sigmoid(np.dot(self.params['Wz'], c) + self.params['bz'])
        r = self.sigmoid(np.dot(self.params['Wr'], c) + self.params['br'])
        ht = np.tanh(np.dot(self.params['Wh'][:, :self.in_s], x) + np.dot(self.params['Wh'][:, self.in_s:], r*self.h) + self.params['bh'])
        self.h = (1-z)*self.h + z*ht; self.z_h.append(z); self.r_h.append(r); self.ht_h.append(ht); self.h_hist.append(self.h.copy())
        self.y = np.dot(self.params['Why'], self.h) + self.params['by']
        return self.y

    def backward(self, target):
        dy = self.y - target; steps = len(self.x_hist); dWhy = np.dot(dy, self.h_hist[-1].T); dby = dy
        dWz, dWr, dWh, dbz, dbr, dbh, dh_next = np.zeros_like(self.params['Wz']), np.zeros_like(self.params['Wr']), np.zeros_like(self.params['Wh']), np.zeros_like(self.params['bz']), np.zeros_like(self.params['br']), np.zeros_like(self.params['bh']), np.zeros((self.h_s, 1))
        for t in reversed(range(steps)):
            dh = (dh_next + np.dot(self.params['Why'].T, dy if t == steps-1 else 0))
            z, r, ht, hp, xt = self.z_h[t], self.r_h[t], self.ht_h[t], self.h_hist[t], self.x_hist[t]
            dht = dh * z * (1 - ht**2); dz = dh * (ht - hp) * z * (1 - z); dr = np.dot(self.params['Wh'][:, self.in_s:].T, dht) * hp * r * (1 - r)
            dbh += dht; dbz += dz; dbr += dr; dWz += np.dot(dz, np.vstack((xt, hp)).T); dWr += np.dot(dr, np.vstack((xt, hp)).T); dWh += np.dot(dht, np.vstack((xt, r*hp)).T)
            dh_next = dh*(1-z) + np.dot(self.params['Wz'][:, self.in_s:].T, dz) + np.dot(self.params['Wr'][:, self.in_s:].T, dr) + np.dot(self.params['Wh'][:, self.in_s:].T, dht)*r
        for n, dw in [('Why', dWhy), ('by', dby), ('Wz', dWz), ('Wr', dWr), ('Wh', dWh), ('bz', dbz), ('br', dbr), ('bh', dbh)]: self.upd(n, dw)
        return float(np.sum(dy**2))

    def export(self, filename):
        with open(filename, "w") as f:
            for k, v in self.params.items(): f.write(f"const float pt_{k}[] = {{{', '.join(map(lambda x: f'{x:.8f}f', v.flatten()))}}};\n")

if __name__ == "__main__":
    platform = 'esp32' if len(sys.argv) < 2 else sys.argv[1]
    model = GRU(6, 16, 1) if platform == 'esp32' else RNN(6, 6, 1)
    ADC_MAX = 4095.0 if platform == 'esp32' else 1023.0
    for i in range(10000):
        R = 10**np.random.uniform(1, 7); temp = np.random.uniform(0, 50)
        p, res, v_s, t_adc, ridx = simulate_data(platform, R, temp)
        model.reset()
        for j, pv in enumerate(p):
            # Normalization sync: dur/6.0, j/10.0, v_s/5.0, res/ADC_MAX, temp/ADC_MAX, ridx
            x = np.array([[(np.log10(pv+1))/6.0], [j/10.0], [v_s/5.0], [res[j]/ADC_MAX], [t_adc/ADC_MAX], [float(ridx)]])
            model.forward(x)
        model.backward(np.array([[(np.log10(R)-1.0)/6.0]]))
    model.export(f"pretrained_{platform}.h")
    print(f"Exported {platform}")

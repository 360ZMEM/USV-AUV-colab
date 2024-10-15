import numpy as np
import matplotlib.pyplot as plt
import time


def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


# Tidewave Class
class TideWave:
    def __init__(self, H, X_max, Y_max, T_max) -> None:
        # spacial paras, 2-dim
        self.dx = 1
        self.dy = 1  # spacial step
        self.dt = 0.025  # not 1s for avoiding numerical overflow (NaN)
        self.X_max = X_max
        self.Y_max = Y_max
        self.H = H
        self.T_max = T_max
        # nt
        self.nx = int(self.X_max / self.dx)
        self.ny = int(self.Y_max / self.dy)
        self.nt = int(self.T_max / self.dt)
        self.g = 9.8
        self.eta0 = 5
        # amplitude
        self.w = 2 * np.pi / 200
        # omega

    def low_pass(self, v):
        V = np.fft.fft(v)
        frequencies = np.fft.fftfreq(len(v), 1 / 10)
        filter_mask = np.abs(frequencies) <= 1
        V_filtered = V * filter_mask
        return np.real(np.fft.ifft(V_filtered))

    def calc_tideWave(self):
        # variables & boundary conditions
        # initialize velocities by low-frequency noises
        ux = (self.eta0 / 10) * np.random.normal(size=(self.nx,))
        uy = (self.eta0 / 10) * np.random.normal(size=(self.ny,))
        ux = self.low_pass(ux)
        uy = self.low_pass(uy)
        etax = np.zeros(self.nx + 1)
        etay = np.zeros(self.ny + 1)
        uxn = ux
        uyn = uy
        etaxn = etax
        etayn = etay
        upx = np.zeros((2, self.nx, self.nt))
        upy = np.zeros((2, self.ny, self.nt))
        self.etapx = np.zeros((2, self.nx + 1, self.nt))
        self.etapy = np.zeros((2, self.ny + 1, self.nt))
        for n in range(1, self.nt):  # x-dim calculation
            etax[self.nx] = self.eta0 * np.cos(self.w * n * self.dt - self.g)
            etaxn[self.nx] = etax[self.nx]
            for i in range(1, self.nx):
                uxn[i] = -self.g * (etax[i + 1] - etax[i]) * self.dt / self.dx + ux[i]
                etaxn[i] = -self.H * (uxn[i] - uxn[i - 1]) * self.dt / self.dx + etax[i]
            ux = uxn
            etax = etaxn
            for k in [0, 1]:
                upx[k, :, n] = ux
                self.etapx[k, :, n] = etax
        for n in range(1, self.nt):  # y-dim calculation
            etay[self.ny] = self.eta0 * np.cos(self.w * n * self.dt - self.g)
            etayn[self.ny] = etay[self.ny]
            for i in range(1, self.ny):
                uyn[i] = -self.g * (etay[i + 1] - etay[i]) * self.dt / self.dy + uy[i]
                etayn[i] = -self.H * (uyn[i] - uyn[i - 1]) * self.dt / self.dy + etay[i]
            uy = uyn
            etay = etayn
            for k in [0, 1]:
                upy[k, :, n] = uy
                self.etapy[k, :, n] = etay

    def get_tideHeight(self, x_r, y_r, t):
        x_in = int(x_r * self.nx)
        x_in = constrain(x_in, 0, self.etapx.shape[1] - 1)
        y_in = int(y_r * self.ny)
        y_in = constrain(y_in, 0, self.etapy.shape[1] - 1)
        try:
            return (
                self.etapx[0, x_in, int(t / self.dt)]
                + self.etapy[0, y_in, int(t / self.dt)]
                + self.H
            )
        except:
            return self.etapx[0, x_in, -1] + self.etapy[0, y_in, -1] + self.H


# USBL Class
class USBL:
    def __init__(self) -> None:
        # Preset comm. freq.
        self.f = np.array([1.2e4, 1.4e4, 1.6e4, 1.8e4])
        self.c = 1500
        # wavelength
        self.lamda = self.c / np.max(self.f)
        self.d = 0.4 * self.lamda
        # sampling freq.
        self.f0 = 2.016e6
        # cross-shape hydrophones array
        self.hyd_posit = np.array(
            [
                [self.d / 2, 0, 0],
                [-self.d / 2, 0, 0],
                [0, self.d / 2, 0],
                [0, -self.d / 2, 0],
            ]
        )

    # calculating SN Ratio
    def calcSNR(self, f, b, d, format="active"):
        # sonar power
        SL = 145
        lgNt = 17 - 30 * np.log10(f)
        lgNs = 40 + 26 * np.log10(f) - 60 * np.log10(f + 0.03)
        lgNw = 50 + 20 * np.log10(f) - 40 * np.log10(f + 0.4)
        lgNth = -15 + 20 * np.log10(f)
        NL = 10 * np.log10(
            1000
            * b
            * (
                10 ** (lgNt / 10)
                + 10 ** (lgNs / 10)
                + 10 ** (lgNw / 10)
                + 10 ** (lgNth / 10)
            )
        )
        alpha = (
            0.11 * ((f**2) / (1 + f**2))
            + 44 * ((f**2) / (4100 + f**2))
            + (2.75e-4) * (f**2)
            + 0.003
        )
        TL = 15 * np.log10(d) + alpha * (0.001 * d)
        TS = 3
        if format == "active":
            SNR = SL - 2 * TL - NL + TS
        elif format == "passive":
            SNR = SL - TL + TS
        else:
            raise NotImplementedError
        return SNR

    # Measure the phase difference between the acoustic signal sent by sonar and the signal received with noise
    def get_phasedelay(self, dist, idx=0):  # idx -> AUV index
        # generate original singal, 10T
        t_length = int(10 * (self.f0 / self.f[idx]))
        t = np.arange(t_length) / self.f0
        # generate received signal,
        real_det_t = dist / self.c
        recv_signal = np.sin(2 * np.pi * self.f[idx] * (t - real_det_t))
        # calculate SNR
        SNR = self.calcSNR(self.f[idx] / 1000, 1, dist, format="active")
        # add noise
        noise = (10 ** (-SNR / 10)) * np.random.randn(t_length)
        recv_signal += noise
        A = np.column_stack(
            (np.sin(2 * np.pi * self.f[idx] * t), np.cos(2 * np.pi * self.f[idx] * t))
        )
        coeffs, _, _, _ = np.linalg.lstsq(A, recv_signal, rcond=None)
        sin_coeff, cos_coeff = coeffs
        phase_diff = np.arctan2(cos_coeff, sin_coeff)
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        return phase_diff

    def time_estimate(self, signal, pulse):
        N = len(signal)
        M = len(pulse)
        J = np.zeros(N - M + 1)
        for n0 in range(N - M + 1):
            signal_dat = signal[n0 : n0 + M]
            J[n0] = np.dot(signal_dat, pulse)
        # signal_matrix = np.lib.stride_tricks.sliding_window_view(signal, M)
        # l = time.time()
        # Compute the dot product for each shifted version with the pulse
        # J = np.dot(signal_matrix, pulse)
        # Find the index of the maximum value in J
        n0hat = np.argmax(J)
        time_delay = n0hat / self.f0
        return time_delay

    # time delay using correlation
    def calc_timeDelay(self, real_t, f_idx=0):
        ret_delayt = np.zeros_like(real_t)
        K = 3
        t_origin = np.linspace(
            0, K * 2 * np.pi, int(K * self.f0 / self.f[f_idx]), dtype=np.float64
        )
        y_origin = np.sin(t_origin)
        T = 2 * K / self.f[f_idx]
        if real_t >= 1.0:
            raise NotImplementedError  # too large delay
        y_rec1 = np.zeros(int(self.f0 * T))
        y_rec2 = np.zeros(int(self.f0 * T))
        rt_idx = real_t * self.f0
        int_rt_idx = int(rt_idx)
        rt_idx -= int_rt_idx - 10
        y_rec1[int(rt_idx) : int(rt_idx) + int(K * self.f0 / self.f[f_idx])] = y_origin
        y_rec2[
            int(rt_idx) + 1 : int(rt_idx) + int(K * self.f0 / self.f[f_idx]) + 1
        ] = y_origin
        k_yrec2 = rt_idx - int(rt_idx)
        y_rec = k_yrec2 * y_rec2 + (1 - k_yrec2) * y_rec1

        SNR = self.calcSNR(
            self.f[f_idx] / 1000, 1, real_t * self.c / 2, format="active"
        )
        r_SNR = 10 ** (-SNR / 10)
        y_rec = y_rec + np.random.normal(0, r_SNR, size=y_rec2.shape)
        return self.time_estimate(y_rec, y_origin) + (int_rt_idx - 10) / self.f0

    def calcPosit(self, real_posit, idx=0):
        calc_posit = np.zeros(3)
        real_dposit = real_posit + self.hyd_posit
        real_delayt = np.linalg.norm(real_dposit, axis=1) / self.c * 2
        # calc delayt one by one
        calc_phaset = np.array(
            [self.get_phasedelay(real_delayt[i] * self.c / 2, idx) for i in range(4)]
        )

        # normalize phase
        # dphasex = (
        #     np.arctan2(
        #         np.sin(calc_phaset[1] - calc_phaset[0]),
        #         np.cos(calc_phaset[1] - calc_phaset[0]),
        #     )
        #     / 2
        # )
        # dphasey = (
        #     np.arctan2(
        #         np.sin(calc_phaset[3] - calc_phaset[2]),
        #         np.cos(calc_phaset[3] - calc_phaset[2]),
        #     )
        #     / 2
        # )
        if abs(calc_phaset[0] - calc_phaset[1]) > np.pi:
            calc_phaset[0] -= np.sign(calc_phaset[0] - calc_phaset[1]) * 2 * np.pi
        if abs(calc_phaset[3] - calc_phaset[2]) > np.pi:
            calc_phaset[3] -= np.sign(calc_phaset[3] - calc_phaset[2]) * 2 * np.pi
        dphasex = calc_phaset[1] - calc_phaset[0]
        dphasey = calc_phaset[3] - calc_phaset[2]
        # time calc
        calc_delayt = self.calc_timeDelay(real_delayt[0], f_idx=idx) / 2
        # calculate position
        calc_posit[0] = (
            (self.c)
            / (2 * np.pi * self.f[idx] * self.d)
            * (dphasex)
            * (self.c * calc_delayt)
        )
        calc_posit[1] = (
            (self.c)
            / (2 * np.pi * self.f[idx] * self.d)
            * (dphasey)
            * (self.c * calc_delayt)
        )
        calc_posit[2] = np.sqrt(
            (self.c * calc_delayt) ** 2 - calc_posit[0] ** 2 - calc_posit[1] ** 2
        )
        return calc_posit


# Test
if __name__ == "__main__":
    # ---- USBL ----
    usbl = USBL()
    # real_posit = np.array([34, 67, 100])  # feel free to change this
    real_posit = np.array([-299, -19, 100])
    pred_posit = usbl.calcPosit(real_posit, idx=0)
    print('pred posit / m', pred_posit)
    print('error / m', np.linalg.norm(pred_posit[:2] - real_posit[:2]))
    # ---- TideWave ----
    tidewave = TideWave(H=100, X_max=100, Y_max=100, T_max=200)
    tidewave.calc_tideWave()
    x = np.arange(tidewave.nx) * tidewave.dx
    y = np.arange(tidewave.ny) * tidewave.dy
    X, Y = np.meshgrid(x, y)
    tw_height = np.zeros(X.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            tw_height[j, i] = (
                tidewave.etapx[0, i, int(100 / tidewave.dt)]
                + tidewave.etapy[0, j, int(100 / tidewave.dt)]
            )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, tw_height, cmap="viridis")
    ax.set_xlabel("x/m")
    ax.set_ylabel("y/m")
    ax.set_zlabel("Tidewave Height/m")
    plt.show()

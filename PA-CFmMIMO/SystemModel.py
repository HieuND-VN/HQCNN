import numpy as np


class SysMod():
    def __init__(self, args):
        self.num_ap = args.num_ap
        self.num_ue = args.num_ue
        self.tau_p = args.tau_p
        self.length = args.length #region from -1 to 1 in both axis (km)
        self.f = args.freq
        self.h_ap = args.h_ap #km
        self.h_ue = args.h_ue #km
        self.d1 = args.d1
        self.d0 = args.d0
        self.B = self.B
        self.L = 46.5 + 33.9*np.log10(self.f) - 13.82*np.log10(self.h_ap) - (1.1*np.log10(self.f) - 0.7)*self.h_ue + (1.56*np.log10(self.f) - 0.8)
        self.Pd = (self.B*10**(-17.4)*10**(-3))**-1
        self.p = args.p #1mW, Power UL of pilot signal
        self.P = args.P #mW, Power Downlink in each AP
        self.eta = self.P / self.num_ue
        self.pilot_index_random = np.random.randint(self.tau_p-1, size = self.num_ue)

        self.AP_position = self.AP_location_generator()

    
    def AP_location_generator(self):
        sensitivity_1 = 0.1
        sensitivity_2 = 0.2
        AP_position = np.zeros((self.num_ap,1), dtype = "complex")
        theta = np.linsapce(0,2*np.pi,self.num_ap)
        for i in range(self.num_ap):
            AP_position[i] = self.length * sensitivity_1 * np.cos(theta[i]) + 1j * self.length * sensitivity_2 * np.sin(theta[i])
        return AP_position

    def distance_calculator(self, UE_position):
        diff = self.AP_position[:, np.newaxis] - UE_position[np.newaxis, :]
        return (np.sqrt(np.real(diff) ** 2 + np.imag(diff) ** 2)).reshape(10, 8)

    def LSF_calculator(self, distanceUE2AP):
        pathloss = np.zeros((self.num_ap, self.num_ue))
        # beta = np.zeros((self.num_ap, self.num_ue))
        for m in range(self.num_ap):
            for k in range(self.num_ue):
                if distanceUE2AP[m,k] > self.d1:
                    pathloss[m,k] = -self.L - 35*np.log10(distanceUE2AP[m,k])
                elif distanceUE2AP[m,k] <= self.d1 and distanceUE2AP[m,k] > self.d0:
                    pathloss[m,k] = -self.L -15*np.log10(self.d1) - 20*np.log10(distanceUE2AP[m,k])
                else:
                    pathloss[m,k] = -self.L -15*np.log10(self.d1) - 20*np.log10(self.d0)
        beta = self.Pd*10**(pathloss/10)
        return beta

    def calculate_c(self, pilot_index, beta):
        c = np.zeros((self.num_ap, self.num_ue))
        for m in range(self.num_ap):
            for k in range(self.num_ue):
                # Tính tử số
                numerator = np.sqrt(self.tau_p * self.p) * beta[m, k]
                sum = 0
                # Tính mẫu sổ
                for j in range(self.num_ue):
                    if (pilot_index[j] == pilot_index[k]):
                        sum += beta[m, j]
                denominator = self.tau_p * self.p * sum + 1
                c[m, k] = numerator / denominator
        return c

    def sinr_calculator(self, pilot_index, beta, gamma, c_mk):
        sinr = np.zeros(self.num_ue)
        for k in range(self.num_ue):
            # DS_k
            numerator = 0
            numerator = self.P * self.eta * (np.sum(gamma[:, k])) ** 2

            # denominator
            demoninator = 0
            # UI
            UI = 0
            sum_12 = 0
            for j in range(self.num_ue):
                # if (j==k): continue;
                # if (pilot_index[j] == pilot_index[k]): continue;
                if (j != k) and (pilot_index[j] == pilot_index[k]):
                    sum_11 = 0
                    for m in range(self.num_ap):
                        sum_11 += (gamma[m, j] * beta[m, k] / beta[m, j])
                else:
                    sum_11 = 0
                sum_12 += sum_11 ** 2
            UI = self.P * self.eta * sum_12

            # BU
            BU = 0
            sum_BU = 0
            for j in range(self.num_ue):
                for m in range(self.num_ap):
                    sum_BU += gamma[m, j] * beta[m, k]
            BU = self.P * self.eta * sum_BU
            demoninator = UI + BU + 1
            sinr[k] = numerator / demoninator
        return sinr
    def dl_rate_calculator(self, pilot_index, beta):
        c_mk = self.calculate_c(pilot_index,beta)
        gamma = np.sqrt(self.tau_p * self.p) * beta * c_mk
        sinr = self.sinr_calculator(pilot_index, beta, gamma, c_mk)
        return np.log2(1+sinr)
    def greedy_assignment(self, beta, N):
        pilot_index = self.pilot_index_random
        for n in range(N):
            dl_rate = self.dl_rate_calculator(pilot_index, beta)
            k_star = np.argmin(dl_rate)
            sum_beta = np.zeros(self.tau_p)
            for tau in range(self.tau_p):
                for m in range(self.num_ap):
                    for k in range(self.num_ue):
                        if (k != k_star) and (pilot_index[k] == tau):
                            sum_beta[tau] += beta[m, k]
            pilot_index[k_star] = np.argmin(sum_beta)
        return pilot_index

    def data_sample_generator(self):
        UE_position = np.random.uniform(low=-self.length, high=self.length, size=self.num_ue) + 1j*np.random.uniform(low=-self.length, high=self.length, size=self.num_ue)
        distanceUE2AP = self.distance_calculator(UE_position)
        beta = self.LSF_calculator(distanceUE2AP)
        pilot_index = self.greedy_assignment(beta, N = 20)
        beta_flatten = beta.flatten()
        pilot_index_flatten = pilot_index.flatten()
        sample = np.concatenate((beta_flatten, pilot_index_flatten))
        return sample


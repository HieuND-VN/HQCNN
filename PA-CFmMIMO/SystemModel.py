import numpy as np


class SysMod():
    def __init__(self, args):
        self.num_ap = args.num_ap
        self.num_ue = args.num_ue
        self.tau_p = args.tau_p
        self.length = args.length #region from -1 to 1 in both axis (km)
        self.f = args.freq
        self.h_ap = args.h_ap/1000 #km
        self.h_ue = args.h_ue/1000 #km
        self.B = self.B
        self.L = 46.5 + 33.9*np.log10(self.f) - 13.82*np.log10(self.h_ap) - (1.1*np.log10(self.f) - 0.7)*self.h_ue + (1.56*np.log10(self.f) - 0.8)
        self.Pd = (self.B*10**(-17.4)*10**(-3))**-1
        self.p = args.p #1mW, Power UL of pilot signal
        self.P = args.P #mW, Power Downlink in each AP
        self.eta = self.P / self.num_ue

        self.AP_position = self.AP_location_generator

    
    def AP_location_genertor(self):
        sensitivity_1 = 0.1
        sensitivity_2 = 0.2
        AP_position = np.zeros((self.num_ap,1), dtype = "complex")
        theta = np.linsapce(0,2*np.pi,self.num_ap)
        return AP_position

    
    def distance_calculator(self):
        distance = np.zeros((self.num_ap, self.num_ue))
        return distance
    
    def pathloss_calculator(self):
        pathloss = np.zeros((self.num_ap, self.num_ue))
        return pathloss
    
    def LSF_calculator(self):
        beta = np.zeros((self.num_ap,self.num_ue))
        return beta
    
    # Calculate gamma coefficient
    def gamma_calculator(self):
        gamma = np.zeros((self.num_ap,self.num_ue))
        return gamma
    
    def channel_coefficient_calculator(self):
        channel_coeff = np.zeros((self.num_ap,self.num_ue))
        return channel_coeff
    
    # Calculate SINR by calculate numerator and deminorator
    def SINR_calculator(self, pilot_index):
        sinr = np.zeros(self.num_ue)
        return sinr
    
    def dl_rate_calculator(self, pilot_index):
        sinr = self.SINR_calculator(pilot_index)
        dl_rate = np.log2(1+sinr)
        return dl_rate
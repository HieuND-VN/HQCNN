import numpy as np


class SysMod():
    def __init__(self, args):
        self.num_ap = args.num_ap
        self.num_ue = args.num_ue
        self.tau_p = args.tau_p
        self.AP_position = self.AP_location()
        self.UE_position = self.UE_location()
        self.distanceUE2AP = self.distance_calculator()
        self.pathloss = self.pathloss_calculator()
        self.beta = self.LSF_calculator()
        self.gamma = self.gamma_calculator()
        self.c = self.channel_coefficient_calculator()
    
    def AP_location_genertor(self):
        AP_position = np.zeros((self.num_ap,2))
        return AP_position
    
    def UE_location_generator(self):
        UE_position = np.zeros((self.num_ue,2))
        return UE_position
    
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
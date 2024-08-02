import time
import argparse
import torch 
from System Model import SysMod

if __name__ == "__main__":
    total_start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_ap', "--number_AP", type=int, default=10, help="Number of Access Point.")
    parser.add_argument('-num_ue', "--number_UE", type=int, default=8, help="Number of User Equipment.")
    parser.add_argument('-tau_p', "--pilot_length", type=int, default=6, help="Number of pilot sequences.") #Also the number of pilot = tau_p
    parser.add_argument('-h_ap', "--height_AP", type=int, default = 0.015, help="Height of Access Point (in km).")
    parser.add_argument('-h_ue', "--height_UE", type=int, default=0.00165, help="Height of User Equipment (in km).")
    parser.add_argument('-p', "--power_pilot", type=float, default=1, help="Power of pilot signal (in mW)")
    parser.add_argument('-P', "-power_AP", type=float, default=100, help="Transmition power of each AP")



    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])

    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
    
    system = SysMod(args)
    
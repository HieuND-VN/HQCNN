import time
import argparse
import torch 
from SystemModel import SysMod

if __name__ == "__main__":
    total_start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_ap', "--number_AP", type=int, default=10, help="Number of Access Point.")
    parser.add_argument('-num_ue', "--number_UE", type=int, default=8, help="Number of User Equipment.")
    parser.add_argument('-tau_p', "--pilot_length", type=int, default=6, help="Number of pilot sequences.") #Also the number of pilot = tau_p
    parser.add_argument('-length', "--area_length", type=int, default=1, help="Area length, from -1 to 1.")
    parser.add_argument('-f', "--frequency", type=int, default=1900, help="Frequency 1900 MHz.")
    parser.add_argument('-B', "--bandwidth", type=int, default=20e6, help="Frequency 20 MHz.")
    parser.add_argument('-h_ap', "--height_AP", type=float, default=0.015, help="Height of Access Point (in km).")
    parser.add_argument('-h_ue', "--height_UE", type=float, default=0.0017, help="Height of User Equipment (in km).")
    parser.add_argument('-p', "--power_pilot", type=float, default=1, help="Power of pilot signal (in mW)")
    parser.add_argument('-P', "--power_AP", type=float, default=100, help="Transmission power of each AP")
    parser.add_argument('-d1', "--distance1", type=float, default=0.05, help="Distance range number 1")
    parser.add_argument('-d0', "--distance1", type=float, default=0.01, help="Distance range number 0")

    # parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-num_train', "--training_sample", type=int, default=3000, help="Number of training samples.")
    parser.add_argument('-num_test', "--testing_sample", type=int, default=500, help="Number of testing samples.")

    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not available.\n")
    
    system = SysMod(args)

    train_data_loader = torch.utils.data.DataLoader(list(zip(X_train, Y_train_hot)), batch_size = args.batch_size, shuffle = False, drop_last = True)
    val_data_loader = torch.utils.data.DataLoader(list(zip(X_val, Y_val_hot)), batch_size = args.batch_size, shuffle = True, drop_last = True)
    test_data_loader = torch.utils.data.DataLoader(list(zip(X_test, Y_test_hot)), batch_size = args.batch_size, shuffle = True, drop_last = True)
    
import profile
import client
from os import path
from os import mkdir

profiles_dir = "./node_profiles"

def check_nodes(path):
    if not path.exists(path):
        print("No node profiles found!")
        print("Initial setup?")
        init_check = input("\t[Y/n] >>> ")
        if init_check == "n":
            print("Exiting...")
            exit()
        else:
            print("No input found")
            print("Creating default directory...")
            mkdir(path)

def create_node(n, subset)
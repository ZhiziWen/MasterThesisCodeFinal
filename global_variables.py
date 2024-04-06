"""
This file contains the global variables for the project.
"""


import time

timestr = time.strftime("%Y%m%d-%H%M")
project_folder = "D:/SS2023/MasterThesis/codeFINAL/code_final/" # change this to the path where the project folder is located, example: D:/SS2023/MasterThesis/codeFINAL/code_final/

if project_folder == "":
    raise ValueError("Please specify the project folder path in the global_variables.py file.")
import time
import math
import numpy as np

from DOMAIN import DOMAIN_ANGLES
from mycobot_main import MyCobot

mycobot = MyCobot()
mc = mycobot.mc

if mc.is_controller_connected() != 1:
    print("Please connect the robot arm correctly for program writing")
    exit(0)

mycobot.reset()
time.sleep(9)

angles_list = [np.around(float(np.random.uniform(low=i[0],high=i[1],size=1)),decimals=2) for i in DOMAIN_ANGLES]

iteration = 6
angles_orig = mc.get_angles()
while not angles_orig:
    angles_orig = mc.get_angles()

start = np.array(angles_orig)
end = np.array(angles_list)
diff = (end-start) /iteration
new_coords=[]
while not new_coords:
    new_coords = mc.get_coords()

i = 1
new_coords = []
while i <=iteration:
    old_coords = new_coords
    angles_orig = angles_orig + diff
    print(i, "=", angles_orig)
    
    mc.step(angles_orig.tolist(),speed=10)
    time.sleep(2)
    
    new_coords = mc.get_coords()
    while not new_coords:
        new_coords = mc.get_coords()
    
    if new_coords == old_coords:
        raise ValueError("Body Collision happens.")
    i += 1
    if new_coords[2] <= 60:
        print("The gripper will touch the surface of table.")
        print("Mycobot stops at", angles_orig, ", the final destination is", angles_list)
        break
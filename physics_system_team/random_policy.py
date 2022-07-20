import numpy as np

from time import sleep
from mycobot_main import MyCobot
from DOMAIN import DOMAIN_ANGLES

count = 0
max_random_policy_test = 100
while True and count < max_random_policy_test:
    count += 1
    print(f"\nNo.{count} random policy test.")
    angles = [np.around(float(np.random.uniform(low=i[0],high=i[1],size=1)),decimals=2) for i in DOMAIN_ANGLES]
    print("Now, we're moving mycobot to angles:\n",angles)
    MyCobot.step(angles)
    sleep(5)
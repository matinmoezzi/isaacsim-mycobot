JOINT1_MIN = -190
JOINT1_MAX = 190
JOINT2_MIN = -80
JOINT2_MAX = 80
JOINT3_MIN = -110
JOINT3_MAX = 150
JOINT4_MIN = -85
JOINT4_MAX = 140
JOINT5_MIN = -190
JOINT5_MAX = 190
JOINT6_MIN = -190
JOINT6_MAX = 190

JOINT_LOWER_BOUND = [JOINT1_MIN,JOINT2_MIN,JOINT3_MIN,
                     JOINT4_MIN,JOINT5_MIN,JOINT6_MIN]

JOINT_UPPER_BOUND = [JOINT1_MAX,JOINT2_MAX,JOINT3_MAX,
                     JOINT4_MAX,JOINT5_MAX,JOINT6_MAX]

DOMAIN_ANGLES = [[JOINT1_MIN, JOINT1_MAX], [JOINT2_MIN, JOINT2_MAX],
                 [JOINT3_MIN, JOINT3_MAX], [JOINT4_MIN, JOINT4_MAX],
                 [JOINT5_MIN, JOINT5_MAX], [JOINT6_MIN, JOINT6_MAX]]

SPEED_DEFAULT = 20
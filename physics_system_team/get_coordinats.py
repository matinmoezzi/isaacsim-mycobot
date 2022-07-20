import numpy as np

'''
The parameters of mycobot docs: 
https://docs.elephantrobotics.com/docs/gitbook-en/2-serialproduct/2.1-280/2.1.2.1%20Introduction%20of%20product%20parameters.html

All length unit is Millimetre and plane angle unit is Degree
In Sim2real project, Physics System part
Created by Chongyu Zhu, July 22nd, 2022
'''

class CoordinatesTransform():
    def __init__(self):
        # transformation matrix construction parameter
        self.bottom_row = np.array([0,0,0,1],dtype=float).reshape((1,4))

        # myCobot main part dimension
        self.coordinates = np.array([(  0.000,  0.000,  0.000),  # world coordinates
                                     (  0.000,  0.000, 89.426),  # Joint 1 coordinates refers to world
                                     ( 34.705,  0.000, 59.705),  # Joint 2 coordinates refers to Joint 1
                                     ( -6.789,  0.000,110.000),  # Joint 3 coordinates refers to Joint 2
                                     (  1.444,  0.000, 96.000),  # Joint 4 coordinates refers to Joint 3
                                     ( 33.245,  0.000, 34.534),  # Joint 5 coordinates refers to Joint 4
                                     (  0.000, 32.405, 40.516),  # Joint 6 coordinates refers to Joint 5
                                     (-35.355, 35.000, 35.355)]) # Camera coordinates refers to Joint 6
        
        # Camera parameters
        self.pixel_size = 3.4 * 10**(-3)
        self.focal_length = 1.7
        self.camera_resolution = [640, 480]

    # This is the main function to calculate the object location refering to robot arm base.
    def object_coordinates(self, p: list , joints: list) -> np.array:
        # print("The position of object in the end effector coordinates is\n({}, {})".format(p[0],p[1]))
        # print("\njoints angles are:\n{}\n".format(joints))

        # joint angle direction calibration
        joints = self.joints_direction_calibration(joints)

        # initialize a dict to store each step transformation matrix
        transform_matrix = {}

        # Transformation matrix 3D format:
        # [rxx, rxy, rxz, x
        #  ryx, ryy ,ryz, y
        #  rzx, rzy, rzz, z
        #    0,   0,   0, 1]
        # Transform matrix of Joint 1
        transform_matrix.update\
            ({1:np.bmat([[self.rotate_z(joints[0]),self.coordinates[1,:].reshape((3,1))],\
                         [self.bottom_row]])})
        
        # Transform matrix of Joint 2,3,4
        for i in range(1,4):
            transform_matrix.update\
                ({i+1:np.bmat([[self.rotate_x(joints[i]),self.coordinates[i+1,:].reshape((3,1))],\
                               [self.bottom_row]])})

        # Transform matrix of Joint 5
        transform_matrix.update\
            ({5:np.bmat([[self.rotate_z(joints[4]),self.coordinates[5,:].reshape((3,1))],\
                         [self.bottom_row]])})

        # Transform matrix of Joint 6
        transform_matrix.update\
            ({6:np.bmat([[self.rotate_y(joints[5]),self.coordinates[6,:].reshape((3,1))],\
                         [self.bottom_row]])})

        # Transform matrix of Camera
        transform_matrix.update\
            ({7:np.bmat([[self.rotate_y(45),self.coordinates[7,:].reshape((3,1))],\
                         [self.bottom_row]])})
        transform_matrix.update\
            ({8:np.bmat([[self.rotate_x(-90),np.zeros((3,1))],\
                         [self.bottom_row]])})
        transform_matrix.update\
            ({9:np.bmat([[self.rotate_z(-180),np.zeros((3,1))],\
                         [self.bottom_row]])})

        # initialize T
        T = np.eye(4, dtype=float)
        # get the transformation matrix from world to camera
        for t in transform_matrix.values():
            T = T * t
            # print(np.round(T,5),"\n")

        # coordinates calculation from camera
        height = T[2,3]
        center = np.array([self.camera_resolution[0]/2, self.camera_resolution[1]/2], dtype=int)
        # print(center)
        p_array = np.array(p, dtype=float) * np.array([1,1],dtype=float)
        # print(p_array)

        '''
        Further Improvement:
        Currently, we deal the monocular camera visual as parallel light result instead of through 
        the pinhole and set the camera face right down instead of any angle. So, regarding to these
        two limitation:
        1. Eliminate the distortion caused by monocular camera at the edges of camera video
        2. Be able to handle different angle of camera to get correct object position regarding to 
        the world coordinates.
        '''
        cor_in_camera = (p_array - center) * height / self.focal_length * self.pixel_size / 1.5
        coordinate = np.array([cor_in_camera[0], cor_in_camera[1], height], dtype=float) * np.array([1,1,1],dtype=float)
        # print(np.array(T[0:2,3].reshape(-1))[0], coordinate[0:2])

        # Get transformation matrix from world to object
        transform_matrix.update\
            ({'cam2object':np.bmat([[self.rotate_x(0),coordinate.reshape((3,1))],\
                                    [self.bottom_row]])})
        T = T * transform_matrix['cam2object']
        # print("The final transformation matrix from world to object:\n {}\n".format(np.round(T,5)))

        # return T for function validation. in later use, comment out T
        return np.array(T[0:2,3].reshape(-1))[0], T
        

    def rotate_x(self, angle):
        theta = np.deg2rad(angle)
        return np.array([(1,0,0),(0,np.cos(theta),-np.sin(theta)),(0,np.sin(theta),np.cos(theta))])

    def rotate_y(self, angle):
        theta = np.deg2rad(angle)
        return np.array([(np.cos(theta),0,np.sin(theta)),(0,1,0),(-np.sin(theta),0,np.cos(theta))])

    def rotate_z(self, angle):
        theta = np.deg2rad(angle)
        return np.array([(np.cos(theta),-np.sin(theta),0),(np.sin(theta),np.cos(theta),0),(0,0,1)])

    # unsure about the positive direction of its rotation
    # write this function to modify the angle of the input joints
    # only need to amend corresponding number in calibration_list
    def joints_direction_calibration(self, joints):
        calibration_list = [1,1,1,1,1,1]
        return [a*b for a,b in zip(joints,calibration_list)]


class IntegrationTest():
    def __init__(self):
        self.integration_test_1()
        self.integration_test_2()
        self.integration_test_3()
        # self.integration_test_4()

    # camera location
    def p_center_calculation(self,x,y,w,h):
        return [x+w/2, y+h/2]
        
    def integration_test_1(self):
        origin_x = 236
        origin_y = 48
        width = 361
        height = 420
        joints = [0,0,-45,-45,0,0]

        coor,_ = CoordinatesTransform().object_coordinates\
            (self.p_center_calculation(origin_x,origin_y,width,height),joints)
        print("Target: ", coor) # now(56, 135) true: around(0, 185)
        print("Desire: ", np.array([0,185], dtype=float))
        print(coor-np.array([0,185], dtype=float),"\n")
        # print("Integration Test 1 success.\n")

    def integration_test_2(self):
        origin_x = 74
        origin_y = -10
        width = 285
        height = 477
        joints = [0,0,-45,-45,0,0]

        coor,_ = CoordinatesTransform().object_coordinates\
            (self.p_center_calculation(origin_x,origin_y,width,height),joints)
        print("Target: ",coor) # now(-7, 221) true: around(57, 171)
        print("Desire: ", np.array([57,171], dtype=float))
        print(coor-np.array([57,171], dtype=float))
        # print("Integration Test 2 success.\n")

    def integration_test_3(self):
        origin_x = 320
        origin_y = 240
        width = 0
        height = 0
        joints = [0,0,-45,-45,0,0]

        coor,_ = CoordinatesTransform().object_coordinates\
            (self.p_center_calculation(origin_x,origin_y,width,height),joints)
        print("\nTarget: ",coor)

    def integration_test_4(self):
        origin_x = 300
        origin_y = 200
        width = 40
        height = 80
        joints = [0,0,-45,-45,0,0]

        coor,_ = CoordinatesTransform().object_coordinates\
            (self.p_center_calculation(origin_x,origin_y,width,height),joints)
        print("\nTarget: ",coor)

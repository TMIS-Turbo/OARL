from __future__ import absolute_import
from __future__ import print_function
import gym
import numpy as np
from gym import spaces
import os
import sys
import xml.dom.minidom

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci
import traci.constants as tc

gui = True
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')
config_path = os.path.dirname(__file__)+"/../../../Data/StraightRoad.sumocfg"


class HighwayEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.minAutoVelocity = 0
        self.maxAutoVelocity = 30

        self.minOtherVehVelocity = 0
        self.maxOtherVehVelocity = 20

        self.minDistanceFrontVeh = 0
        self.maxDistanceFrontVeh = 100

        self.minDistanceRearVeh = 0
        self.maxDistanceRearVeh = 100

        self.minLaneNumber = 0
        self.maxLaneNumber = 3

        self.maxAcceleration = 30
        self.minAcceleration = -30

        self.maxTotalDistanceCovered = 40000
        self.minTotalDistanceCovered = -1

        high = np.array([self.maxAutoVelocity, self.maxOtherVehVelocity, self.maxDistanceFrontVeh, self.maxOtherVehVelocity, self.maxDistanceRearVeh, self.maxOtherVehVelocity, self.maxDistanceFrontVeh, self.maxOtherVehVelocity, self.maxDistanceRearVeh, self.maxOtherVehVelocity, self.maxDistanceFrontVeh, self.maxOtherVehVelocity, self.maxDistanceRearVeh, self.maxLaneNumber, self.maxAcceleration, self.maxTotalDistanceCovered])
        low = np.array([self.minAutoVelocity, self.minOtherVehVelocity, self.minDistanceFrontVeh, self.minOtherVehVelocity, self.minDistanceRearVeh, self.minOtherVehVelocity, self.minDistanceFrontVeh, self.minOtherVehVelocity, self.minDistanceRearVeh, self.minOtherVehVelocity, self.minDistanceFrontVeh, self.minOtherVehVelocity, self.minDistanceRearVeh, self.minLaneNumber, self.minAcceleration, self.minTotalDistanceCovered])
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.pre_findstate = None
        self.result = []
        self.VehicleIds = []
        self.AutoCarID = 'Auto'
        self.end = False
        self.reset_times = 0

        # front vehicle characteristics previous step values
        self.PrevFrontVehID = 'None'
        self.PrevFrontVehDistance = 100
        self.Pre_angle = 90

        # Current Vehicle characterisitic of vehicle in front
        self.CurrFrontVehID = 'None'
        self.CurrFrontVehDistance = 100
        self.StartTime = 0
        self.TotalReward = 0
        self.AutocarSpeed = 0

        # Fault Simulation
        self.CommRange = 100
        self.DisableFaultSimulation = True
        self.ErrorPropability = 0
        self.numberOfLaneChanges = 0
        self.numberOfOvertakes = 0
        self.currentTrackingVehId = 'None'

    def _findRearVehDistance(self, vehicleparameters):
        parameters = [[0 for x in range(4)] for x in range(len(vehicleparameters))]
        i = 0
        d1 = -1
        d2 = -1
        d3 = -1
        d4 = -1
        d5 = -1
        d6 = -1
        v1 = -1
        v2 = -1
        v3 = -1
        v4 = -1
        v5 = -1
        v6 = -1
        for VehID in self.VehicleIds:
            parameters[i][0] = VehID
            parameters[i][1] = vehicleparameters[VehID][tc.VAR_POSITION][0]
            parameters[i][2] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]
            parameters[i][3] = vehicleparameters[VehID][tc.VAR_POSITION][1]
            i = i + 1
        parameters = sorted(parameters, key=lambda x: x[1])
        # Find Row with Auto Car
        index = [x for x in parameters if self.AutoCarID in x][0]
        RowIDAuto = parameters.index(index)

        # if there are no vehicles in front
        if RowIDAuto == len(self.VehicleIds)-1:
            d1 = -1
            v1 = -1
            d3 = -1
            v3 = -1
            d5 = -1
            v5 = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 100
            if(self.currentTrackingVehId !='None' and (vehicleparameters[self.currentTrackingVehId][tc.VAR_POSITION][0] < vehicleparameters[self.AutoCarID][tc.VAR_POSITION][0])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = 'None'
        else:
            if parameters[RowIDAuto][2] == 0:
                d5 = -1
                v5 = -1
                d6 = -1
                v6 = -1
            elif parameters[RowIDAuto][2] == self.maxLaneNumber:   # ?
                d3 = -1
                v3 = -1
                d4 = -1
                v4 = -1
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    dx1 = parameters[index][1] - parameters[RowIDAuto][1]
                    dy1 = parameters[index][3] - parameters[RowIDAuto][3]
                    d1 = np.sqrt(dx1**2+dy1**2)
                    v1 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d1 = -1
                v1 = -1
                self.CurrFrontVehID = 'None'
                self.CurrFrontVehDistance = 100
            # find d3 and v3
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    dx3 = parameters[index][1] - parameters[RowIDAuto][1]
                    dy3 = parameters[index][3] - parameters[RowIDAuto][3]
                    d3 = np.sqrt(dx3 ** 2 + dy3 ** 2)
                    v3 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d3 = -1
                v3 = -1
            # find d5 and v5
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    dx5 = parameters[index][1] - parameters[RowIDAuto][1]
                    dy5 = parameters[index][3] - parameters[RowIDAuto][3]
                    d5 = np.sqrt(dx5 ** 2 + dy5 ** 2)
                    v5 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d5 = -1
                v5 = -1
            # find d2 and v2
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    dx2 = parameters[index][1] - parameters[RowIDAuto][1]
                    dy2 = parameters[index][3] - parameters[RowIDAuto][3]
                    d2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
                    v2 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            if index < 0:
                d2 = -1
                v2 = -1
            # find d4 and v4
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    dx4 = parameters[index][1] - parameters[RowIDAuto][1]
                    dy4 = parameters[index][3] - parameters[RowIDAuto][3]
                    d4 = np.sqrt(dx4 ** 2 + dy4 ** 2)
                    v4 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            if index < 0:
                d4 = -1
                v4 = -1
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    dx6 = parameters[index][1] - parameters[RowIDAuto][1]
                    dy6 = parameters[index][3] - parameters[RowIDAuto][3]
                    d6 = np.sqrt(dx6 ** 2 + dy6 ** 2)
                    v6 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            if index < 0:
                d6 = -1
                v6 = -1
            # Find if any overtakes has happend
            if (self.currentTrackingVehId != 'None' and (vehicleparameters[self.currentTrackingVehId][tc.VAR_POSITION][0] <vehicleparameters[self.AutoCarID][tc.VAR_POSITION][0])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = parameters[RowIDAuto + 1][0]
        if RowIDAuto == 0: #This means that there is no car behind   # ?
            RearDist = -1
        else: # There is a car behind return the distance between them
            RearDist =  (parameters[RowIDAuto][1] - parameters[RowIDAuto-1][1])
        # Return car in front distance
        if RowIDAuto == len(self.VehicleIds)-1:
            FrontDist = -1
            # Save the current front vehicle Features
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 100
        else:
            FrontDist = (parameters[RowIDAuto+1][1] - parameters[RowIDAuto][1])
            # Save the current front vehicle Features
            self.CurrFrontVehID = parameters[RowIDAuto+1][0]
            self.CurrFrontVehDistance = FrontDist
        return d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6


    def _findstate(self):
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        # find d1,v1,d2,v2,d3,v3,d4,v4, d5, v5, d6, v6
        d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6 = self._findRearVehDistance(VehicleParameters)
        # For Fault Simulation use random generation to generate if a communication fault should occur
        commErrorStatus = False
        if np.random.rand() < self.ErrorPropability:
            commErrorStatus = True
        # check if they are between the limits if they are not then give the maximum possible
        if ((d1 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d1 <= self.CommRange))):
            d1 = self.maxDistanceFrontVeh
        elif d1 < 0: # if there is no vehicle ahead in L0
            d1 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v1 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d1 <= self.CommRange))) : # there is no vehicle ahead in L0 or there is a communication error: # there is no vehicle ahead in L0
            v1 = 0

        if ((d2 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d2 <= self.CommRange))):
            d2 = self.maxDistanceRearVeh
        elif d2 < 0: #There is no vehicle behind in L0
            d2 = self.maxDistanceRearVeh
        if ((v2 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d2 <= self.CommRange))) : # there is no vehicle behind in L0 or there is a communication error
            v2 = 0

        if ((d3 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d3 <= self.CommRange))):
            d3 = self.maxDistanceFrontVeh
        elif d3 < 0: # no vehicle ahead in L1
            d3 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v3 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d3 <= self.CommRange))) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v3 = 0

        if ((d4 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d4 <= self.CommRange))):
            d4 = self.maxDistanceRearVeh
        elif d4 < 0: #There is no vehicle behind in L1
            d4 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v4 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d4 <= self.CommRange))) : # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v4 = 0

        if ((d5 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d5 <= self.CommRange))):
            d5 = self.maxDistanceFrontVeh
        elif d5 < 0: # no vehicle ahead in L1
            d5 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v5 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d5 <= self.CommRange))) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v5 = 0

        if ((d6 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d6 <= self.CommRange))):
            d6 = self.maxDistanceRearVeh
        elif d6 < 0: #There is no vehicle behind in L1
            d6 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v6 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d6 <= self.CommRange))) : # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v6 = 0

        va = VehicleParameters[self.AutoCarID][tc.VAR_SPEED]
        if va > 50 or va < 0:
            va = 0

        angle = traci.vehicle.getAngle(self.AutoCarID)
        yaw_rate = max(angle - self.Pre_angle, 10)

        vacc = traci.vehicle.getAcceleration(self.AutoCarID)
        if abs(vacc) > 10:
            vacc = 0

        return va, v1, d1, v2, d2, v3, d3, v4, d4, v5, d5, v6, d6, max(0, VehicleParameters[self.AutoCarID][tc.VAR_LANE_INDEX]), vacc, yaw_rate

    def obs_to_state(self):
        Vehicle_Params = traci.vehicle.getAllSubscriptionResults()
        obs = self._findstate()
        self.state = [obs[0]/35, obs[1]/35, obs[2]/100, obs[3]/35, obs[4]/100, obs[5]/35, obs[6]/100, obs[7]/35, obs[8]/100, obs[9]/35, obs[10]/100, obs[11]/35, obs[12]/100, obs[13]/10, obs[14]/10, obs[15]/100]

        if Vehicle_Params[self.AutoCarID][tc.VAR_LANE_INDEX] == self.maxLaneNumber:
            self.state[6] = 0
            self.state[8] = 0

        if Vehicle_Params[self.AutoCarID][tc.VAR_LANE_INDEX] == 0:
            self.state[10] = 0
            self.state[12] = 0

        return self.state

    def get_reward_v(self, action_l):
        obs = self.pre_findstate
        v0 = obs[0]
        yaw_rate = obs[15]
        dis_front = obs[2]

        reward_v = v0/35

        if dis_front < 30:
            reward_v = reward_v - 0.1

        if v0 > 30 and abs(yaw_rate)*3.14/180 > 0.85*0.9*9.8/v0:
            reward_v = reward_v - 0.05

        if action_l != 2 and v0 > 20:
            reward_v = reward_v - v0/350

        collision = self.collision_detection(action_l)
        if collision is True:
            reward_v = reward_v - 0.1
        return reward_v

    def collision_detection(self, action):
        obs = self.pre_findstate
        dis_front = obs[2]
        dis_back = obs[4]
        dis_left_f = obs[6]
        dis_left_b = obs[8]
        dis_right_f = obs[10]
        dis_right_b = obs[12]
        Vehicle_Params = traci.vehicle.getAllSubscriptionResults()

        if Vehicle_Params[self.AutoCarID][tc.VAR_LANE_INDEX] == self.maxLaneNumber:
            dis_left_f = 0
            dis_left_b = 0

        if Vehicle_Params[self.AutoCarID][tc.VAR_LANE_INDEX] == 0:
            dis_right_f = 0
            dis_right_b = 0

        if action == 0:
            if dis_right_f < 6 or dis_right_b < 5:
                value = True
            else:
                value = False
        elif action == 1:
            if dis_left_f < 6 or dis_left_b < 5:
                value = True
            else:
                value = False
        else:
            if dis_front < 6 or dis_back < 5:
                value = True
            else:
                value = False

        return value

    def step(self, action_l):
        self.pre_findstate = self._findstate()
        self.pre_angle = traci.vehicle.getAngle(self.AutoCarID)

        # action 0 -> move to right lane
        if action_l == 0:
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            if laneindex != 0:
                traci.vehicle.changeLane(self.AutoCarID, laneindex - 1, 100)
                self.numberOfLaneChanges += 1
        # action 1 change to left Lane 1
        elif action_l == 1:
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            if laneindex != self.maxLaneNumber:
                traci.vehicle.changeLane(self.AutoCarID, laneindex + 1, 100)
                self.numberOfLaneChanges += 1

        traci.simulationStep()
        self.state = self.obs_to_state()

        self.VehicleIds = traci.vehicle.getIDList()
        if self.AutoCarID in traci.vehicle.getIDList():
            # Subscribe for vehicle data as new vehicles may have come into existence
            for VehId in self.VehicleIds:
                traci.vehicle.subscribe(VehId, (tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_LANE_INDEX, tc.VAR_DISTANCE))
                traci.vehicle.subscribeLeader(self.AutoCarID, 50)  # Subscribe the vehicle information of the car in front of Auto
            if VehId == self.AutoCarID:
                speedMode = traci.vehicle.getSpeedMode(self.AutoCarID)
                speedMode = speedMode & int('11000', 2)
            Vehicle_Params = traci.vehicle.getAllSubscriptionResults()
            self.AutocarSpeed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]
            reward_v = self.get_reward_v(action_l)

            self.PrevFrontVehID = self.CurrFrontVehID
            self.PrevFrontVehDistance = self.CurrFrontVehDistance
            DistanceTravelled = Vehicle_Params[self.AutoCarID][tc.VAR_DISTANCE]
        # Condition 4 -> Reward for Collision
        else:
            # change Distance covered in the input state to -1 to so that the learning algorithm can learn that a collision has occured
            self.end = True
            print('AutoCar is not available!')

            reward_v = 0
            DistanceTravelled = 0
        self.TotalReward += reward_v
        return self.state, reward_v, self.end, DistanceTravelled,self.numberOfLaneChanges, self.numberOfOvertakes, {}

    def reset(self):
        self.end = False
        self.TotalReward = 0
        self.numberOfLaneChanges = 0
        self.numberOfOvertakes = 0
        self.currentTrackingVehId = 'None'
        self.PrevFrontVehID = 'None'
        self.PrevFrontVehDistance = 100
        self.Pre_angle = 90

        dom = xml.dom.minidom.parse(config_path)
        root = dom.documentElement
        random_seed_element = root.getElementsByTagName("seed")[0]

        if self.reset_times % 2 == 0:
            random_seed = "%d" % self.reset_times
            random_seed_element.setAttribute("value", random_seed)

        with open(config_path, "w") as file:
            dom.writexml(file)

        traci.load(["-c", config_path])
        print('Resetting the layout')
        traci.simulationStep()
        AutoCarAvailable = False
        self.reset_times += 1

        while AutoCarAvailable == False:
            traci.simulationStep()
            self.VehicleIds = traci.vehicle.getIDList()
            if self.AutoCarID in traci.vehicle.getIDList():
                AutoCarAvailable = True
                self.StartTime = traci.simulation.getCurrentTime()
        self.VehicleIds = traci.vehicle.getIDList()
        # Just check if the auto car still exisits and that there has not been any collision
        for VehId in self.VehicleIds:
            traci.vehicle.subscribe(VehId, (tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_LANE_INDEX, tc.VAR_DISTANCE))
            traci.vehicle.subscribeLeader(self.AutoCarID, 50)  # Subscribe the vehicle information of the car in front of Auto
            # traci.vehicle.setSpeedMode(VehId,0) #Disable Car Following
            if VehId == self.AutoCarID:
                speedMode = traci.vehicle.getSpeedMode(VehId)
                speedMode = speedMode & int('11000', 2)
                # traci.vehicle.subscribeLeader(self.AutoCarID,50)  # Subscribe the vehicle information of the car in front of Auto
                traci.vehicle.setLaneChangeMode(VehId, 0)  # Disable automatic lane changing
        traci.simulationStep()
        self.state = self.obs_to_state()

        return self.state

    def close(self):
        traci.close()

    def start(self, gui=False):
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        traci.start([sumoBinary, "-c", config_path])





from flow.controllers.car_following_models import SumoCarFollowingController
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SumoLaneChangeController
import collections
import logging

from copy import deepcopy
import traci.constants as tc

from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams


class Vehicles:

    speed_modes = {"aggressive": 0, "no_collide": 1}
    lane_changing_modes = {"aggressive": 0, "no_lat_collide": 256,
                           "strategic": 853, "execute_all": 597}

    def __init__(self):
        """
        Base vehicle class used to describe the state of all vehicles in the
        network. State information on the vehicles for a given time step can be
        set or retrieved from this class.
        """
        self.__ids = []  # stores the ids of all vehicles
        self.__human_ids = []  # stores the ids of human-driven vehicles
        self.__controlled_ids = []  # stores the ids of flow-controlled vehicles
        self.__controlled_lc_ids = []  # ids of flow lc-controlled vehicles
        self.__rl_ids = []  # stores the ids of rllab-controlled vehicles

        # vehicles: Key = Vehicle ID, Value = Dictionary describing the vehicle
        # Ordered dictionary used to keep neural net inputs in order
        self.__vehicles = collections.OrderedDict()

        # create a sumo_observations variable that will carry all information
        # on the state of the vehicles for a given time step
        self.__sumo_observations = None

        self.num_vehicles = 0  # total number of vehicles in the network
        self.num_rl_vehicles = 0  # number of rl vehicles in the network
        self.num_types = 0  # number of unique types of vehicles in the network
        self.types = []  # types of vehicles in the network
        self.initial_speeds = []  # speed of vehicles at the start of a rollout

        # contains the parameters associated with each type of vehicle
        self.type_parameters = dict()

        # contain the minGap attribute of each type of vehicle
        self.minGap = dict()

    def add_vehicles(self,
                     veh_id,
                     acceleration_controller=(SumoCarFollowingController, {}),
                     lane_change_controller=(SumoLaneChangeController, {}),
                     routing_controller=None,
                     initial_speed=0,
                     num_vehicles=1,
                     speed_mode='no_collide',
                     custom_speed_mode=None,
                     lane_change_mode="no_lat_collide",
                     custom_lane_change_mode=256,
                     sumo_car_following_params=None,
                     sumo_lc_params=None):
        """
        Adds a sequence of vehicles to the list of vehicles in the network.

        Parameters
        ----------
        veh_id: str
            base vehicle ID for the vehicles (will be appended by a number)
        acceleration_controller: tup, optional
            1st element: flow-specified acceleration controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        lane_change_controller: tup, optional
            1st element: flow-specified lane-changer controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        routing_controller: tup, optional
            1st element: flow-specified routing controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        initial_speed: float, optional
            initial speed of the vehicles being added (in m/s)
        num_vehicles: int, optional
            number of vehicles of this type to be added to the network
        speed_mode: str, optional
            may be one of the following:
             - "no_collide" (default): Human and RL cars are preventing from
               reaching speeds that may cause crashes (also serves as a failsafe).
             - "aggressive": Human and RL cars are not limited by sumo with regard
               to their accelerations, and can crash longitudinally
             - "custom": uses the number provided with custom_speed_mode,
        custom_speed_mode: int, optional:
            custom speed mode for the given vehicles. specified at:
            http://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#speed_mode_.280xb3.29
        lane_change_mode: str, optional
            - "strategic": Human cars make lane changes in accordance with SUMO to
                provide speed boosts
            - "no_lat_collide": Human cars will not make lane changes, RL cars can
                lane change into any space, no matter how likely it is to crash (default)
            - "aggressive": RL cars are not limited by sumo with regard to their
                lane-change actions, and can crash longitudinally
        custom_lane_change_mode: int, optional:
            custom speed mode for the given vehicles. specified at:
            http://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#lane_change_mode_.280xb6.29
        sumo_car_following_params:
            Params object specifying attributes for Sumo CFM.
        sumo_lc_params:
            Params object specifying attributes for Sumo lane changing model.
        """
        if sumo_car_following_params is None:
            sumo_car_following_params = SumoCarFollowingParams()

        if sumo_lc_params is None:
            sumo_lc_params = SumoLaneChangeParams()

        type_params = {}
        type_params.update(sumo_car_following_params.controller_params)
        type_params.update(sumo_lc_params.controller_params)

        # If the vehicle is not a sumo vehicle, set its max acceleration /
        # deceleration to a very large value to allow fuller control
        if acceleration_controller[0] != SumoCarFollowingController:
            type_params["accel"] = 1000
            type_params["decel"] = 1000

        # If a vehicle is not sumo or RL, let the minGap be as small as possible
        # so that it does not tamper with the dynamics of the controller
        if acceleration_controller[0] != SumoCarFollowingController \
                and acceleration_controller[0] != RLController:
            type_params["minGap"] = 0.0

        # this dict will be used when trying to introduce new vehicles into
        # the network via a flow
        self.type_parameters[veh_id] = \
            {"acceleration_controller": acceleration_controller,
             "lane_change_controller": lane_change_controller,
             "routing_controller": routing_controller,
             "initial_speed": initial_speed,
             "speed_mode": speed_mode,
             "custom_speed_mode": custom_speed_mode,
             "lane_change_mode": lane_change_mode,
             "custom_lane_change_mode": custom_lane_change_mode,
             "sumo_car_following_params": sumo_car_following_params,
             "sumo_lc_params": sumo_lc_params}

        # this is used to return the actual headways from the vehicles class
        self.minGap[veh_id] = type_params["minGap"]

        for i in range(num_vehicles):
            vehID = veh_id + '_%d' % i

            # add the vehicle to the list of vehicle ids
            self.__ids.append(vehID)

            self.__vehicles[vehID] = dict()

            # specify the type
            self.__vehicles[vehID]["type"] = veh_id

            # specify the acceleration controller class
            self.__vehicles[vehID]["acc_controller"] = \
                acceleration_controller[0](veh_id=vehID,
                                           **acceleration_controller[1])

            # specify the lane-changing controller class
            self.__vehicles[vehID]["lane_changer"] = \
                lane_change_controller[0](veh_id=vehID,
                                          **lane_change_controller[1])

            # specify the routing controller class
            if routing_controller is not None:
                self.__vehicles[vehID]["router"] = \
                    routing_controller[0](veh_id=vehID,
                                          router_params=routing_controller[1])
            else:
                self.__vehicles[vehID]["router"] = None

            # specify the speed of vehicles at the start of a rollout
            self.__vehicles[vehID]["initial_speed"] = initial_speed

            # check if the vehicle is human-driven or autonomous
            if acceleration_controller[0] == RLController:
                self.__rl_ids.append(vehID)
            else:
                self.__human_ids.append(vehID)

                # check if the vehicle's lane-changing / acceleration actions
                # are controlled by sumo or not.
                if acceleration_controller[0] != SumoCarFollowingController:
                    self.__controlled_ids.append(vehID)
                if lane_change_controller[0] != SumoLaneChangeController:
                    self.__controlled_lc_ids.append(vehID)

            # set the speed mode for the vehicle
            self.__vehicles[vehID]["speed_mode_name"] = speed_mode
            if speed_mode in Vehicles.speed_modes:
                self.__vehicles[vehID]["speed_mode_value"] = Vehicles.speed_modes[speed_mode]
            elif speed_mode == "custom":
                self.__vehicles[vehID]["speed_mode_value"] = custom_speed_mode
            else:
                logging.error("Invalid speed mode!")
                self.__vehicles[vehID]["speed_mode_value"] = Vehicles.speed_modes["no_collide"]

            # set the lane changing mode for the vehicle
            self.__vehicles[vehID]["lane_change_mode_name"] = lane_change_mode
            if lane_change_mode in Vehicles.lane_changing_modes:
                self.__vehicles[vehID]["lane_change_mode_value"] = Vehicles.lane_changing_modes[lane_change_mode]
            elif lane_change_mode == "custom":
                self.__vehicles[vehID]["lane_change_mode_value"] = custom_lane_change_mode
            else:
                logging.error("Invalid lane change mode!")
                self.__vehicles[vehID]["lane_change_mode_value"] = Vehicles.speed_modes["no_lat_collide"]

        # update the variables for the number of vehicles in the network
        self.num_vehicles = len(self.__ids)
        self.num_rl_vehicles = len(self.__rl_ids)

        # increase the number of unique types of vehicles in the network, and
        # add the type to the list of types
        self.num_types += 1
        self.types.append((veh_id, type_params))

    def set_sumo_observations(self, sumo_observations, veh_ids, env,
                              tele_ids=None):

        if tele_ids is None:
            tele_ids = []

        if env.time_counter == 0:
            for veh_id in self.__ids:
                # update the sumo observations variable
                self.__sumo_observations = deepcopy(sumo_observations)

                # set the initial last_lc
                self.set_state(
                    veh_id, "last_lc", -1 * env.lane_change_duration)

                # set the initial absolute_position
                self.set_absolute_position(veh_id, env.get_x_by_id(veh_id))
        else:
            # check for exiting vehicles (vehicles in self.__ids but not in
            # veh_ids)
            for veh_id in list(set(self.__ids) - set(veh_ids)):
                if veh_id not in tele_ids:
                    self.remove_vehicle(veh_id)

            # check for entering vehicles (vehicles in veh_ids but not in
            # self.__ids)
            for veh_id in list(set(veh_ids) - set(self.__ids)):
                veh_type = env.traci_connection.vehicle.getTypeID(veh_id)
                self.add_inflow_vehicle(veh_id, veh_type, env)

            for veh_id in self.__ids:
                # update the "last_lc" variable
                prev_lane = self.get_lane(veh_id)
                if sumo_observations[veh_id][tc.VAR_LANE_INDEX] != \
                        prev_lane and veh_id in self.__rl_ids:
                    self.set_state(veh_id, "last_lc", env.time_counter)

                # update the "absolute_position" variable
                prev_pos = env.get_x_by_id(veh_id)
                this_edge = sumo_observations[veh_id][tc.VAR_ROAD_ID]
                this_pos = sumo_observations[veh_id][tc.VAR_LANEPOSITION]
                try:
                    change = env.scenario.get_x(this_edge, this_pos) - prev_pos
                    if change < 0:
                        change += env.scenario.length
                    new_abs_pos = self.get_absolute_position(
                        veh_id) + change
                    self.set_absolute_position(veh_id, new_abs_pos)
                except (ValueError, TypeError):
                    self.set_absolute_position(veh_id, -1001)

                # update the "headway", "leader", and "follower" variables
                try:
                    headway = sumo_observations[veh_id][tc.VAR_LEADER]
                    vtype = self.get_state(veh_id, "type")
                    minGap = self.minGap[vtype]
                    if headway is None:
                        self.__vehicles[veh_id]["leader"] = None
                        self.__vehicles[veh_id]["follower"] = None
                        self.__vehicles[veh_id]["headway"] = 1e-3
                    else:
                        self.__vehicles[veh_id]["headway"] = headway[1] + minGap
                        self.__vehicles[veh_id]["leader"] = headway[0]
                        self.__vehicles[headway[0]]["follower"] = veh_id
                except KeyError:
                    # this is used to deal with the absence of network
                    # observations upon reset. It only applies for the very
                    # first time step
                    self.__vehicles[veh_id]["leader"] = None
                    self.__vehicles[veh_id]["follower"] = None
                    self.__vehicles[veh_id]["headway"] = 1e-3

            # update the sumo observations variable
            self.__sumo_observations = deepcopy(sumo_observations)

    def add_inflow_vehicle(self, veh_id, veh_type, env):
        """
        Adds a vehicle that entered the network from an inflow.
        """
        if veh_type not in self.type_parameters:
            raise KeyError("Entering vehicle is not a valid type.")

        self.num_vehicles += 1
        self.__ids.append(veh_id)
        self.__vehicles[veh_id] = dict()

        # specify the type
        self.__vehicles[veh_id]["type"] = veh_type

        # specify the acceleration controller class
        accel_controller = self.type_parameters[veh_type]["acceleration_controller"]
        self.__vehicles[veh_id]["acc_controller"] = \
            accel_controller[0](veh_id=veh_id, **accel_controller[1])

        # specify the lane-changing controller class
        lc_controller = self.type_parameters[veh_type]["lane_change_controller"]
        self.__vehicles[veh_id]["lane_changer"] = \
            lc_controller[0](veh_id=veh_id, **lc_controller[1])

        # specify the routing controller class
        rt_controller = self.type_parameters[veh_type]["routing_controller"]
        if rt_controller is not None:
            self.__vehicles[veh_id]["router"] = \
                rt_controller[0](veh_id=veh_id, router_params=rt_controller[1])
        else:
            self.__vehicles[veh_id]["router"] = None

        # check if the vehicle is human-driven or autonomous
        if accel_controller[0] == RLController:
            self.__rl_ids.append(veh_id)
        else:
            self.__human_ids.append(veh_id)

            # check if the vehicle's lane-changing / acceleration actions
            # are controlled by sumo or not.
            if accel_controller[0] != SumoCarFollowingController:
                self.__controlled_ids.append(veh_id)
            if lc_controller[0] != SumoLaneChangeController:
                self.__controlled_lc_ids.append(veh_id)

        # subscribe the new vehicle
        env.traci_connection.vehicle.subscribe(
            veh_id, [tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION,
                     tc.VAR_ROAD_ID, tc.VAR_SPEED])
        env.traci_connection.vehicle.subscribeLeader(veh_id, 2000)

        # set the speed mode for the vehicle
        speed_mode = self.type_parameters[veh_type]["speed_mode"]
        custom_speed_mode = self.type_parameters[veh_type]["custom_speed_mode"]
        self.__vehicles[veh_id]["speed_mode_name"] = speed_mode
        if speed_mode in Vehicles.speed_modes:
            self.__vehicles[veh_id]["speed_mode_value"] = Vehicles.speed_modes[
                speed_mode]
        elif speed_mode == "custom":
            self.__vehicles[veh_id]["speed_mode_value"] = custom_speed_mode
        else:
            logging.error("Invalid speed mode!")
            self.__vehicles[veh_id]["speed_mode_value"] = Vehicles.speed_modes[
                "no_collide"]

        # set the absolute position of the vehicle
        self.set_absolute_position(veh_id, 0)

        # set the absolute position of the vehicle
        self.set_state(veh_id, "last_lc", env.time_counter)

        # set the absolute position of the vehicle
        self.set_route(
            veh_id, "route" + env.traci_connection.vehicle.getRoadID(veh_id))

        # set speed mode in sumo
        env.traci_connection.vehicle.setSpeedMode(
            veh_id, self.get_speed_mode(veh_id))

        # set the lane changing mode for the vehicle
        lc_mode = self.type_parameters[veh_type]["lane_change_mode"]
        custom_lc_mode = self.type_parameters[veh_type]["custom_lane_change_mode"]
        self.__vehicles[veh_id]["lane_change_mode_name"] = lc_mode
        if lc_mode in Vehicles.lane_changing_modes:
            self.__vehicles[veh_id]["lane_change_mode_value"] = \
                Vehicles.lane_changing_modes[lc_mode]
        elif lc_mode == "custom":
            self.__vehicles[veh_id]["lane_change_mode_value"] = \
                custom_lc_mode
        else:
            logging.error("Invalid lane change mode!")
            self.__vehicles[veh_id]["lane_change_mode_value"] = \
                Vehicles.speed_modes["no_lat_collide"]

        # set lane change mode in sumo
        env.traci_connection.vehicle.setLaneChangeMode(
            veh_id, self.get_lane_change_mode(veh_id))

        # change the color of the vehicle based on its type
        env.traci_connection.vehicle.setColor(veh_id, env.colors[veh_type])

    def remove_vehicle(self, veh_id):
        """
        Removes a vehicle from the vehicles class and all valid ID lists, and
        decrements the total number of vehicles in this class.

        Parameters
        ----------
        veh_id: str
            unique identifier of th vehicle to be removed
        """
        del self.__vehicles[veh_id]
        self.__ids.remove(veh_id)
        self.num_vehicles -= 1

        # remove it from all other ids (if it is there)
        if veh_id in self.__human_ids:
            self.__human_ids.remove(veh_id)
            if veh_id in self.__controlled_ids:
                self.__controlled_ids.remove(veh_id)
            if veh_id in self.__controlled_lc_ids:
                self.__controlled_lc_ids.remove(veh_id)
        else:
            self.__rl_ids.remove(veh_id)

    def set_absolute_position(self, veh_id, absolute_position):
        self.__vehicles[veh_id]["absolute_position"] = absolute_position

    def set_route(self, veh_id, route):
        self.__vehicles[veh_id]["route"] = route

    def set_leader(self, veh_id, leader):
        self.__vehicles[veh_id]["leader"] = leader

    def set_follower(self, veh_id, follower):
        self.__vehicles[veh_id]["follower"] = follower

    def set_headway(self, veh_id, headway):
        self.__vehicles[veh_id]["headway"] = headway

    def get_ids(self):
        return self.__ids

    def get_human_ids(self):
        return self.__human_ids

    def get_controlled_ids(self):
        return self.__controlled_ids

    def get_controlled_lc_ids(self):
        return self.__controlled_lc_ids

    def get_rl_ids(self):
        return self.__rl_ids

    def get_initial_speed(self, veh_id):
        return self.__vehicles[veh_id]["initial_speed"]

    def get_lane_change_mode(self, veh_id):
        return self.__vehicles[veh_id]["lane_change_mode_value"]

    def get_lane_change_mode_name(self, veh_id):
        return self.__vehicles[veh_id]["lane_change_mode_name"]

    def get_speed_mode(self, veh_id):
        return self.__vehicles[veh_id]["speed_mode_value"]

    def get_speed_mode_name(self, veh_id):
        return self.__vehicles[veh_id]["speed_mode_name"]

    def get_speed(self, veh_id="all"):
        """
        Return the speed of the specified vehicle at the current time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        # if a list of vehicle ids are requested, call the function for each
        # requested vehicle id
        if not isinstance(veh_id, str):
            return [self.get_speed(vehID) for vehID in veh_id]
        elif veh_id == "all":
            return [self.get_speed(vehID) for vehID in self.__ids]

        # perform the value retrieval for a specific vehicle
        try:
            return self.__sumo_observations[veh_id][tc.VAR_SPEED]
        except KeyError:
            # if the vehicle does not exist, return an error value (-1001)
            return -1001

    def get_absolute_position(self, veh_id="all"):
        """
        Return the absolute position of the specified vehicle at the current
        time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if not isinstance(veh_id, str):
            return [self.__vehicles[vehID]["absolute_position"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["absolute_position"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["absolute_position"]

    def get_position(self, veh_id="all"):
        """
        Returns the position of the specified vehicle (relative to the current
        edge) at the current time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        # if a list of vehicle ids are requested, call the function for each
        # requested vehicle id
        if not isinstance(veh_id, str):
            return [self.get_position(vehID) for vehID in veh_id]
        elif veh_id == "all":
            return [self.get_position(vehID) for vehID in self.__ids]

        # perform the value retrieval for a specific vehicle
        try:
            return self.__sumo_observations[veh_id][tc.VAR_LANEPOSITION]
        except KeyError:
            # if the vehicle does not exist, return an error value (-1001)
            return -1001

    def get_edge(self, veh_id="all"):
        """
        Returns the position of the specified vehicle (relative to the current
        edge) at the current time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        # if a list of vehicle ids are requested, call the function for each
        # requested vehicle id
        if not isinstance(veh_id, str):
            return [self.get_edge(vehID) for vehID in veh_id]
        elif veh_id == "all":
            return [self.get_edge(vehID) for vehID in self.__ids]

        # perform the value retrieval for a specific vehicle
        try:
            return self.__sumo_observations[veh_id][tc.VAR_ROAD_ID]
        except KeyError:
            # if the vehicle does not exist, return an empty edge value
            return ""

    def get_lane(self, veh_id="all"):
        """
        Returns the lane index of the specified vehicle at the current time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        # if a list of vehicle ids are requested, call the function for each
        # requested vehicle id
        if not isinstance(veh_id, str):
            return [self.get_lane(vehID) for vehID in veh_id]
        elif veh_id == "all":
            return [self.get_lane(vehID) for vehID in self.__ids]

        # perform the value retrieval for a specific vehicle
        try:
            return self.__sumo_observations[veh_id][tc.VAR_LANE_INDEX]
        except KeyError:
            # if the vehicle does not exist, return an error value (-1001)
            return -1001

    def get_acc_controller(self, veh_id="all"):
        """
        Returns the acceleration controller of the specified vehicle.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if not isinstance(veh_id, str):
            return [self.__vehicles[vehID]["acc_controller"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["acc_controller"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["acc_controller"]

    def get_lane_changing_controller(self, veh_id="all"):
        """
        Returns the lane changing controller of the specified vehicle.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if not isinstance(veh_id, str):
            return [self.__vehicles[vehID]["lane_changer"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["lane_changer"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["lane_changer"]

    def get_routing_controller(self, veh_id="all"):
        """
        Returns the routing controller of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if not isinstance(veh_id, str):
            return [self.__vehicles[vehID]["router"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["router"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["router"]

    def get_route(self, veh_id="all"):
        """
        Returns the route of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if not isinstance(veh_id, str):
            return [self.__vehicles[vehID]["route"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["route"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["route"]

    def get_leader(self, veh_id="all"):
        """
        Returns the leader of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if not isinstance(veh_id, str):
            return [self.__vehicles[vehID]["leader"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["leader"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["leader"]

    def get_follower(self, veh_id="all"):
        """
        Returns the follower of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if not isinstance(veh_id, str):
            return [self.__vehicles[vehID]["follower"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["follower"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["follower"]

    def get_headway(self, veh_id="all"):
        """
        Returns the headway of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if not isinstance(veh_id, str):
            return [self.__vehicles[vehID]["headway"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["headway"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["headway"]

    # TODO: everything past here must be thought through
    def set_state(self, veh_id, state_name, state):
        """
        Generic set function. Updates the state *state_name* of the vehicle with
        id *veh_id* with the value *state*.
        """
        self.__vehicles[veh_id][state_name] = state

    def get_state(self, veh_id, state_name):
        """
        Generic get function. Returns the value of *state_name* of the specified
        vehicles at the current time step.

        veh_id may be:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if not isinstance(veh_id, str):
            return [self.__vehicles[vehID][state_name] for vehID in veh_id]
        if veh_id == "all":
            return [self.__vehicles[vehID][state_name] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id][state_name]

    def get_full_state(self, veh_id):
        """
        Return a dict of all state variables of a specific vehicle:
        """
        # FIXME: add sumo observations as well
        return self.__vehicles[veh_id]
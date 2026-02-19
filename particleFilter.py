from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from .pf_base import PFLocaliserBase
import copy
from .util import rotateQuaternion, getHeading
import numpy as np
import math
import heapq
from itertools import combinations as combinations
from collections import deque


class PFLocaliser(PFLocaliserBase):
    def __init__(self, logger, clock):
        # ----- Call the superclass constructor
        super().__init__(logger, clock)

        # Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.4  # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.4  # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.4  # Odometry model y axis (side-to-side) noise
        self.scan = None

        # Size of particle cloud
        self.SIZE = 1000
        self.NOISE_AMPLIFICATION_CARTESIAN = 0.1
        self.NOISE_AMPLIFICATION_YAW = 0.1
        self.P_RANDOM = 0.1

        # Weighted moving averages of total weight of particle cloud at one frame
        self.fast_ema_param = 0.001 * self.SIZE
        self.slow_ema_param = 0.1 * self.SIZE
        self.fast_ema = deque()
        self.slow_ema = deque()

        # Highest weighted particle after recent particle cloud update
        self.maxWeightParticle = None

        # Distance threshold to merge clusters in K-means++
        self.mergeThreshold = 5
        self.NUM_CLUSTERS = 10

        # Difference in yaw influence in K means cluster
        self.diffYawInfluence = 1

        # Store particles and their weighting for later retrieval
        self.particleWeights = {}
        self.particleWeightsCleared = False
        self.totalWeight = 0

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 40  # Number of readings to predict

        # Instantiate random number generator
        self.rndGen = np.random.default_rng()

    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        :Args:
        | initialpose: the initial pose estimate
        :Return:
        | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        poseArray = PoseArray()
        poseArray.header.frame_id = "initial particle cloud"
        position = initialpose.pose.pose.position  # Point message: x, y, z
        orientation = initialpose.pose.pose.orientation  # Quaternion: x, y, z, w

        # Generate random numbers:
        meanPosition = [position.x, position.y]
        covPosition = [[1, 0], [0, 1]]
        meanOrientation = getHeading(orientation)
        stdOrientation = 0.3  # radians

        randomCartesianNoise = self.rndGen.multivariate_normal(
            meanPosition, covPosition, size=self.SIZE, method="cholesky"
        )  # faster matrix algorithm
        randomYaws = self.rndGen.normal(
            loc=meanOrientation, scale=stdOrientation, size=self.SIZE
        )
        rotatedQuaternions = [
            rotateQuaternion(orientation, randomYaw * self.NOISE_AMPLIFICATION_YAW)
            for randomYaw in randomYaws
        ]

        poseArray.poses = [
            Pose(
                position=Point(
                    x=float(
                        rndPosition[0]
                        + position.x * self.NOISE_AMPLIFICATION_CARTESIAN
                    ),
                    y=float(
                        rndPosition[1]
                        + position.y * self.NOISE_AMPLIFICATION_CARTESIAN
                    ),
                    z=0.0,
                ),
                orientation=rndOrientation,
            )
            for rndPosition, rndOrientation in zip(
                randomCartesianNoise, rotatedQuaternions
            )
        ]
        return poseArray

    def injectRandomParticles(self):
        averageWeight = self.totalWeight / len(self.particlecloud.poses)

        if len(self.fast_ema) >= self.fast_ema_param:
            self.fast_ema.popleft()
        self.fast_ema.append(averageWeight)

        if len(self.slow_ema) >= self.slow_ema_param:
            self.slow_ema.popleft()
        self.slow_ema.append(averageWeight)

        if len(self.slow_ema) < self.slow_ema_param:
            self.slow_ema.append(averageWeight)
            self._logger.info("Adding element to slow ema")
        if len(self.fast_ema) < self.fast_ema_param:
            self.fast_ema.append(averageWeight)
            self._logger.info("Adding element to fast ema")

        num_random = int(self.SIZE * self.P_RANDOM)
        self._logger.info(
            f"number of random particles to be injected: {num_random}. "
            f"Probability of random particles: {self.P_RANDOM}"
        )

        if len(self.slow_ema) >= self.slow_ema_param and len(self.fast_ema) >= self.fast_ema_param:
            self._logger.info("INJECTING RANDOM PARTICLES")

            ### Inject particles from occupancy map selected uniformly at random
            baseQuaternion = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            mapWidth = float(self.sensor_model.map_resolution * self.sensor_model.map_width)
            mapHeight = float(self.sensor_model.map_height * self.sensor_model.map_resolution)

            # initialise base quaternion for generating random quaternions
            randomYaws = self.rndGen.uniform(0, 2 * math.pi, size=num_random)
            rotatedQuaternions = [
                rotateQuaternion(baseQuaternion, randomYaw) for randomYaw in randomYaws
            ]
            self.particlecloud.poses.extend(
                [
                    Pose(
                        position=Point(
                            x=self.rndGen.uniform(0, mapWidth),
                            y=self.rndGen.uniform(0, mapHeight),
                        ),
                        orientation=rndQuaternion,
                    )
                    for rndQuaternion in rotatedQuaternions
                ]
            )

    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        :Args:
        | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

        """
        self.scan = scan
        selectedParticles = []
        particleCloud = self.particlecloud
        self.particleWeights.clear()

        if len(self.slow_ema) >= self.slow_ema_param and len(self.fast_ema) >= self.fast_ema_param:
            self._logger.info(f"Setting P_RANDOM: {self.P_RANDOM}")
            fast_av = sum(self.fast_ema) / len(self.fast_ema)
            slow_av = sum(self.slow_ema) / len(self.slow_ema)
            self.P_RANDOM = max(0.1, 1.0 - fast_av / slow_av)  # else self.P_RANDOM initialised to 0.1

        num_keep = max(0.1, self.SIZE * (1 - self.P_RANDOM))
        particleSpacing = 1 / self.SIZE

        self.injectRandomParticles()

        # Stochastic Universal Sampling
        self.totalWeight = 0

        # Get max weight particle for pose estimation K means centroid initialisation
        maxWeight = 0
        for particle in particleCloud.poses:
            likelihoodWeighting = self.sensor_model.get_weight(scan, particle)
            if likelihoodWeighting > maxWeight:
                maxWeight = likelihoodWeighting
                self.maxWeightParticle = particle
            self.totalWeight += likelihoodWeighting
            self.particleWeights[str(particle)] = (likelihoodWeighting, particle)

        # Create a temporary dictionary for the NEW particles
        new_particle_weights = {}
        startPoint = self.rndGen.uniform(0, particleSpacing)
        cumulativeSum = 0
        pointer = startPoint

        for particle, tup in self.particleWeights.items():
            probability = tup[0] / self.totalWeight
            while pointer < cumulativeSum + probability:
                new_p = Pose()
                new_p.position.x = tup[1].position.x
                new_p.position.y = tup[1].position.y
                new_p.position.z = tup[1].position.z
                new_p.orientation.x = tup[1].orientation.x
                new_p.orientation.y = tup[1].orientation.y
                new_p.orientation.z = tup[1].orientation.z
                new_p.orientation.w = tup[1].orientation.w
                selectedParticles.append(new_p)

                weight = tup[0]
                new_particle_weights[str(new_p)] = (weight, new_p)

                pointer += particleSpacing
            cumulativeSum += probability

        self.particleWeights = new_particle_weights
        self.particlecloud.poses = selectedParticles
        return selectedParticles

    # Calculates difference in yaw
    def diffYaw(self, point1, point2):
        point1Yaw = getHeading(point1.orientation)
        point2Yaw = getHeading(point2.orientation)
        diff = point1Yaw - point2Yaw
        return abs(np.arctan2(np.sin(diff), np.cos(diff)))

    def euclideanNorm(self, point1, point2):  # point1 and point2 are vectors
        return math.hypot(
            point1.position.x - point2.position.x,
            point1.position.y - point2.position.y,
        )

    def get_weight(self, particle):
        if str(particle) in self.particleWeights:
            return self.particleWeights[str(particle)][0]
        else:
            return self.sensor_model.get_weight(self.scan, particle)

    # Designed to intake a particle cloud and return a Pose[]. Weighted K-means++ approach
    def KmeansPlusPlus(self, particleCloud, intialCentroid, K):  # K = number of clusters, data
        self._logger.info(f"--- Starting KmeansPlusPlus with K={K} ---")
        centroids = {str(intialCentroid): intialCentroid}

        while len(centroids) < K:
            self._logger.info(f"Seeding centroid #{len(centroids) + 1}...")
            distancesMaxHeap = []

            for point in particleCloud.poses:
                minDistance = 1000000000000
                minDistanceCentroid = None

                for centroid in centroids.values():
                    distance = self.euclideanNorm(centroid, point)
                    if distance < minDistance:
                        minDistance = distance
                        minDistanceCentroid = centroid

                diffYaw = self.diffYaw(minDistanceCentroid, point)

                # Debug log if key is missing (common crash point)
                if str(point) not in self.particleWeights:
                    self._logger.error(f"Point missing from weights: {str(point)}")

                weight = self.particleWeights[str(point)][0]
                heapq.heappush(
                    distancesMaxHeap,
                    (
                        -(minDistance * weight + diffYaw * self.diffYawInfluence),
                        weight + self.rndGen.uniform(0, 0.1),
                        point,
                    ),
                )

            newCentroid = heapq.heappop(distancesMaxHeap)[2]
            centroids[str(newCentroid)] = newCentroid
            self._logger.info(f"Found new centroid. Total centroids: {len(centroids)}")

        self._logger.info("Seeding complete. Entering merge loop.")
        loop_count = 0

        while True:
            loop_count += 1
            combinationsPoints = list(combinations(centroids.values(), 2))
            merged = False

            if len(combinationsPoints) > 1:
                for centroid1, centroid2 in combinationsPoints:
                    dist_metric = (
                        self.euclideanNorm(centroid1, centroid2)
                        + self.diffYaw(centroid1, centroid2) * self.diffYawInfluence
                    )
                    if dist_metric < self.mergeThreshold:
                        self._logger.info(
                            f"Merging centroids! Distance {dist_metric:.2f} < Threshold {self.mergeThreshold}"
                        )
                        mergedCentroid = self.weightedAverageMean([centroid1, centroid2])
                        centroids.pop(str(centroid1))
                        centroids.pop(str(centroid2))
                        centroids[str(mergedCentroid)] = mergedCentroid
                        merged = True
                        break

                if not merged:
                    self._logger.info("No more merges possible. Exiting merge loop.")
                    break
            else:
                self._logger.info("Less than 2 centroids remaining. Exiting.")
                break

        self._logger.info(
            f"--- KmeansPlusPlus Finished. Returning {len(centroids)} centroids ---"
        )
        return list(centroids.values())

    def weightedAverageMean(self, particleCloud):
        weights = [self.particleWeights[str(particle)][0] for particle in particleCloud]
        coords = [
            [
                particle.position.x,
                particle.position.y,
                particle.orientation.x,
                particle.orientation.y,
                particle.orientation.z,
                particle.orientation.w,
            ]
            for particle in particleCloud
        ]
        averages = np.average(coords, weights=weights, axis=0)

        q_avg = averages[2:]
        v_mag = np.linalg.norm(q_avg)
        if v_mag > 0:
            q_avg /= v_mag

        new_pose = Pose(
            position=Point(x=averages[0], y=averages[1]),
            orientation=Quaternion(x=q_avg[0], y=q_avg[1], z=q_avg[2], w=q_avg[3]),
        )

        # We use Option B to match your original intent:
        new_weight = self.sensor_model.get_weight(self.scan, new_pose)

        # STORE IT so lookup doesn't fail later
        self.particleWeights[str(new_pose)] = (new_weight, new_pose)

        return new_pose

    # TODO: Implement ISODATA inspired version of K-means++ by starting with high number of clusters and iterating down to achieve an adaptive clustering algorithm.
    # (reference: http://www.wu.ece.ufl.edu/books/EE/communications/UnsupervisedClassification.html)
    # - Merge clusters with distance between centroids < 1m, only maintain valid hypothesis. (Pose estimate is centroid with highest weight)
    # TODO: Implement Adaptive Augmented Monte Carlo Localisation, a hybrid algorithm with key ideas taken from Augemnted-MCL and KLD-Sampling (Adaptive MCL).
    # - add noise when particle efficiency ratio is low (slides)
    # - add noise in line with moving average signal

    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        Naive implementation: Returns the particle with the highest weight.
        """
        # 1. Safety check: If no particles exist, return a default Pose to prevent crashing
        if not self.particleWeights:
            return Pose()

        # 2. Iterate through the dictionary to find the highest weight
        best_pose = None
        max_weight = -1.0

        for id_val, (weight, particle) in self.particleWeights.items():
            if weight > max_weight:
                max_weight = weight
                best_pose = particle

        # 3. Fallback: If for some reason best_pose is still None (e.g., all weights -1),
        # return the first available particle
        if best_pose is None:
            return list(self.particleWeights.values())[0][1]

        return best_pose

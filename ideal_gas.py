"""
Created on Mon Nov 21 11:20:23 2016

@author: mlb115
"""

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import random
import operator
import os

with_animation = True

global temp
temp = 300
# T in Kelvin
# This allows the user to deterministically set the temperature of the gas
# temp may vary a little over the course of the simulation as the particles impart energy to the container
# the temperature of the gas is therefore calculated after each frame in temperature_of_gas below

class Ball:

    def __init__(self, mass, position, velocity, radius=.01, color='r'):

        self.m = float(mass)  # kg
        self.r = float(radius)  # m
        self.p = np.array(position, dtype='float')  # m
        self.v = np.array(velocity, dtype='float')  # m/s
        self.col = color

        ''' # Leave this out for code efficiency. Not needed as I make sure to only input valid positions and velocities.
        # Raise error  if position and velocity vectors don't have 2 components
        if len(self.p) != 2:
            raise Exception("Position needs 2 vector components")
        if len(self.v) != 2:
            raise Exception("Velocity needs 2 vector components")
        '''
        self.patch = plt.Circle((self.p[0], self.p[1]), self.r, fc=self.col)

    def __repr__(self):
        return '{0:s}({1:e}, ({2[0]:e}, {2[1]:e}), ({3[0]:e}, {3[1]:e}), {4:e}, "{5:s}")'.format(
            self.__class__.__name__, self.m, self.p, self.v, self.r, self.col)

    def pos(self):
        return self.p

    def vel(self):
        return self.v

    def set_vel(self, v):
        if isinstance(v, (list, tuple)):
            v = np.array(v, dtype='float')
        self.v = v

    def rad(self):
        return self.r

    def mass(self):
        return self.m

    def color(self):
        return self.col

    def get_patch(self):
        return self.patch

    def move(self, dt):
        self.p += dt * self.v
        self.patch.center = self.p

    def time_to_collision(self, other):
        r12 = other.pos() - self.p
        v12 = other.vel() - self.v
        a = np.inner(v12, v12)
        b = 2 * np.inner(r12, v12)
        other_r = other.rad()
        centers_distance = np.linalg.norm(r12)

        if centers_distance + self.r < other_r:  # Ball is inside other ball
            c = np.inner(r12, r12) - (other_r - self.r) ** 2
            times = real_roots(a, b, c)
        elif centers_distance < self.r + other_r:   # Balls are partially intersected.
                            # This should not happen as balls are generated so that they don't intersect
                            # Keep these lines of code anyways for double security
            times = None    # Ignore collision to let the intersected balls loose
        else:  # Collision between two balls
            c = np.inner(r12, r12) - (other_r + self.r) ** 2
            times = real_roots(a, b, c)
        epsilon = 1.0e-21
        if times is None:
            return None
        else:
            dt1, dt2 = times
            if dt1 > epsilon:   # effectively, if dt > 0. The epsilon takes care of the
                                # collision that just took place being counted as the next
                                # collision due to rounding errors.
                return dt1
            elif dt2 > epsilon:
                return dt2
            else:
                return None     # Both collisions occurred in the past

    def collide(self, other):
        r12 = self.p - other.pos()
        v12 = self.v - other.vel()
        unit_ctrs_vector = unit_vector(r12)
        # Component of the relative velocity parallel to the distance vector between the two balls
        parallel_v = np.inner(v12, unit_ctrs_vector) * unit_ctrs_vector
        other_m = other.mass()
        velocity_change = parallel_v * 2 * other_m / (other_m + self.m)
        self.v -= velocity_change
        # Equation for final velocity as result of elastic collision in 2D
        other.set_vel(other.vel() + parallel_v * 2 * self.m / (self.m + other_m))
        momentum_exchanged = np.linalg.norm(velocity_change) * self.m
        return momentum_exchanged


class Gas:

    '''
    Animation adapted from lab script available from
    https://bb.imperial.ac.uk/bbcswebdav/pid-957933-dt-content-rid-3193661_1/courses/COURSE-PHY2_LAB-16_17/
    Y2Computing/Y2PyCourse/Students/Projects/html/Snooker.html
    '''

    def __init__(self, container_rad=5.3e-9, num_balls=100):
        """Initialise the Gas class for N random balls.
        """

        container = Ball(1.0e-20, (0, 0), (0, 0), container_rad, 'w')

        self._balls = [container]  # Container is always the first element of the list
        self.num_balls = num_balls
        self.collision_counter = 0
        self.kb = 1.38e-23    # Boltzmann's constant Nm/K

        # random.seed(12345)    # Debug XXX

        pos_max = 0.6 * container.rad()

        h_mass = 1.67e-27   # Mass of monoatomic hydrogen
        h_rad = 5.3e-11     # Radius of monoatomic hydrogen
        for _ in range(self.num_balls):
            if with_animation:  # Use following values to start simulation with differently-sized balls:
                ball_types = (
                    (h_mass, h_rad, 'r'),
                    (2.25 * h_mass, 1.5 * h_rad, 'g'),
                    (6.25 * h_mass, 2.0 * h_rad, 'b'))  # (mass, radius, color)
            else:  # Use following values to simulate hydrogen atom:
                ball_types = ((h_mass, h_rad, 'r'), )
            m, r, c = random.choice(ball_types)

            # Avoid creating overlapping balls
            attempts_to_place_ball = 20
            for _ in range(attempts_to_place_ball):
                p = random.uniform(-pos_max, pos_max), random.uniform(-pos_max, pos_max)
                if all(np.linalg.norm(p - other_b.pos()) > r + other_b.rad() for other_b in self._balls[1:]):
                    break   # Succeeded placing the ball when all other balls are distant enough from this new ball
            else:   # Failed placing the ball after max number of attempts -- give up
                raise Exception("Cannot place ball -- density is too high")

            v = (0.0, 0.0)
            b = Ball(m, p, v, r, c)
            self._balls.append(b)

        hot_ball = self._balls[-1]  # the last ball is initially the only particle in the gas that has energy
        hot_ball.set_vel((np.sqrt(temp * 2 * self.kb * (len(self._balls) - 1) / hot_ball.mass()), 0.0))
        self.__text0 = None

        # this dictionary will register all the momenta imparted to the container by collision with the Balls
        # this is useful in order to calculate the pressure on the container.
        # See the bottom of page Y2-38 in the lab book for a more detailed derivation
        self.momentum_changes = []

    def init_figure(self):
        """
        Initialise the container diagram and add it to the plot.
        This method is called once by FuncAnimation with no arguments.
        Returns a list or tuple of the 'patches' to be animated.
        """
        # initialise the text to be animated and add it to the plot
        container = self._balls[0]
        container_rad = container.rad()
        self.__text0 = ax.text(-1.3 * container_rad, container_rad, "f={:4d}".format(0, fontsize=12))
        patches = [self.__text0]
        # add the patches for the balls to the plot
        for b in self._balls:
            pch = b.get_patch()
            ax.add_patch(pch)
            patches.append(pch)
        return patches

    def next_frame(self, framenumber):
        """
        Do the next frame of the animation.
        This method is called by FuncAnimation with a single argument
        representing the frame number.
        Returns a list or tuple of the 'patches' being animated.
        """
        # Simulation frame period in the same units as the collision times
        frame_period = 1.0e-13 if with_animation else 5.0e-13   #sec
        current_period = frame_period

        momentum_list_length = 100  # the pressure calculated will be an average over the last momentum_list_length number of frames
        self.momentum_changes.append(0.0)
        if len(self.momentum_changes) > momentum_list_length:
            self.momentum_changes.pop(0)

        container = self._balls[0]

        # Indicates that a collision might occur during the frame and has to be dealt with
        collision_in_frame = True

        while collision_in_frame:
            time, b1, b2 = self.__get_next_collision()
            collision_in_frame = time < current_period
            if collision_in_frame:
                for b in self._balls:
                    b.move(time)
                momentum_exchanged = b1.collide(b2)
                self.collision_counter += 1

                if b1 is container or b2 is container:
                    self.momentum_changes[-1] += momentum_exchanged

                energy = sum(b.mass() * np.inner(b.vel(), b.vel()) / 2 for b in self._balls)
                momentum = sum(b.mass() * b.vel() for b in self._balls)
                #  Check conservation of angular momentum (see lab book pg. Y2-40 for derivation of formula)
                # angular_momentum = sum(b.mass() * np.cross(b.p, b.v) for b in self._balls)  # In kg * m^2 * s^(-1)
                # print('framenumber = {0:d}\tenergy = {1:0.5e}\tmomentum = ({2[0]:0.5e}, {2[1]:0.5e}\tangular momentum = {3:0.5e})'.format(framenumber, energy, momentum, angular_momentum))

                current_period -= time

        gas = self._balls[1:]
        energy_of_gas = sum(b.mass() * np.inner(b.vel(), b.vel()) / 2.0 for b in gas)  # in Joules
        self.temperature_of_gas =  energy_of_gas / (len(gas) * self.kb)
        self.pressure = sum(self.momentum_changes) / len(self.momentum_changes) / (2.0 * np.pi * container.rad()) / (frame_period)
        if not with_animation:
            print(framenumber, self.pressure)
        # print('temperature: {0:0.5e}\tpressure: {1:0.5e}'.format(temperature_of_gas, pressure))

        if with_animation:
            self.__text0.set_text("f = {0:4d}\nT = {1:f} K\nP = {2:f} Pa".format(framenumber, self.temperature_of_gas, self.pressure))
        patches = [self.__text0]

        for b in self._balls:
            b.move(current_period)
            patches.append(b.get_patch())

        return patches

    def __get_next_collision(self):
        """Compute time and balls of the next collision.
        Returns:
            A tuple with the time to the collision and the two balls
        """
        times_to_collision = []
        for i, b2 in enumerate(self._balls):
            for b1 in self._balls[i + 1:]:  # This prevents from calculating each collision twice
                t = b1.time_to_collision(b2)
                if t is not None:
                    times_to_collision.append((t, b1, b2))
        return min(times_to_collision, key=operator.itemgetter(0)) # Compare first item in the tuple which are all the dt's

    # Save file with output
    def write_pressure(self, filename):
        filename = os.path.join('Output', filename)
        f = open(filename, 'a')
        f.write("{0:0.0f}, {1:0.4e}\n".format(temp, self.pressure))
        f.close()

    # Boltzmann distribution
    def write_velocities(self, filename):
        filename = os.path.join('Output', filename)
        f = open(filename, 'a')
        for b in self._balls[1:]:
            f.write("{0:0.4f}\n".format(np.linalg.norm(b.vel())))
        f.close()

    # Method for P vs A graph
    def write_pressure_area(self, filename):
        filename = os.path.join('Output', filename)
        f = open(filename, 'a')
        container = self._balls[0]
        area = np.pi * container.rad() ** 2
        f.write("{0:0.4e}, {1:0.4e}\n".format(area, self.pressure))
        f.close()

    # Method for P vs N graph
    def write_pressure_n(self, filename):
        filename = os.path.join('Output', filename)
        f = open(filename, 'a')
        f.write("{0:d}, {1:0.4e}\n".format(self.num_balls, self.pressure))
        f.close()


# Utility functions
def real_roots(a, b, c):
    """Compute real roots of second-degree equation.
    Arguments:
        a -- real coefficient of 2nd-degree term
        b -- real coefficient of 1st-degree term
        c -- real coefficient of 0th-degree term
    Returns:
        A tuple with the roots sorted in ascending order,
        or None if the equation has no real roots.
    """
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    d = math.sqrt(disc)
    return (-b - d) / (2.0 * a), (-b + d) / (2.0 * a)


def unit_vector(v):
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    if with_animation:
        fig = plt.figure()
        ax = plt.axes(xlim=(-4.5e-9, 4.5e-9), ylim=(-4.5e-9, 4.5e-9))
        ax.axes.set_aspect('equal')

        movie = Gas(container_rad=3.0e-9, num_balls=30)
        anim = animation.FuncAnimation(fig,
                                       movie.next_frame,
                                       init_func=movie.init_figure,
                                       # frames = 1000,
                                       interval=20,
                                       blit=False)  # blit needs to be False if simulation run on Macintosh
        plt.show()

    else:

        # The following are the codes used to produce the graphs on the lab report
        directory = 'Output'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Data for P vs T plot
        for temp in range(50, 1001, 50):
            gas = Gas()
            for framenumber in range(150):
                gas.next_frame(framenumber)
            gas.write_pressure("Graph1.csv")

        # Data for Boltzmann distribution
        for temp in range(50, 551, 250):
            gas = Gas()
            last_counter = 0
            for framenumber in range(50 + 100 * 25 + 1):
                gas.next_frame(framenumber)
                if framenumber % 25 == 0 and framenumber > 50:
                    print(gas.collision_counter - last_counter)
                    last_counter = gas.collision_counter
                    gas.write_velocities("Graph2_T{0:d}.csv".format(temp))

        # Data for P vs A for fixed T=300
        r2 = 5.3e-9 ** 2
        for percentage in range(50, 151, 5):
            r = np.sqrt(r2 * percentage / 100.0)
            gas = Gas(container_rad=r)
            for framenumber in range(150):
                gas.next_frame(framenumber)
            gas.write_pressure_area("Graph3.csv")

        # Data for P vs N for fixed T=300
        for num_balls in range(50, 201, 10):
            gas = Gas(num_balls=num_balls)
            for framenumber in range(150):
                gas.next_frame(framenumber)
            gas.write_pressure_n("Graph4.csv")

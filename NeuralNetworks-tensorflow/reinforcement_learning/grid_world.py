#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
The grid world game model.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    19-1-24 上午9:19
"""

# The common libs.
import random
import os
import time

# The 3rd-part libs.
import numpy as np
import itertools
import scipy.misc
import matplotlib.pyplot as plt


class GameObject(object):
    """The game object including hero object with blue color, goal object
    with green color and fire object with red color.

    Attributes:
        x: The x coordinate.
        y: The y coordinate.
        size: The obj size.
        intensity: The intensity.
        channel: The channel value. The channel of hero is 2 with blue color.
        The channel of goal is 1 with green color. The channel of fire is 0
        with red color.
        reward: The reward value when hero get the address.
        name: The object name(hero, goal or fire).
    """

    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name


class GameEnv(object):
    """The game environment.

    Attributes:
        size_x: The width size.
        size_y: The height size.
        actions: The total action steps.
        objects: The complete objects in the game environment.
        state: The current state of the game.
    """

    def __init__(self, size_x, size_y, actions=50):
        """Initialize the game environment with size and actions space.

        Args:
            size_x: The width size.
            size_y: THe hidth size.
            actions: The total action steps.
        """
        self.size_x = size_x
        self.size_y = size_y
        self.actions = actions
        self.objects = []
        # a = self.reset()
        # print(a.shape)
        # print(a)
        # self.show_env(a)

    def reset(self):
        """Reset the game environment.

        Returns:
            The initial environment render of the game.
        """
        self.objects = []
        hero = GameObject(self.__new_position(), 1, 1, 2, None, "hero")
        self.objects.append(hero)
        goal = GameObject(self.__new_position(), 1, 1, 1, 1, "goal")
        self.objects.append(goal)
        fire = GameObject(self.__new_position(), 1, 1, 0, -1, "fire")
        self.objects.append(fire)
        goal2 = GameObject(self.__new_position(), 1, 1, 1, 1, "goal")
        self.objects.append(goal2)
        fire2 = GameObject(self.__new_position(), 1, 1, 0, -1, "fire")
        self.objects.append(fire2)
        goal3 = GameObject(self.__new_position(), 1, 1, 1, 1, "goal")
        self.objects.append(goal3)
        goal4 = GameObject(self.__new_position(), 1, 1, 1, 1, "goal")
        self.objects.append(goal4)
        # print(self.objects)
        self.state = self.render_env()
        return self.state

    def __new_position(self):
        """Get a new position never used.

        Returns:
            A new position.
        """
        iterables = [range(self.size_x), range(self.size_y)]
        points = []  # Save all points in size.
        for point in itertools.product(*iterables):
            points.append(point)

        current_points = []  # Save used points.
        for object in self.objects:
            if (object.x, object.y) not in current_points:
                current_points.append((object.x, object.y))

        for point in current_points:
            points.remove(point)  # Remove all used points.

        location = np.random.choice(a=range(len(points)), replace=False)
        return points[location]

    def render_env(self):
        """Render the current environment.

        Returns:
            The environment render.
        """
        # The circle outside is white, and the interior is black.
        a = np.ones(shape=[self.size_x + 2, self.size_y + 2, 3])
        a[1:-1, 1:-1, :] = 0

        for object in self.objects:
            a[object.y + 1: object.y + object.size + 1,
            object.x + 1: object.x + object.size + 1,
            object.channel] = object.intensity

        a = scipy.misc.imresize(a, size=[84, 84, 3], interp="nearest")
        # b = scipy.misc.imresize(a[:, :, 0], size=[84, 84, 1], interp="nearest")
        # c = scipy.misc.imresize(a[:, :, 1], size=[84, 84, 1], interp="nearest")
        # d = scipy.misc.imresize(a[:, :, 2], size=[84, 84, 1], interp="nearest")

        # a = np.stack([b, c, d], axis=2)
        return a

    def move_step(self, direction):
        """Hero moves one step.

        Args:
            direction: The direction(0->up, 1->down, 2->left, 3->right).
        """
        x = self.objects[0].x
        y = self.objects[0].y
        if direction == 0 and y >= 1:
            self.objects[0].y -= 1
        elif direction == 1 and y <= self.size_y - 2:
            self.objects[0].y += 1
        elif direction == 2 and x >= 1:
            self.objects[0].x -= 1
        elif direction == 3 and x <= self.size_x - 2:
            self.objects[0].x += 1

    def check_goal(self):
        """Check the goal of hero current position.

        Returns:
            The reward of the hero current position.
        """
        hero = self.objects[0]
        others = self.objects[1:]

        for other in others:
            if other.x == hero.x and other.y == hero.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(GameObject(self.__new_position(), 1,
                                                   1, 1, 1, "goal"))
                elif other.reward == -1:
                    self.objects.append(GameObject(self.__new_position(), 1,
                                                   1, 0, -1, "fire"))
                return other.reward, False
        return 0.0, False

    def step(self, action):
        """Get the new state, reward, and done state of the action.

        Args:
            action: The action.

        Returns:
            s_new is the new state. r is the step reward. d is the done state.
        """
        self.move_step(action) # Move.
        r, d = self.check_goal() # Check the reward and done state, and create
        # new environment.
        s_new= self.render_env() # Render the new environment.
        return s_new, r, d

    def show_env(self, img):
        """Show current environment by pyplot.

        Args:
            img: The current environment image.
        """
        plt.figure(1)
        plt.subplot(111)
        plt.imshow(img, interpolation="nearest")
        plt.show()


if __name__ == "__main__":
    game_env = GameEnv(5, 5, 10)
    img = game_env.reset()
    game_env.show_env(img)
    time.sleep(0.1)
    game_env.move_step(0)
    reward = game_env.check_goal()
    print("reward:", str(reward))
    img = game_env.render_env()
    game_env.show_env(img)
    time.sleep(0.2)
    game_env.move_step(1)
    reward = game_env.check_goal()
    print("reward:", str(reward))
    img = game_env.render_env()
    game_env.show_env(img)
    time.sleep(0.1)
    game_env.move_step(2)
    reward = game_env.check_goal()
    print("reward:", str(reward))
    img = game_env.render_env()
    game_env.show_env(img)
    time.sleep(0.1)
    game_env.move_step(3)
    reward = game_env.check_goal()
    print("reward:", str(reward))
    img = game_env.render_env()
    game_env.show_env(img)
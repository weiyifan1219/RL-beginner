# src/gridworld.py

from typing import Tuple


State = Tuple[int, int]


class GridWorld:
    """
    A simple deterministic GridWorld MDP.

    State:
        (row, col)

    Action:
        0 -> UP
        1 -> DOWN
        2 -> LEFT
        3 -> RIGHT

    Reward:
        normal step: -1
        reach goal: +10
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = [UP, DOWN, LEFT, RIGHT]

    ACTION_NAMES = {
        UP: "UP",
        DOWN: "DOWN",
        LEFT: "LEFT",
        RIGHT: "RIGHT",
    }

    ACTION_DELTAS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    def __init__(
        self,
        height: int = 4,
        width: int = 4,
        start_state: State = (0, 0),
        goal_state: State = (3, 3),
        obstacles=None,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
    ):
        self.height = height
        self.width = width

        self.start_state = start_state
        self.goal_state = goal_state

        self.obstacles = set(obstacles or [(1, 1)])

        self.step_reward = step_reward
        self.goal_reward = goal_reward

        self.state = self.start_state

    def reset(self) -> State:
        """
        Reset the environment to the start state.
        """
        self.state = self.start_state
        return self.state

    def is_valid_state(self, state: State) -> bool:
        """
        Check whether a state is inside the map and not an obstacle.
        """
        row, col = state

        inside_map = (
            0 <= row < self.height
            and 0 <= col < self.width
        )

        return inside_map and state not in self.obstacles

    def is_terminal(self, state: State) -> bool:
        """
        Check whether a state is terminal.
        """
        return state == self.goal_state

    def get_next_state(
        self,
        state: State,
        action: int,
    ) -> State:
        """
        Compute the next state for a deterministic transition.
        """
        if self.is_terminal(state):
            return state

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        row, col = state
        d_row, d_col = self.ACTION_DELTAS[action]

        candidate_state = (
            row + d_row,
            col + d_col,
        )

        if not self.is_valid_state(candidate_state):
            return state

        return candidate_state

    def get_reward(
        self,
        state: State,
        action: int,
        next_state: State,
    ) -> float:
        """
        Return reward for transition:
            state --action--> next_state
        """
        if next_state == self.goal_state:
            return self.goal_reward

        return self.step_reward

    def step(self, action: int):
        """
        Execute one environment step.

        Returns:
            next_state
            reward
            done
        """
        next_state = self.get_next_state(
            self.state,
            action,
        )

        reward = self.get_reward(
            self.state,
            action,
            next_state,
        )

        done = self.is_terminal(next_state)

        self.state = next_state

        return next_state, reward, done

    def get_states(self):
        """
        Return all valid states in the MDP.
        """
        states = []

        for row in range(self.height):
            for col in range(self.width):

                state = (row, col)

                if self.is_valid_state(state):
                    states.append(state)

        return states
from typing import Dict, List, Optional, Tuple


State = Tuple[int, int]
Action = int


class GridWorld:
    """
    A simple deterministic GridWorld environment.

    Default environment:
        - Grid size: 4 x 4
        - Start state: (0, 0)
        - Goal state: (3, 3)
        - Obstacle: (1, 1)

    Rewards:
        - Normal step: -1
        - Reach goal: +10
        - Hit wall / obstacle: stay in place and receive -1

    Actions:
        0: UP
        1: DOWN
        2: LEFT
        3: RIGHT
    """

    # ============================================================
    # Action definitions
    # ============================================================

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = (
        UP,
        DOWN,
        LEFT,
        RIGHT,
    )

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

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        height: int = 4,
        width: int = 4,
        start_state: State = (0, 0),
        goal_state: State = (3, 3),
        obstacles: Optional[List[State]] = None,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
    ):
        self.height = height
        self.width = width

        self.start_state = start_state
        self.goal_state = goal_state

        if obstacles is None:
            obstacles = [(1, 1)]

        self.obstacles = set(obstacles)

        self.step_reward = step_reward
        self.goal_reward = goal_reward

        # Validate environment configuration
        if not self._inside_grid(self.start_state):
            raise ValueError(
                f"Invalid start state: {self.start_state}"
            )

        if not self._inside_grid(self.goal_state):
            raise ValueError(
                f"Invalid goal state: {self.goal_state}"
            )

        if self.start_state in self.obstacles:
            raise ValueError(
                "Start state cannot be an obstacle."
            )

        if self.goal_state in self.obstacles:
            raise ValueError(
                "Goal state cannot be an obstacle."
            )

        self.state = self.start_state

    # ============================================================
    # Basic environment interface
    # ============================================================

    def reset(self) -> State:
        """
        Reset environment to the start state.
        """
        self.state = self.start_state

        return self.state

    def step(
        self,
        action: Action,
    ) -> Tuple[State, float, bool]:
        """
        Execute one action in the environment.

        Returns:
            next_state
            reward
            done
        """

        if action not in self.ACTIONS:
            raise ValueError(
                f"Invalid action: {action}"
            )

        # If already terminal, stay terminal.
        if self.is_terminal(self.state):
            return self.state, 0.0, True

        next_state = self.get_next_state(
            self.state,
            action,
        )

        reward = self.get_reward(
            self.state,
            action,
            next_state,
        )

        self.state = next_state

        done = self.is_terminal(next_state)

        return next_state, reward, done

    # ============================================================
    # State / action space
    # ============================================================

    def get_states(self) -> List[State]:
        """
        Return all valid states.

        Obstacles are not included.
        Goal state is included.
        """

        states = []

        for row in range(self.height):
            for col in range(self.width):

                state = (row, col)

                if state in self.obstacles:
                    continue

                states.append(state)

        return states

    def get_actions(self) -> List[Action]:
        """
        Return all available actions.

        This interface is used by SARSA, Q-Learning,
        and other tabular control algorithms.
        """

        return list(self.ACTIONS)

    # ============================================================
    # Environment model
    # ============================================================

    def get_next_state(
        self,
        state: State,
        action: Action,
    ) -> State:
        """
        Return the next state for a given state-action pair.

        The environment is deterministic.

        If the action would:
            - leave the grid
            - hit an obstacle

        the agent remains in the current state.
        """

        if action not in self.ACTIONS:
            raise ValueError(
                f"Invalid action: {action}"
            )

        if not self.is_valid_state(state):
            raise ValueError(
                f"Invalid state: {state}"
            )

        # Terminal state is absorbing
        if self.is_terminal(state):
            return state

        dr, dc = self.ACTION_DELTAS[action]

        candidate_state = (
            state[0] + dr,
            state[1] + dc,
        )

        if not self.is_valid_state(candidate_state):
            return state

        return candidate_state

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
    ) -> float:
        """
        Return reward for transition:

            state --action--> next_state
        """

        if self.is_terminal(state):
            return 0.0

        if next_state == self.goal_state:
            return self.goal_reward

        return self.step_reward

    def get_transition_probabilities(
        self,
        state: State,
        action: Action,
    ) -> Dict[State, float]:
        """
        Return P(s' | s, a).

        Current GridWorld is deterministic, therefore only
        one next state has probability 1.
        """

        next_state = self.get_next_state(
            state,
            action,
        )

        return {
            next_state: 1.0
        }

    # ============================================================
    # State utilities
    # ============================================================

    def _inside_grid(
        self,
        state: State,
    ) -> bool:
        """
        Check whether state coordinates are inside the grid.
        """

        row, col = state

        return (
            0 <= row < self.height
            and
            0 <= col < self.width
        )

    def is_valid_state(
        self,
        state: State,
    ) -> bool:
        """
        Check whether a state is accessible.
        """

        return (
            self._inside_grid(state)
            and
            state not in self.obstacles
        )

    def is_terminal(
        self,
        state: State,
    ) -> bool:
        """
        Check whether state is terminal.
        """

        return state == self.goal_state

    # ============================================================
    # Visualization helper
    # ============================================================

    def render(self) -> None:
        """
        Print the current GridWorld.

        Symbols:
            A : agent
            S : start
            G : goal
            # : obstacle
            . : empty state
        """

        for row in range(self.height):

            cells = []

            for col in range(self.width):

                state = (row, col)

                if state == self.state:
                    symbol = "A"

                elif state == self.goal_state:
                    symbol = "G"

                elif state == self.start_state:
                    symbol = "S"

                elif state in self.obstacles:
                    symbol = "#"

                else:
                    symbol = "."

                cells.append(symbol)

            print(" ".join(cells))

        print()
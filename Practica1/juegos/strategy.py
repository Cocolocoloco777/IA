"""Strategies for two player games.

   Authors:
        Fabiano Baroni <fabiano.baroni@uam.es>,
        Alejandro Bellogin Kouki <alejandro.bellogin@uam.es>
        Alberto Suárez <alberto.suarez@uam.es>
"""

from __future__ import annotations  # For Python 3.7

from abc import ABC, abstractmethod
import time
from typing import List

import numpy as np

from game import TwoPlayerGame, TwoPlayerGameState
from heuristic import Heuristic


class Strategy(ABC):
    """Abstract base class for player's strategy."""

    def __init__(self, verbose: int = 0) -> None:
        """Initialize common attributes for all derived classes."""
        self.verbose = verbose

    @abstractmethod
    def next_move(
        self,
        state: TwoPlayerGameState,
        gui: bool = False,
    ) -> TwoPlayerGameState:
        """Compute next move."""

    def generate_successors(
        self,
        state: TwoPlayerGameState,
    ) -> List[TwoPlayerGameState]:
        """Generate state successors."""
        assert isinstance(state.game, TwoPlayerGame)
        successors = state.game.generate_successors(state)
        assert successors  # Error if list is empty
        return successors


class RandomStrategy(Strategy):
    """Strategy in which moves are selected uniformly at random."""

    def next_move(
        self,
        state: TwoPlayerGameState,
        gui: bool = False,
    ) -> TwoPlayerGameState:
        """Compute next move."""
        successors = self.generate_successors(state)
        return np.random.choice(successors)


class ManualStrategy(Strategy):
    """Strategy in which the player inputs a move."""

    def next_move(
        self,
        state: TwoPlayerGameState,
        gui: bool = False,
    ) -> TwoPlayerGameState:
        """Compute next move"""
        successors = self.generate_successors(state)

        assert isinstance(state.game, TwoPlayerGame)
        if gui:
            index_successor = state.game.graphical_input(state, successors)
        else:
            index_successor = state.game.manual_input(successors)

        next_state = successors[index_successor]

        if self.verbose > 0:
            print('My move is: {:s}'.format(str(next_state.move_code)))

        return next_state


class MinimaxStrategy(Strategy):
    """Minimax strategy."""

    def __init__(
        self,
        heuristic: Heuristic,
        max_depth_minimax: int,
        max_sec_per_evaluation: float = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.heuristic = heuristic
        self.max_depth_minimax = max_depth_minimax
        self.max_sec_per_evaluation = max_sec_per_evaluation
        self.timed_out = False

    def next_move(
        self,
        state: TwoPlayerGameState,
        gui: bool = False,
    ) -> TwoPlayerGameState:
        """Compute the next state in the game."""

        minimax_value, minimax_successor = self._max_value(
            state,
            self.max_depth_minimax,
        )

        if self.verbose > 0:
            if self.verbose > 1:
                print('\nGame state before move:\n')
                print(state.board)
                print()
            print('Minimax value = {:.2g}'.format(minimax_value))

        return minimax_successor

    def _min_value(
        self,
        state: TwoPlayerGameState,
        depth: int,
    ) -> float:
        """Min step of the minimax algorithm."""

        if state.end_of_game or depth == 0:
            if self.timed_out:
                minimax_value = 0
            else:
                time0 = time.time()
                minimax_value = self.heuristic.evaluate(state)
                time1 = time.time()
                timediff = time1 - time0
                if (self.max_sec_per_evaluation > 0) and (timediff > self.max_sec_per_evaluation):
                    print("Heuristic {} timeout: {} > {}".format(self.heuristic.get_name(), timediff, self.max_sec_per_evaluation))
                    self.timed_out = True
            minimax_successor = None
        else:
            minimax_value = np.inf

            for successor in self.generate_successors(state):
                if self.verbose > 1:
                    print('{}: {}'.format(state.board, minimax_value))

                successor_minimax_value, _ = self._max_value(
                    successor,
                    depth - 1,
                )

                if (successor_minimax_value < minimax_value):
                    minimax_value = successor_minimax_value
                    minimax_successor = successor

        if self.verbose > 1:
            print('{}: {}'.format(state.board, minimax_value))

        return minimax_value, minimax_successor

    def _max_value(
        self,
        state: TwoPlayerGameState,
        depth: int,
    ) -> float:
        """Max step of the minimax algorithm."""

        if state.end_of_game or depth == 0:
            if self.timed_out:
                minimax_value = 0
            else:
                time0 = time.time()
                minimax_value = self.heuristic.evaluate(state)
                time1 = time.time()
                timediff = time1 - time0
                if (self.max_sec_per_evaluation > 0) and (timediff > self.max_sec_per_evaluation):
                    print("Heuristic {} timeout: {} > {}".format(self.heuristic.get_name(), timediff, self.max_sec_per_evaluation))
                    self.timed_out = True
            minimax_successor = None
        else:
            minimax_value = -np.inf

            for successor in self.generate_successors(state):
                if self.verbose > 1:
                    print('{}: {}'.format(state.board, minimax_value))

                successor_minimax_value, _ = self._min_value(
                    successor,
                    depth - 1,
                )
                if (successor_minimax_value > minimax_value):
                    minimax_value = successor_minimax_value
                    minimax_successor = successor

        if self.verbose > 1:
            print('{}: {}'.format(state.board, minimax_value))

        return minimax_value, minimax_successor


class MinimaxAlphaBetaStrategy(Strategy):
    """Minimax alpha-beta strategy."""

    def __init__(
            self,
            heuristic: Heuristic,
            max_depth_minimax: int,
            max_sec_per_evaluation: float = 0,
            verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.heuristic = heuristic
        self.max_depth_minimax = max_depth_minimax
        self.max_sec_per_evaluation = max_sec_per_evaluation
        self.timed_out = False

    def next_move(
            self,
            state: TwoPlayerGameState,
            gui: bool = False,
    ) -> TwoPlayerGameState:
        """Compute the next state in the game."""

        minimax_value, minimax_successor = self._max_value(
            state,
            -np.inf,
            np.inf,
            self.max_depth_minimax,
        )

        """
        # Use this code snippet to trace the execution of the algorithm

                 if self.verbose > 1:
                    print('{}: [{:.2g}, {:.2g}]'.format(
                            state.board,
                            alpha,
                            beta,
                        )
                    )
        """

        return minimax_successor

    def _min_value(
            self,
            state: TwoPlayerGameState,
            alpha: float,
            beta: float,
            depth: int,
    ) -> float:
        """Min step of the minimax algorithm."""

        if self.verbose > 1:
            indent = ' ' * 2 * (self.max_depth_minimax - depth)
            print(indent + '[#] Alpha-beta interval for node '
                  + f'{state.board}: [{alpha}, {beta}]')

        if state.end_of_game or depth == 0:
            if self.verbose > 1:
                print(indent + '[$] Leaf node found.', end=' ')

            if self.timed_out:
                minimax_value = 0
                if self.verbose > 1:
                    print('Timed out :(')
            else:
                time0 = time.time()
                minimax_value = self.heuristic.evaluate(state)
                time1 = time.time()
                timediff = time1 - time0
                if (self.max_sec_per_evaluation > 0
                        and timediff > self.max_sec_per_evaluation):
                    if self.verbose > 1:
                        print(
                            "Heuristic timeout: {} > {}".format(
                                timediff, self.max_sec_per_evaluation))
                    self.timed_out = True
            minimax_successor = None

            if not self.timed_out and self.verbose > 1:
                print(f'Minimax value: {minimax_value}')
        else:
            minimax_value = np.inf

            for successor in self.generate_successors(state):
                if self.verbose > 1:
                    print(indent + '[#] Minimax value for '
                          + f'{state.board}: {minimax_value}. '
                          + f'Analizing successor {successor.board}...')

                successor_minimax_value, _ = self._max_value(
                    successor,
                    alpha,
                    beta,
                    depth - 1,
                    )

                if (successor_minimax_value < minimax_value):
                    if self.verbose > 1:
                        print(
                            indent +
                            f'[#] MIN node {state.board}: ' +
                            f'successor {successor.board}\'s minimax value ' +
                            'is smaller. Updating minimax value...')
                    minimax_value = successor_minimax_value
                    minimax_successor = successor

                if minimax_value <= alpha:
                    if self.verbose > 1:
                        print(indent + f'\033[91m[!] Alpha-beta prune for '
                              + f'MIN node {state.board}\033[0m')
                    break

                beta = min(beta, minimax_value)
                if self.verbose > 1:
                    print(indent + '[#] Alpha-beta interval for '
                          + f'node {state.board}: [{alpha}, {beta}]')

        if self.verbose > 1:
            print(indent + f'[#] Final alpha-beta interval for '
                  + f'node {state.board}: [{alpha}, {beta}]')
            print(indent + f'[#] Final minimax value for '
                  + f'{state.board}: {minimax_value}')
        return minimax_value, minimax_successor

    def _max_value(
            self,
            state: TwoPlayerGameState,
            alpha: float,
            beta: float,
            depth: int,
    ) -> float:
        """Max step of the minimax algorithm."""

        if self.verbose > 1:
            indent = ' ' * 2 * (self.max_depth_minimax - depth)
            print(indent + '[#] Alpha-beta interval for node '
                  + f'{state.board}: [{alpha}, {beta}]')

        if state.end_of_game or depth == 0:
            if self.verbose > 1:
                print(indent + '[$] Leaf node found.', end=' ')

            if self.timed_out:
                minimax_value = 0
                if self.verbose > 1:
                    print('Timed out :(')
            else:
                time0 = time.time()
                minimax_value = self.heuristic.evaluate(state)
                time1 = time.time()
                timediff = time1 - time0
                if (self.max_sec_per_evaluation > 0
                        and timediff > self.max_sec_per_evaluation):
                    if self.verbose > 1:
                        print(
                            "Heuristic timeout: {} > {}".format(
                                timediff, self.max_sec_per_evaluation))
                    self.timed_out = True
            minimax_successor = None

            if not self.timed_out and self.verbose > 1:
                print(f'Minimax value: {minimax_value}')
        else:
            minimax_value = -np.inf

            for successor in self.generate_successors(state):
                if self.verbose > 1:
                    print(indent + '[#] Minimax value for '
                          + f'{state.board}: {minimax_value}. '
                          + f'Analizing successor {successor.board}...')

                successor_minimax_value, _ = self._min_value(
                    successor,
                    alpha,
                    beta,
                    depth - 1,
                    )

                if (successor_minimax_value > minimax_value):
                    if self.verbose > 1:
                        print(
                            indent +
                            f'[#] MAX node {state.board}: ' +
                            f'successor {successor.board}\'s minimax value ' +
                            'is greater. Updating minimax value...')
                    minimax_value = successor_minimax_value
                    minimax_successor = successor

                if minimax_value >= beta:
                    if self.verbose > 1:
                        print(indent + f'\033[91m[!] Alpha-beta prune for '
                              + f'MAX node {state.board}\033[0m')
                    break

                alpha = max(alpha, minimax_value)
                if self.verbose > 1:
                    print(indent + '[#] Alpha-beta interval for '
                          + f'node {state.board}: [{alpha}, {beta}]')

        if self.verbose > 1:
            print(indent + '[#] Final alpha-beta interval for '
                  + f'node {state.board}: [{alpha}, {beta}]')
            print(indent + '[#] Final minimax value for '
                  + f'{state.board}: {minimax_value}')
        return minimax_value, minimax_successor
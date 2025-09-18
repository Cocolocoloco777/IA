# othello.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UAM.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# This file was implemented by Alejandro Bellogin (alejandro.bellogin@uam.es).

import search
import random

from timeit import default_timer as timer
from datetime import timedelta


def from_dictionary_to_array_board(board_dictionary, height, width):
    """From dictionary to array representation."""
    board_array = []

    for i in range(height):
        board_array.append('')
        for j in range(width):
            key = (j + 1, i + 1)
            if key in board_dictionary:
                board_array[i] += board_dictionary[key]
            else:
                board_array[i] += '.'

    return board_array


def from_array_to_dictionary_board(board_array):
    """Create a state from an initial board."""
    if board_array is None:
        return None

    n_rows = len(board_array)
    n_columns = len(board_array[0])
    try:
        board_dictionary = dict(
            [((j + 1, i + 1), board_array[i][j])
                for i in range(n_rows) for j in range(n_columns)
                if board_array[i][j] != '.']
        )
    except IndexError:
        raise IndexError('Wrong configuration of the board')
    else:
        return board_dictionary


# Module Classes
class OthelloState:
    """
    This class defines the mechanics of an Othello game.
    The task of recasting this game state as a search problem is left to
    the OthelloSearchProblem class.
    """

    def __init__(self, board: dict, player1='B', player2='W', cur_player='B', height=8, width=8):
        "Creates a new OthelloState."
        self.board = board
        self.player1 = player1
        self.player2 = player2
        self.cur_player = cur_player
        self.height = height
        self.width = width

    def isGoal(self, min_corners=1):
        """
          Checks to see if any of the players have conquered min_corners in the board.
        """
        corners = [self.board.get((1, 1)), self.board.get((1, self.height)),
                   self.board.get((self.width, 1)), self.board.get((self.width, self.height))]
        return corners.count(self.player1) + corners.count(self.player2) >= min_corners

    def __capture_enemy_in_dir(self, board: dict, move, player: str, delta_x_y) -> list:
        enemy = self.player2 if player == self.player1 else self.player1
        (delta_x, delta_y) = delta_x_y
        x, y = move
        x, y = x + delta_x, y + delta_y
        enemy_list_0 = []
        while board.get((x, y)) == enemy:
            enemy_list_0.append((x, y))
            x, y = x + delta_x, y + delta_y
        if board.get((x, y)) != player:
            del enemy_list_0[:]
        x, y = move
        x, y = x - delta_x, y - delta_y
        enemy_list_1 = []
        while board.get((x, y)) == enemy:
            enemy_list_1.append((x, y))
            x, y = x - delta_x, y - delta_y
        if board.get((x, y)) != player:
            del enemy_list_1[:]
        return enemy_list_0 + enemy_list_1

    def __enemy_captured_by_move(self, board: dict, move, player: str) -> list:
        return self.__capture_enemy_in_dir(board, move, player, (0, 1)) \
            + self.__capture_enemy_in_dir(board, move, player, (1, 0)) \
            + self.__capture_enemy_in_dir(board, move, player, (1, -1)) \
            + self.__capture_enemy_in_dir(board, move, player, (1, 1))

    def __get_valid_moves(self, board: dict, player: str) -> list:
        """Returns a list of valid moves for the player judging from the board."""
        # Get all positions adjacent to existing pieces
        candidates = set()
        # Check all 8 directions around each occupied position
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for (x, y) in board.keys():
            for dx, dy in directions:
                adj_x, adj_y = x + dx, y + dy
                # Only consider positions within board bounds and not occupied
                if (1 <= adj_x <= self.width and
                    1 <= adj_y <= self.height and
                        (adj_x, adj_y) not in board):
                    candidates.add((adj_x, adj_y))
        # Now check only these candidate positions for valid moves
        valid_moves = []
        for pos in candidates:
            if self.__enemy_captured_by_move(board, pos, player):
                valid_moves.append(pos)

        return valid_moves

    def legalMoves(self):
        """
          Returns a list of legal moves from the current state.
        """
        next_player = self.player2 if self.cur_player == self.player1 else self.player1
        return self.__get_valid_moves(self.board, next_player)

    def result(self, move):
        """
          Returns a new board with the current state updated based on the provided move.

        NOTE: This function *does not* change the current object.  Instead,
        it returns a new object.
        """
        result_board = self.board.copy()  # shallow copy is enough
        # show the move on the board
        adversary = self.player2 if self.cur_player == self.player1 else self.player1
        result_board[move] = adversary
        # flip enemy
        for enemy in self.__enemy_captured_by_move(self.board, move, adversary):
            result_board[enemy] = adversary
        newState = OthelloState(result_board, self.player1,
                               self.player2, adversary,
                               self.height, self.width)

        return newState

    # Utilities for comparison and display
    def __eq__(self, other):
        """
            Overloads '=='.
        """
        return self.board == other.board

    def __hash__(self):
        return hash(self.__getAsciiString())

    def __getAsciiString(self):
        """
          Returns a display string for the state
        """
        adversary = self.player2 if self.cur_player == self.player1 else self.player1
        moves = self.__get_valid_moves(self.board, adversary)
        lines = []
        for y in range(0, self.height + 1):
            rowLine = ''
            for x in range(0, self.width + 1):
                if x > 0 and y > 0:
                    if (x, y) in moves:
                        rowLine = rowLine + self.board.get((x, y), '_',)
                    else:
                        rowLine = rowLine + self.board.get((x, y), '.',)
                if x == 0:
                    if y > 0:
                        rowLine = rowLine + str(y) + ' '
                if y == 0:
                    rowLine = rowLine + (chr(x+96) if x > 0 else '  ')
            lines.append(rowLine)
        return '\n'.join(lines)

    def __str__(self):
        return self.__getAsciiString()


class OthelloSearchProblem(search.SearchProblem):
    """
      Implementation of a SearchProblem for Othello

      Each state is represented by an instance of a valid Othello board.
    """

    def __init__(self, othello_state):
        "Creates a new OthelloSearchProblem which stores search information."
        self.state = othello_state

    def getStartState(self):
        return self.state

    def getSuccessors(self, state):
        """
          Returns list of (successor, action, stepCost) pairs where
          each succesor is a new board from the original state and the cost is 1.0 for each
        """
        succ = []
        for a in state.legalMoves():
            succ.append((state.result(a), a, 1))
        return succ

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


class OthelloSearchProblemOneCorner(OthelloSearchProblem):
    """
      Implementation of a SearchProblem for Othello where
      the goal is satisfied when 1 corner is conquered. 
    """

    def isGoalState(self, state):
        return state.isGoal(1)


class OthelloSearchProblemTwoCorners(OthelloSearchProblem):
    """
      Implementation of a SearchProblem for Othello where
      the goal is satisfied when 2 corners are conquered. 
    """

    def isGoalState(self, state):
        return state.isGoal(2)


class OthelloSearchProblemAllCorners(OthelloSearchProblem):
    """
      Implementation of a SearchProblem for Othello where
      the goal is satisfied when all the corners are conquered. 
    """

    def isGoalState(self, state):
        return state.isGoal(4)


def createRandomOthelloGeneralBoard(moves, h, w):
    puzzle = OthelloState(
        from_array_to_dictionary_board(['......',
                                        '......',
                                        '..WB..',
                                        '..BW..',
                                        '......',
                                        '......']), height=h, width=w)
    for _ in range(moves):
        # Execute a random legal move
        puzzle = puzzle.result(random.sample(puzzle.legalMoves(), 1)[0])
    return puzzle


def createSmallRandomOthelloBoard(moves=10):
    """
      moves: number of random moves to apply

      Creates a random Othello board by applying
      a series of 'moves' random moves to a solved
      board. The size of the board is smaller than
      the default board.
    """
    return createRandomOthelloGeneralBoard(moves, 6, 6)


def createRandomOthelloBoard(moves=100):
    """
      moves: number of random moves to apply

      Creates a random Othello board by applying
      a series of 'moves' random moves to a solved
      board.
    """
    return createRandomOthelloGeneralBoard(moves, 8, 8)


manhattan = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)


def cornerHeuristicGeneral(state, problem, type):
    board = state.board
    width = state.width
    height = state.height
    corners = [(1, 1), (1, height), (width, 1), (width, height)]

    # initialise the distances list
    distances = []
    for (x, y) in board.keys():
        for corner in corners:
            distances.append(manhattan(x, y, corner[0], corner[1]))
    h = 0  # set heuristic = 0
    if len(distances) > 0:
        if type == "min":
            h = min(distances)
        elif type == "max":
            h = max(distances)
        elif type == "sum":
            h = sum(distances)
    return h


def cornerHeuristicMin(state, problem):
    return cornerHeuristicGeneral(state, problem, "min")


def cornerHeuristicMax(state, problem):
    return cornerHeuristicGeneral(state, problem, "max")


def cornerHeuristicSum(state, problem):
    return cornerHeuristicGeneral(state, problem, "sum")


def cornerHeuristicComplex(state, problem):
    """
    Complex heuristic that considers:
    1. Minimum distance to any corner (admissible component)
    2. Number of pieces already in corners (reward progress)
    3. Penalty for pieces far from corners
    """
    board = state.board
    width = state.width
    height = state.height
    corners = [(1, 1), (1, height), (width, 1), (width, height)]

    if len(board) == 0:
        return 0

    # Count pieces already in corners
    pieces_in_corners = sum(1 for corner in corners if corner in board)

    # Calculate minimum distance to any corner for each piece
    min_distances = []
    for (x, y) in board.keys():
        min_dist_to_corner = min(manhattan(x, y, cx, cy) for cx, cy in corners)
        min_distances.append(min_dist_to_corner)

    # Base heuristic: minimum distance to reach any corner
    min_distance = min(min_distances) if min_distances else 0
    # Bonus for pieces already in corners (negative cost)
    corner_bonus = pieces_in_corners * -2
    # Small penalty for total spread (encourages consolidation)
    spread_penalty = sum(min_distances) * 0.1

    return max(0, min_distance + corner_bonus + spread_penalty)


if __name__ == '__main__':
    ## variables to play with
    iterations_initial_state = 5
    create_small_board = True
    problem_type = "two_corners"  # "one_corner" "two_corners" "all_corners"
    do_print_boards_in_path = False
    wait_for_user_input = True
    ##

    if create_small_board:
        board = createSmallRandomOthelloBoard(iterations_initial_state)
    else:
        board = createRandomOthelloBoard(iterations_initial_state)

    print('Initial board after %d iterations:' % (iterations_initial_state))
    print(board)

    if problem_type == "one_corner":
        problem = OthelloSearchProblemOneCorner(board)
    elif problem_type == "two_corners":
        problem = OthelloSearchProblemTwoCorners(board)
    else:
        problem = OthelloSearchProblemAllCorners(board)

    print('Problem to be solved: %s' % (problem.__class__))

    sols = dict()
    for method in ['BFS', 'DFS', 'UCS', 'Amin', 'Amax', 'Asum', 'Acomplex']:
        start = timer()
        if method == 'BFS':
            path = search.breadthFirstSearch(problem)
        elif method == 'DFS':
            path = search.depthFirstSearch(problem)
        elif method == 'UCS':
            path = search.uniformCostSearch(problem)
        elif method == 'Amin':
            path = search.aStarSearch(problem, cornerHeuristicMin)
        elif method == 'Amax':
            path = search.aStarSearch(problem, cornerHeuristicMax)
        elif method == 'Asum':
            path = search.aStarSearch(problem, cornerHeuristicSum)
        elif method == 'Acomplex':
            path = search.aStarSearch(problem, cornerHeuristicComplex)
        end = timer()

        if path:
            print_path = [(chr(x+96), y) for x, y in path]
            print('%s found a path of %d moves in %s: %s' %
                  (method, len(path), str(timedelta(seconds=end - start)), str(print_path)))
        else:
            print('%s found no path in %s' % (method, str(timedelta(seconds=end - start))))
        sols[method] = path

    if do_print_boards_in_path:
        for method in sols.keys():
            print('Path found by %s:' % (method))
            path = sols[method]
            curr = board
            i = 1
            for a in path:
                curr = curr.result(a)
                print_a = (chr(a[0]+96), a[1])
                print('After %d move%s (last moved by %s): %s' % (i, ("", "s")[i > 1], curr.cur_player, print_a))
                print(curr)

                # wait for key stroke
                if wait_for_user_input:
                    input('\tPress return for the next state...')
                i += 1
            print()
            input('Press return for the next method...')
            print()

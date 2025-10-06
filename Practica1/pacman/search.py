# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

Name student 1: Rocío Martín Campoy
Name student 2: Pablo Fernández Izquierdo
IA lab group and pair: gggg - 13

"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(search_problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(search_problem):
    """
    Search the deepest nodes in the search tree first
    """
    
    return genericSearch(search_problem, util.Stack())

def depthFirstSearchExpandVisited(search_problem):
    """
    Search the deepest nodes in the search tree first without the restriction of not visiting a previously visited node.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    
    # The Path is composed of two lists, one of the coordinates of each step and another of the directions taken
    structure = util.Stack()

    # Insert the initial path with the initial node and no directions
    structure.push([[search_problem.getStartState()], []]) 

    while not structure.isEmpty():
       # Get the last path that entered into the stack
        path = structure.pop()

        # Get the current state of the path
        current_state = path[0][-1] 

        # Check the end condition and return the directions only
        if search_problem.isGoalState(current_state):
            return path[1] 

        # Loop through ALL the successors of the current state
        for successor in search_problem.getSuccessors(current_state):   
            
            # Create a new path with the successor node and push it into the stack
            new_path = [path[0] + [successor[0]], path[1] + [successor[1]]]
            structure.push(new_path)

    return None

def genericSearch(search_problem, structure):
    """
    Generic search that implements BFS or DFS depending on the data structure
    
    
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    
    # The Path is composed of two lists, one of the coordinates of each step and another of the directions taken
    # Insert the initial path with the initial node and no directions
    structure.push([[search_problem.getStartState()], [], 0])

    # List of visited nodes
    visited = []

    while not structure.isEmpty():

        # Get the last path that entered into the stack
        path = structure.pop()

        print(structure.size())

        # Get the current state of the path
        current_state = path[0][-1] 

        # Check the end condition and return the directions only
        if search_problem.isGoalState(current_state):
            return path[1] 

        # Visit the current state
        if current_state not in visited:

            # Loop through the not visited successors of the current state
            for successor in search_problem.getSuccessors(current_state):
                if successor[0] not in visited:
                    
                    # Create a new path with the successor node and push it into the stack
                    new_path = [path[0] + [successor[0]], path[1] + [successor[1]], path[2] + successor[2]]
                    structure.push(new_path)

    return None
    

def breadthFirstSearch(search_problem):
    """
    Implements a breath first search, returning the objective node at the shallowest depth

    """
    return genericSearch(search_problem, util.Queue())


def uniformCostSearch(search_problem):
    """Search the node of least total cost first."""
    
    # Lambda function that retrieves the cost of visiting the successor
    return genericSearch(search_problem, util.PriorityQueueWithFunction(lambda path: path[2]))


def nullHeuristic(state, search_problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(search_problem, heuristic=nullHeuristic):
    
    """Search the node that has the lowest combined cost and heuristic first."""
    return genericSearch(search_problem, util.PriorityQueueWithFunction(lambda path: path[2] + heuristic(path[0][-1], search_problem)))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

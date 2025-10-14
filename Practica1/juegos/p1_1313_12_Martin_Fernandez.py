import numpy as np

from game import (
    TwoPlayerGameState,
)
from heuristic import (
    simple_evaluation_function,
)
from tournament import (
    StudentHeuristic,
)
    
from reversi import Reversi

def generate_weight_matrix(width, height, corner_weight = 50, wall_weight = 5, wall_double_next_to_corner = 10, wall_next_to_corner = -20, pre_corner_weight = -30, pre_wall_weight = -5, inside_weight = 0):

    if width >= 6:
        top_bottom_row = [corner_weight, wall_next_to_corner, wall_double_next_to_corner] + [wall_weight for _ in range(width - 6)] + [wall_double_next_to_corner, wall_next_to_corner, corner_weight]
    elif width >= 4:
        top_bottom_row = [corner_weight, wall_next_to_corner] + [wall_double_next_to_corner for _ in range(width - 4)] + [wall_next_to_corner, corner_weight]
    elif width >= 2:
        top_bottom_row = [corner_weight] + [wall_next_to_corner for _ in range(width - 2)] + [corner_weight]
    else:
        top_bottom_row = [corner_weight for _ in range(width)]

    if width >= 4:
        second_top_bottom_row = [wall_next_to_corner, pre_corner_weight] + [pre_wall_weight for _ in range(width - 4)] + [pre_corner_weight, wall_next_to_corner]
    elif width >= 2:
        second_top_bottom_row = [wall_next_to_corner] + [pre_corner_weight for _ in range(width - 2)] + [wall_next_to_corner]
    else:
        second_top_bottom_row = [wall_next_to_corner for _ in range(width)]
    
    if width >= 4:
        inner_row = [wall_weight, pre_wall_weight] + [inside_weight for _ in range(width - 4)] + [pre_wall_weight, wall_weight]
    elif width >= 2:
        inner_row = [wall_weight] + [pre_wall_weight for _ in range(width - 2)] + [wall_weight]
    else:
        inner_row = [wall_weight for _ in range(width)]

    if height >= 4:
        matrix = np.array([top_bottom_row, second_top_bottom_row] + [inner_row for _ in range(height - 4)] + [second_top_bottom_row, top_bottom_row])
    elif height >= 2:
        matrix = np.array([top_bottom_row] + [second_top_bottom_row for _ in range(height - 2)] + [top_bottom_row])
    else:
        matrix = np.array([top_bottom_row for _ in range(height)])
        
    return matrix
    

class RobocadilloRochorizo(StudentHeuristic):
    POS_WEIGHT = None

    def get_name(self) -> str:
        return "Robocadillo de rochorizo"
    
    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        
        # Se queda pendiente modificar los pesos en funcion del momento de la partida   
        if self.POS_WEIGHT is None:
            self.POS_WEIGHT = generate_weight_matrix(state.game.width, state.game.height, np.array([[50, -20, 10], [-20, -30, 5], [10, 5, 0]]), 0)
            
        #First of all, we will establish the utility of each point in the board given a certain width and height
        if state.is_player_max(state.player1):
            playerType = state.player1.label
            opp_type = state.player2.label
            my_score = state.scores[0]
            opp_score = state.scores[1]
        else:
            playerType = state.player2.label
            opp_type = state.player1.label
            my_score = state.scores[1]
            opp_score = state.scores[0]

        # Obtener el valor posicional del tablero
        my_positional_score, opp_positional_score = 0,0
        for position in state.board.keys():
            if state.board[position] == playerType:
                my_positional_score += self.POS_WEIGHT[position[0] - 1, position[1]-1]
            else:
                opp_positional_score += self.POS_WEIGHT[position[0]-1, position[1]-1]
        
        positional_score = my_positional_score - opp_positional_score

        if isinstance(state.game, Reversi):
            legal_moves = getattr(state.game, "_get_valid_moves", None)
        
        if callable(legal_moves):
            my_moves = len(legal_moves(state.board, playerType))
            opp_moves = len(legal_moves(state.board, opp_type))
            
        mobility_score = my_moves - opp_moves

        free_positions_norm = (state.game.width * state.game.height - my_score - opp_score) / (state.game.width * state.game.height)
        
        return  (1 - free_positions_norm) * 3 * (my_score - opp_score) + free_positions_norm * positional_score + mobility_score*2
    
        

def generate_weight_matrix(width, height, upper_left_corner_matrix, inside_number):

    # Truncate the corners if the size is too small
    if (width//2 < upper_left_corner_matrix.shape[0]):
        upper_left_corner_matrix = upper_left_corner_matrix[:, 0:width//2]

    if (height//2 < upper_left_corner_matrix.shape[0]):
        upper_left_corner_matrix = upper_left_corner_matrix[0:height//2, :]

    # Get walls
    left_wall = upper_left_corner_matrix[-1, :][None, :]
    right_wall = np.fliplr(left_wall)
    upper_wall = upper_left_corner_matrix[:, -1][:, None]
    lower_wall = np.flipud(upper_wall)

    upper_side = np.hstack((upper_left_corner_matrix, np.tile(upper_wall, max(width - 2 * upper_left_corner_matrix.shape[0], 0)), np.fliplr(upper_left_corner_matrix)))
    middle_part = np.hstack((np.tile(left_wall, (max(height - 2 * upper_left_corner_matrix.shape[1], 0), 1)), np.full((max(height - 2 * upper_left_corner_matrix.shape[1], 0), max(width - 2 * upper_left_corner_matrix.shape[0], 0)), inside_number), np.tile(right_wall, (max(height - 2 * upper_left_corner_matrix.shape[1], 0), 1))))
    lower_side = np.hstack((np.flipud(upper_left_corner_matrix), np.tile(lower_wall, max(width - 2 * upper_left_corner_matrix.shape[0], 0)), np.fliplr(np.flipud(upper_left_corner_matrix))))
    
    return np.vstack((upper_side, middle_part, lower_side))
    

class Rochirimoyo(StudentHeuristic):

    DIFFERENCE_WEIGHT = 3
    MOBILITY_WEIGHT = 2
    STABILITY_WEIGHT = 5
    UPPER_LEFT_CORNER_MATRIX_WEIGHTS = np.array([[50, -20, 10], [-20, -30, 5], [10, 5, 0]])
    INSIDE_MATRIX_WEIGHTS = 0

    POS_WEIGHT = None
    DIFFERENCE_WEIGHT = 0
    MOBILITY_WEIGHT = 0
    UPPER_LEFT_CORNER_MATRIX_WEIGHTS = np.zeros((1, 1))
    INSIDE_MATRIX_WEIGHTS = 0
    CORNERS = None

    def get_name(self) -> str:
        return "Rochirimoyo"

    def generate_weight_matrix(width, height, upper_left_corner_matrix, inside_number):

        # Truncate the corners if the size is too small
        if (width//2 < upper_left_corner_matrix.shape[0]):
            upper_left_corner_matrix = upper_left_corner_matrix[:, 0:width//2]

        if (height//2 < upper_left_corner_matrix.shape[0]):
            upper_left_corner_matrix = upper_left_corner_matrix[0:height//2, :]

        # Get walls
        left_wall = upper_left_corner_matrix[-1, :][None, :]
        right_wall = np.fliplr(left_wall)
        upper_wall = upper_left_corner_matrix[:, -1][:, None]
        lower_wall = np.flipud(upper_wall)

        upper_side = np.hstack((upper_left_corner_matrix, np.tile(upper_wall, max(width - 2 * upper_left_corner_matrix.shape[0], 0)), np.fliplr(upper_left_corner_matrix)))
        middle_part = np.hstack((np.tile(left_wall, (max(height - 2 * upper_left_corner_matrix.shape[1], 0), 1)), np.full((max(height - 2 * upper_left_corner_matrix.shape[1], 0), max(width - 2 * upper_left_corner_matrix.shape[0], 0)), inside_number), np.tile(right_wall, (max(height - 2 * upper_left_corner_matrix.shape[1], 0), 1))))
        lower_side = np.hstack((np.flipud(upper_left_corner_matrix), np.tile(lower_wall, max(width - 2 * upper_left_corner_matrix.shape[0], 0)), np.fliplr(np.flipud(upper_left_corner_matrix))))
        
        return np.vstack((upper_side, middle_part, lower_side))

    def get_stability(self, state: TwoPlayerGameState, playerLabel, opp_label):
        height, width = state.game.height, state.game.width
        stable_states = []
        possible_stable_states = []
        if self.CORNERS == None:
           self.CORNERS = [(1,height), (1,1), (width, height), (1)]
        total_score = 0
        
        # For each corner, expand the stability of the states, independently of the player
        for corner in self.CORNERS:
            
            expanding_label = None
            expanding_label = state.board.get(corner, None)
            
            # If the corner is of a player, set the corner to be expanded
            if expanding_label != None:
                possible_stable_states.append(corner)
            
                
            # For every possible stable state
            num_stable_states = 0
            while len(possible_stable_states) > 0:
                # Obtain the first state
                checking_state = possible_stable_states.pop()
                # Get the label of the piece in that state                
                # If the state if stable, add 1 to the number of stable states for that corner
                if checking_state not in stable_states and self.is_stable(expanding_label, checking_state, stable_states, height, width):
                    num_stable_states += 1
                    stable_states.append(checking_state)
                    x, y = checking_state[0], checking_state[1]
                    
                    # Calculate the posible adjacent states 
                    adjacent_states = [(x+1, y),(x-1, y),(x, y+1), (x, y-1), (x+1,y+1), (x+1,y-1), (x-1,y-1), (x-1, y+1)]
                    for adjacent in adjacent_states:
                        # If the adjacent state is in the table and the place is occupied by a piece of the expanding player, add it to be analized
                        if self.is_in_table(width, height, adjacent) and adjacent not in possible_stable_states and state.board.get(adjacent, None) == expanding_label:
                            possible_stable_states.append(adjacent)
            
            if expanding_label == playerLabel:
                total_score += num_stable_states
            else:
                total_score -= num_stable_states
        
        return total_score
       
    def is_in_table(self, width, height, position):
        """
        Function that returns whether a certain position is in the Reversi table
        """
        if position[0] <= width and position[0] > 1 and position[1] > 0 and position[1] <= height:
            return True
        else:
            return False
    
    
    def state_is_border(self, state, height, width):
        """
        Function that returns whether the given state is a border or not 
        """
        
        # If the x coordinate is 1 or the width or the y coordinate is the height or one, the state is a border
        if state[0] == (width+1) or state[0] == 0 or state[1] == (height+1) or state[1] == 0:
            return True

        return False
         
        
    
    def is_stable(self, expanding_label, possible_stable, stable_states, height, width):
        
        # To see if a piece if stable we need to look into the 4 diferent directions NORTH-SOUTH, NORTHWEST-SOUTHEAST, SOUTHWEST-NOTHEAST, EAST-WEST 
        # We need to have at least one stable piece or a border piece for each 2 directions (x-y) for the central piece to be considered stable
        
        NORTH_SOUTH = [(possible_stable[0], possible_stable [1]+1), (possible_stable[0], possible_stable[1]-1)]
        EAST_WEST = [(possible_stable[0]-1, possible_stable[1]), (possible_stable[0]+1, possible_stable[1])]
        NORTHEAST_SOUTHWEST = [(possible_stable[0]-1, possible_stable[1] + 1), (possible_stable[0] + 1, possible_stable[1] - 1)]
        SOUTHEAST_NOTHWEST = [(possible_stable[0] + 1, possible_stable[1] + 1),(possible_stable[0] - 1, possible_stable[1] - 1)]
        
        DIRECTIONS = [NORTH_SOUTH, EAST_WEST, NORTHEAST_SOUTHWEST, SOUTHEAST_NOTHWEST]
        
        if possible_stable in self.CORNERS:
            return True
        
        for direction in DIRECTIONS:
            if direction[0] not in stable_states and not self.state_is_border(direction[0], height, width) and direction[1] not in stable_states and not self.state_is_border(direction[1], height, width):
                return False
        
        return True
        
        
        
    
    def get_players_label_and_score(self, state: TwoPlayerGameState):

        # Get the player labels and score
        if state.is_player_max(state.player1):
            player_label = state.player1.label
            opp_label = state.player2.label
            player_score = state.scores[0]
            opp_score = state.scores[1]
        else:
            player_label = state.player2.label
            opp_label = state.player1.label
            player_score = state.scores[1]
            opp_score = state.scores[0]

        return player_label, opp_label, player_score, opp_score

    def get_positional_score(self, state: TwoPlayerGameState, player_label):

        # Init the matrix if it is the first time
        if self.POS_WEIGHT is None:
            self.POS_WEIGHT = generate_weight_matrix(state.game.width, state.game.height, self.UPPER_LEFT_CORNER_MATRIX_WEIGHTS, self.INSIDE_MATRIX_WEIGHTS)
        
        # Obtain the positional value of each position and compute the scores
        player_positional_score, opp_positional_score = 0, 0
        for position in state.board.keys():

            # Add the positional value to the player score if it is the player's chip 
            if state.board[position] == player_label:
                player_positional_score += self.POS_WEIGHT[position[0] - 1, position[1]-1]
            else:
                # Add the positional value to the opponent score if it is the opponent's chip
                opp_positional_score += self.POS_WEIGHT[position[0]-1, position[1]-1]
        
        return player_positional_score, opp_positional_score

    def get_mobility_score(self, state: TwoPlayerGameState, player_label, opp_label):

        # Get the private function from the game that return the possible valid moves of a board
        if isinstance(state.game, Reversi):
            legal_moves = getattr(state.game, "_get_valid_moves", None)
        
        # Get the number of legal moves from each player
        if callable(legal_moves):
            player_moves = len(legal_moves(state.board, player_label))
            opp_moves = len(legal_moves(state.board, opp_label))

        return player_moves, opp_moves
    
    def get_relative_available_positions(self, state: TwoPlayerGameState, player_score, opp_score):

        # Compute the number of positions in the game
        all_positions = state.game.width * state.game.height

        # Compute the number of available positions
        available_positions = all_positions - player_score - opp_score
        
        return available_positions / (all_positions-4)
    
    def evaluation_function(self, state: TwoPlayerGameState) -> float:

        # Get the players labels and scores
        player_label, opp_label, player_score, opp_score = self.get_players_label_and_score(state)

        # Calculate the difference score
        difference_score = player_score - opp_score

        # Calculate the positional score
        player_positional_score, opp_positional_score = self.get_positional_score(state, player_label)
        positional_score = player_positional_score - opp_positional_score

        # Calculate the mobility score
        player_moves, opp_moves = self.get_mobility_score(state, player_label, opp_label)
        mobility_score = player_moves - opp_moves

        # Get the relative available positions
        available_positions = self.get_relative_available_positions(state, player_score, opp_score)

        # Get the stability score
        stability_score = self.get_stability(state, player_label, opp_label)
        
        # Compute the score
        score =  self.get_difference_weight(available_positions) * difference_score
        score +=  self.get_positional_weight(available_positions) * positional_score
        score +=  self.get_mobility_weight(available_positions) * mobility_score
        score += self.get_stability_weight(available_positions) * stability_score
        
        return score
    
    def get_positional_weight(self, available_positions):
        return available_positions
    
    def get_difference_weight(self, available_positions):
        return self.DIFFERENCE_WEIGHT * (1 - available_positions)
    
    def get_mobility_weight(self, available_positions):
        return self.MOBILITY_WEIGHT
    
    def get_stability_weight(self, available_positions):
        return self.STABILITY_WEIGHT
    
    def get_name(self) -> str:
        return "Rochirimoyo"
    
    def get_positional_weight(self, available_positions):
        return available_positions
    
    def get_difference_weight(self, available_positions):
        return self.DIFFERENCE_WEIGHT* (1- available_positions)
    
    def get_mobility_weight(self, available_positions):
        return self.MOBILITY_WEIGHT*available_positions
    
    def get_stability_weight(self, available_positions):
        return self.STABILITY_WEIGHT*(1-available_positions)
    
class RochocolateCaliente(StudentHeuristic):

    DIFFERENCE_WEIGHT = 3

    MOBILITY_WEIGHT_INITIAL = 4
    MOBILITY_WEIGHT_FINAL = 1
    
    STABILITY_WEIGHT_INITIAL = 1
    STABILITY_WEIGHT_FINAL = 6

    UPPER_LEFT_CORNER_MATRIX_WEIGHTS = np.array([[50, -20, 10], [-20, -30, 5], [10, 5, 0]])
    INSIDE_MATRIX_WEIGHTS = 0

    POS_WEIGHT = None
    DIFFERENCE_WEIGHT = 0
    MOBILITY_WEIGHT = 0
    UPPER_LEFT_CORNER_MATRIX_WEIGHTS = np.zeros((1, 1))
    INSIDE_MATRIX_WEIGHTS = 0
    CORNERS = None

    def get_name(self) -> str:
        return "Rochocolate caliente"

    def generate_weight_matrix(width, height, upper_left_corner_matrix, inside_number):

        # Truncate the corners if the size is too small
        if (width//2 < upper_left_corner_matrix.shape[0]):
            upper_left_corner_matrix = upper_left_corner_matrix[:, 0:width//2]

        if (height//2 < upper_left_corner_matrix.shape[0]):
            upper_left_corner_matrix = upper_left_corner_matrix[0:height//2, :]

        # Get walls
        left_wall = upper_left_corner_matrix[-1, :][None, :]
        right_wall = np.fliplr(left_wall)
        upper_wall = upper_left_corner_matrix[:, -1][:, None]
        lower_wall = np.flipud(upper_wall)

        upper_side = np.hstack((upper_left_corner_matrix, np.tile(upper_wall, max(width - 2 * upper_left_corner_matrix.shape[0], 0)), np.fliplr(upper_left_corner_matrix)))
        middle_part = np.hstack((np.tile(left_wall, (max(height - 2 * upper_left_corner_matrix.shape[1], 0), 1)), np.full((max(height - 2 * upper_left_corner_matrix.shape[1], 0), max(width - 2 * upper_left_corner_matrix.shape[0], 0)), inside_number), np.tile(right_wall, (max(height - 2 * upper_left_corner_matrix.shape[1], 0), 1))))
        lower_side = np.hstack((np.flipud(upper_left_corner_matrix), np.tile(lower_wall, max(width - 2 * upper_left_corner_matrix.shape[0], 0)), np.fliplr(np.flipud(upper_left_corner_matrix))))
        
        return np.vstack((upper_side, middle_part, lower_side))

    def get_stability(self, state: TwoPlayerGameState, playerLabel, opp_label):
        height, width = state.game.height, state.game.width
        stable_states = []
        possible_stable_states = []
        if self.CORNERS == None:
           self.CORNERS = [(1,height), (1,1), (width, height), (1)]
        total_score = 0
        
        # For each corner, expand the stability of the states, independently of the player
        for corner in self.CORNERS:
            
            expanding_label = None
            expanding_label = state.board.get(corner, None)
            
            # If the corner is of a player, set the corner to be expanded
            if expanding_label != None:
                possible_stable_states.append(corner)
            
                
            # For every possible stable state
            num_stable_states = 0
            while len(possible_stable_states) > 0:
                # Obtain the first state
                checking_state = possible_stable_states.pop()
                # Get the label of the piece in that state                
                # If the state if stable, add 1 to the number of stable states for that corner
                if checking_state not in stable_states and self.is_stable(expanding_label, checking_state, stable_states, height, width):
                    num_stable_states += 1
                    stable_states.append(checking_state)
                    x, y = checking_state[0], checking_state[1]
                    
                    # Calculate the posible adjacent states 
                    adjacent_states = [(x+1, y),(x-1, y),(x, y+1), (x, y-1), (x+1,y+1), (x+1,y-1), (x-1,y-1), (x-1, y+1)]
                    for adjacent in adjacent_states:
                        # If the adjacent state is in the table and the place is occupied by a piece of the expanding player, add it to be analized
                        if self.is_in_table(width, height, adjacent) and adjacent not in possible_stable_states and state.board.get(adjacent, None) == expanding_label:
                            possible_stable_states.append(adjacent)
            
            if expanding_label == playerLabel:
                total_score += num_stable_states
            else:
                total_score -= num_stable_states
        
        return total_score
       
    def is_in_table(self, width, height, position):
        """
        Function that returns whether a certain position is in the Reversi table
        """
        if position[0] <= width and position[0] > 1 and position[1] > 0 and position[1] <= height:
            return True
        else:
            return False
    
    
    def state_is_border(self, state, height, width):
        """
        Function that returns whether the given state is a border or not 
        """
        
        # If the x coordinate is 1 or the width or the y coordinate is the height or one, the state is a border
        if state[0] == (width+1) or state[0] == 0 or state[1] == (height+1) or state[1] == 0:
            return True

        return False
         
    def is_stable(self, expanding_label, possible_stable, stable_states, height, width):
        
        # To see if a piece if stable we need to look into the 4 diferent directions NORTH-SOUTH, NORTHWEST-SOUTHEAST, SOUTHWEST-NOTHEAST, EAST-WEST 
        # We need to have at least one stable piece or a border piece for each 2 directions (x-y) for the central piece to be considered stable
        
        NORTH_SOUTH = [(possible_stable[0], possible_stable [1]+1), (possible_stable[0], possible_stable[1]-1)]
        EAST_WEST = [(possible_stable[0]-1, possible_stable[1]), (possible_stable[0]+1, possible_stable[1])]
        NORTHEAST_SOUTHWEST = [(possible_stable[0]-1, possible_stable[1] + 1), (possible_stable[0] + 1, possible_stable[1] - 1)]
        SOUTHEAST_NOTHWEST = [(possible_stable[0] + 1, possible_stable[1] + 1),(possible_stable[0] - 1, possible_stable[1] - 1)]
        
        DIRECTIONS = [NORTH_SOUTH, EAST_WEST, NORTHEAST_SOUTHWEST, SOUTHEAST_NOTHWEST]
        
        if possible_stable in self.CORNERS:
            return True
        
        for direction in DIRECTIONS:
            if direction[0] not in stable_states and not self.state_is_border(direction[0], height, width) and direction[1] not in stable_states and not self.state_is_border(direction[1], height, width):
                return False
        
        return True
        
    def get_players_label_and_score(self, state: TwoPlayerGameState):

        # Get the player labels and score
        if state.is_player_max(state.player1):
            player_label = state.player1.label
            opp_label = state.player2.label
            player_score = state.scores[0]
            opp_score = state.scores[1]
        else:
            player_label = state.player2.label
            opp_label = state.player1.label
            player_score = state.scores[1]
            opp_score = state.scores[0]

        return player_label, opp_label, player_score, opp_score

    def get_positional_score(self, state: TwoPlayerGameState, player_label):

        # Init the matrix if it is the first time
        if self.POS_WEIGHT is None:
            self.POS_WEIGHT = generate_weight_matrix(state.game.width, state.game.height, self.UPPER_LEFT_CORNER_MATRIX_WEIGHTS, self.INSIDE_MATRIX_WEIGHTS)
        
        # Obtain the positional value of each position and compute the scores
        player_positional_score, opp_positional_score = 0, 0
        for position in state.board.keys():

            # Add the positional value to the player score if it is the player's chip 
            if state.board[position] == player_label:
                player_positional_score += self.POS_WEIGHT[position[0] - 1, position[1]-1]
            else:
                # Add the positional value to the opponent score if it is the opponent's chip
                opp_positional_score += self.POS_WEIGHT[position[0]-1, position[1]-1]
        
        return player_positional_score, opp_positional_score

    def get_mobility_score(self, state: TwoPlayerGameState, player_label, opp_label):

        # Get the private function from the game that return the possible valid moves of a board
        if isinstance(state.game, Reversi):
            legal_moves = getattr(state.game, "_get_valid_moves", None)
        
        # Get the number of legal moves from each player
        if callable(legal_moves):
            player_moves = len(legal_moves(state.board, player_label))
            opp_moves = len(legal_moves(state.board, opp_label))

        return player_moves, opp_moves
    
    def get_relative_available_positions(self, state: TwoPlayerGameState, player_score, opp_score):

        # Compute the number of positions in the game
        all_positions = state.game.width * state.game.height

        # Compute the number of available positions
        available_positions = all_positions - player_score - opp_score
        
        return available_positions / (all_positions-4)
    
    def evaluation_function(self, state: TwoPlayerGameState) -> float:

        # Get the players labels and scores
        player_label, opp_label, player_score, opp_score = self.get_players_label_and_score(state)

        # Calculate the difference score
        difference_score = player_score - opp_score

        # Calculate the positional score
        player_positional_score, opp_positional_score = self.get_positional_score(state, player_label)
        positional_score = player_positional_score - opp_positional_score

        # Calculate the mobility score
        player_moves, opp_moves = self.get_mobility_score(state, player_label, opp_label)
        mobility_score = player_moves - opp_moves

        # Get the relative available positions
        available_positions = self.get_relative_available_positions(state, player_score, opp_score)

        # Get the stability score
        stability_score = self.get_stability(state, player_label, opp_label)
        
        # Compute the score
        score =  self.get_difference_weight(available_positions) * difference_score
        score +=  self.get_positional_weight(available_positions) * positional_score
        score +=  self.get_mobility_weight(available_positions) * mobility_score
        score += self.get_stability_weight(available_positions) * stability_score
        
        return score
    
    def get_positional_weight(self, available_positions):
        return available_positions
    
    def get_difference_weight(self, available_positions):
        return self.DIFFERENCE_WEIGHT * (1 - available_positions)
    
    def get_mobility_weight(self, available_positions):
        return self.MOBILITY_WEIGHT
    
    def get_stability_weight(self, available_positions):
        return self.STABILITY_WEIGHT
    
    def get_name(self) -> str:
        return "Rochocolate caliente"
    
    def get_positional_weight(self, available_positions):
        return available_positions
    
    def get_difference_weight(self, available_positions):
        return self.DIFFERENCE_WEIGHT * (1 - available_positions)**4
    
    def get_mobility_weight(self, available_positions):
        return self.MOBILITY_WEIGHT_FINAL + available_positions*(self.MOBILITY_WEIGHT_INITIAL - self.MOBILITY_WEIGHT_FINAL)
    
    def get_stability_weight(self, available_positions):
        return self.STABILITY_WEIGHT_FINAL + available_positions*(self.STABILITY_WEIGHT_INITIAL - self.STABILITY_WEIGHT_FINAL)
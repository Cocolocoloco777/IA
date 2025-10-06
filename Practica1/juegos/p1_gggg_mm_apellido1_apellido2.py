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

TEMPLATE_8x8 = np.array([
    [50, -20,  10,   5,   5,  10, -20, 50],
    [-20, -30,  -5,  -5,  -5,  -5, -30, -20],
    [ 10,  -5,   0,   0,   0,   0,  -5,  10],
    [  5,  -5,   0,   0,   0,   0,  -5,   5],
    [  5,  -5,   0,   0,   0,   0,  -5,   5],
    [ 10,  -5,   0,   0,   0,   0,  -5,  10],
    [-20, -30,  -5,  -5,  -5,  -5, -30, -20],
    [50, -20,  10,   5,   5,  10, -20, 50]
], dtype=float)

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
    

def func_glob(n: int, state: TwoPlayerGameState) -> float:
    return n + simple_evaluation_function(state)


class Solution1(StudentHeuristic):
    POS_WEIGHT = None
    def get_name(self) -> str:
        return "weighted_heuristic"
    
    
    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        
        # Se queda pendiente modificar los pesos en funcion del momento de la partida   
        if self.POS_WEIGHT is None:
            self.POS_WEIGHT = generate_weight_matrix(state.game.width, state.game.height)
            
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
        
        return 3 * (my_score - opp_score) +  positional_score + mobility_score*2



class Solution2(StudentHeuristic):
    def get_name(self) -> str:
        return "solution2"
    
    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        return func_glob(2, state)



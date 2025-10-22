import inspect
import os
import sys
import time
from importlib import util
from typing import Callable, Tuple

from heuristic import Heuristic
from strategy import MinimaxAlphaBetaStrategy, Strategy, MinimaxStrategy
from typing import Callable
from game import Player, TwoPlayerMatch

from demo_tournament import Heuristic1, create_reversi_match
from p1_1313_12_Martin_Fernandez import RochocolateCaliente
import timeit

class Tournament(object):
    def __init__(self, max_depth: int,
                 init_match: Callable[[Player, Player], TwoPlayerMatch],
                 max_evaluation_time: float):
        self.__max_depth = max_depth
        self.__init_match = init_match
        self.__max_eval_time = max_evaluation_time

    def __get_function_from_str(self, name: str, definition: str, max_strat: int) -> list:
        # write content in file with new name
        newfile = "playermodule__" + name
        with open(newfile, 'w') as fp:
            print(definition, file=fp)
        student_classes = list()
        n_strat = 0
        # not needed, but this hack somehow fixes some files not being loaded
        time.sleep(1)
        sp = util.find_spec(newfile.replace(".py", ""))
        if sp:
            m = sp.loader.load_module()
            # return all the objects that satisfy the function signature
            for name, obj in inspect.getmembers(m, inspect.isclass):
                if name != "StudentHeuristic":
                    for name2, obj2 in inspect.getmembers(obj, inspect.isfunction):
                        if name2 == "evaluation_function" and n_strat < max_strat:
                            student_classes.append(obj)
                            n_strat += 1
                        elif name2 == "evaluation_function":
                            print("Ignoring evaluation function in %s because limit of submissions was reached (%d)" % (
                                name, max_strat), file=sys.stderr)
                    # end for
            # end for
        # remove file
        os.remove(newfile)
        return student_classes

    #   we assume there is one file for each student/pair
    def load_strategies_from_folder(self, folder: str, max_strat: int = 3) -> dict:
        student_strategies = dict()
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p):
                with open(p, 'r') as fp:
                    s = fp.read()
                    name = f
                    strategies = self.__get_function_from_str(
                        name, s, max_strat)
                    student_strategies[f] = strategies
        return student_strategies

    def run(self, student_strategies: dict, player_strategy: Strategy, verbosity: int = 0, measure_nodes_visited: bool = False, increasing_depth: bool = True,
            n_pairs: int = 1, allow_selfmatch: bool = False) -> Tuple[dict, dict, dict]:
        """
        Play a tournament among the strategies.
        n_pairs = games each strategy plays as each color against
        each opponent. So with N strategies, a total of
        N*(N-1)*n_pairs games are played.
        """
        scores = dict()
        totals = dict()
        name_mapping = dict()
        for student1 in student_strategies:
            strats1 = student_strategies[student1]
            for student2 in student_strategies:
                if student1 > student2:
                    continue
                if student1 == student2 and not allow_selfmatch:
                    continue
                strats2 = student_strategies[student2]
                for player1 in strats1:
                    for player2 in strats2:
                        # we now instantiate the players
                        for pair in range(2*n_pairs):
                            player1_first = (pair % 2) == 1
                            sh1 = player1()
                            name1 = student1 + "_" + sh1.get_name()
                            name_mapping[name1] = sh1.get_name()
                            sh2 = player2()
                            name2 = student2 + "_" + sh2.get_name()
                            name_mapping[name2] = sh2.get_name()
                            if increasing_depth:
                                for depth in range(1, self.__max_depth):
                                    print("hola")
                                    pl1 = Player(
                                        name=name1,
                                        strategy=player_strategy( 
                                            heuristic=Heuristic(
                                                name=sh1.get_name(),
                                                evaluation_function=sh1.evaluation_function),
                                            max_depth_minimax=depth,
                                            max_sec_per_evaluation=self.__max_eval_time,
                                            verbose=verbosity,
                                            measure_nodes=measure_nodes_visited,
                                        ),
                                    )
                                    pl2 = Player(
                                        name=name2,
                                        strategy=player_strategy(  
                                            heuristic=Heuristic(
                                                name=sh2.get_name(),
                                                evaluation_function=sh2.evaluation_function),
                                            max_depth_minimax=depth,
                                            max_sec_per_evaluation=self.__max_eval_time,
                                            verbose=verbosity,
                                            measure_nodes=measure_nodes_visited,
                                        ),
                                    )

                                    self.__single_run(
                                        player1_first, pl1, name1, pl2, name2, scores, totals)
                            else:
                                depth = self.__max_depth
                                pl1 = Player(
                                    name=name1,
                                    strategy=player_strategy( 
                                        heuristic=Heuristic(
                                            name=sh1.get_name(),
                                            evaluation_function=sh1.evaluation_function),
                                        max_depth_minimax=depth,
                                        max_sec_per_evaluation=self.__max_eval_time,
                                        verbose=verbosity,
                                        measure_nodes=measure_nodes_visited,
                                    ),
                                )
                                pl2 = Player(
                                    name=name2,
                                    strategy=player_strategy(  
                                        heuristic=Heuristic(
                                            name=sh2.get_name(),
                                            evaluation_function=sh2.evaluation_function),
                                        max_depth_minimax=depth,
                                        max_sec_per_evaluation=self.__max_eval_time,
                                        verbose=verbosity,
                                        measure_nodes=measure_nodes_visited,
                                    ),
                                )

                                self.__single_run(
                                    player1_first,
                                    pl1, name1,
                                    pl2, name2,
                                    scores, totals)
        return scores, totals, name_mapping

    def __single_run(self, player1_first: bool, pl1: Player, name1: str,
                     pl2: Player, name2: str, scores: dict, totals: dict):
        players = []
        if player1_first:
            players = [pl1, pl2]
        else:
            players = [pl2, pl1]
        game = self.__init_match(players[0], players[1])
        try:
            game_scores = game.play_match()
            # let's get the scores (do not assume they will always be binary)
            # we assume a higher score is better
            if player1_first:
                score1, score2 = game_scores[0], game_scores[1]
            else:
                score1, score2 = game_scores[1], game_scores[0]
            wins = loses = 0
            if score1 > score2:
                wins, loses = 1, 0
            elif score2 > score1:
                wins, loses = 0, 1
        except Warning:
            wins = loses = 0
        # store the 1-to-1 numbers
        if name1 not in scores:
            scores[name1] = dict()
        if name2 not in scores:
            scores[name2] = dict()
        scores[name1][name2] = wins if name2 not in scores[name1] else wins + \
            scores[name1][name2]
        scores[name2][name1] = loses if name1 not in scores[name2] else loses + \
            scores[name2][name1]
        # store the total values
        if name1 not in totals:
            totals[name1] = 0
        totals[name1] += wins
        if name2 not in totals:
            totals[name2] = 0
        totals[name2] += loses
        # end of function

if __name__ == "__main__":

    tour = Tournament(max_depth=4, init_match=create_reversi_match, max_evaluation_time=0.5)

    strategies = {'opt1': [Heuristic1],'opt2': [Heuristic1]}
    player_strategies = {'opt1': [RochocolateCaliente],'opt2': [RochocolateCaliente]}

    time_testing = True
    nodes_visited_testing = False

    # Time testing
    if time_testing:

        # Time test statements
        statement_1_min_max = "tour.run(student_strategies=strategies, player_strategy = MinimaxStrategy, increasing_depth=False, n_pairs=1, allow_selfmatch=False)"
        statement_2_min_max = "tour.run(student_strategies=player_strategies, player_strategy = MinimaxStrategy, increasing_depth=False, n_pairs=1, allow_selfmatch=False)"
        statement_1_alpha_beta = "tour.run(student_strategies=strategies, player_strategy = MinimaxAlphaBetaStrategy, increasing_depth=False, n_pairs=1, allow_selfmatch=False)"
        statement_2_alpha_beta = "tour.run(student_strategies=player_strategies, player_strategy = MinimaxAlphaBetaStrategy, increasing_depth=False, n_pairs=1, allow_selfmatch=False)"

        # Time testing
        n = 3
        
        #times = timeit.repeat(stmt = statement_1_min_max, repeat = n, number = 1, globals=globals())
        #print(f"Heuristic 1 min max: Minimum time: {min(times)}, mean time: {sum(times) / len(times)}")

        #times = timeit.repeat(stmt = statement_2_min_max, repeat = n, number = 1, globals=globals())
        #print(f"Player Heuristic min max: Minimum time: {min(times)}, mean time: {sum(times) / len(times)}")

        #times = timeit.repeat(stmt = statement_1_alpha_beta, repeat = n, number = 1, globals=globals())
        #print(f"Heuristic 1 alpha beta: Minimum time: {min(times)}, mean time: {sum(times) / len(times)}")

        times = timeit.repeat(stmt = statement_2_alpha_beta, repeat = n, number = 1, globals=globals())
        print(f"Player Heuristic alpha beta: Minimum time: {min(times)}, mean time: {sum(times) / len(times)}")

    # Counting nodes visited
    if nodes_visited_testing:
        # Counting Nodes testing
        MinimaxStrategy.heuristic_executed = 0
        MinimaxStrategy.nodes_visited = 0
        tour.run(student_strategies=strategies, player_strategy = MinimaxStrategy,  measure_nodes_visited = True, increasing_depth=False, n_pairs=1, allow_selfmatch=False)
        print(f"Heuristic 1 min max: nodes visited {MinimaxStrategy.nodes_visited} heuristic executed: {MinimaxStrategy.heuristic_executed}")

        MinimaxStrategy.heuristic_executed = 0
        MinimaxStrategy.nodes_visited = 0
        tour.run(student_strategies=player_strategies, player_strategy = MinimaxStrategy,  measure_nodes_visited = True, increasing_depth=False, n_pairs=1, allow_selfmatch=False)
        print(f"Heuristic 1 min max: nodes visited {MinimaxStrategy.nodes_visited} heuristic executed: {MinimaxStrategy.heuristic_executed}")

        MinimaxAlphaBetaStrategy.heuristic_executed = 0
        MinimaxAlphaBetaStrategy.nodes_visited = 0
        tour.run(student_strategies=strategies, player_strategy = MinimaxAlphaBetaStrategy,  measure_nodes_visited = True, increasing_depth=False, n_pairs=1, allow_selfmatch=False)
        print(f"Heuristic 1 alpha beta: nodes visited {MinimaxAlphaBetaStrategy.nodes_visited} heuristic executed: {MinimaxAlphaBetaStrategy.heuristic_executed}")

        MinimaxAlphaBetaStrategy.heuristic_executed = 0
        MinimaxAlphaBetaStrategy.nodes_visited = 0
        tour.run(student_strategies=player_strategies, player_strategy = MinimaxAlphaBetaStrategy,  measure_nodes_visited = True, increasing_depth=False, n_pairs=1, allow_selfmatch=False)
        print(f"Heuristic 1 alpha beta: nodes visited {MinimaxAlphaBetaStrategy.nodes_visited} heuristic executed: {MinimaxAlphaBetaStrategy.heuristic_executed}")

        



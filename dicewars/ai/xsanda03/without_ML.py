import logging

from dicewars.client.ai_driver import import BattleCommand, EndTurnCommand, TransferCommand


class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI-WITHOUT-ML                          ')

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        PSEUDOCODE:
        if (time_left > time_treshold):
            action = maxn(args)
        #problem 1 = mam ziskavat iba jednu akciu alebo skupinu akcii(cely tah)
        #problem 2 = aj transfery ratam za akcie?
        #problem 3 = na konci kola dostanem nejake jednotky nahodne rozmiestnene medzi moje uzemia
        else:
            akcia = find_best_simple_attack(args)
            if akcia == None
                return EndTurnCommand()

    def maxn(self,args):
        for move in available_moves:
            game_state = do_move(move,game_state)
            if (current_player == last):
                heuristic = calculate_heurisitc(game_state, current_player, args)
            else:
                move, heuristic = maxn(game_state,current_player+1))
            heuristics.append(heuristic)

        best_move_index = find_max_for_current_player(available_moves,heuristics)
        return available_moves[best_move_index], heuristics[best_move_index]

    def calculate_heuristic(self,args):
        #ohodnot tah pomocou nejakej heuristiky

    def calculate_heuristic_withML(self,args):
        return model.predict(args)

    def find_max_for_current_player(self,moves,move_values):
        # hodnota kazdeho tahu by mala byt reprezentovana vektorom [a,b,c,d]
        # kde a je ohodnotenie pre 1. hraca, b pre duheho hraca atd.
        # aktualny hrac bude pozerat iba na svoju hodnotu a podla nej vybere najlepsi tah
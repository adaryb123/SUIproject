import logging
from copy import deepcopy

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from .utils import possible_turns, get_transfer_to_border

class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.players_order = players_order
        self.player_name = player_name
        self.logger = logging.getLogger('AI-WITHOUT-ML                          ')
        self.max_transfers = max_transfers
        self.stage = "TRANSFER"
        self.action_count = 0

        # rozne konstanty do maxn algoritmu
        self.TIME_THRESHOLD = 1.00
        self.MAX_POTENTIAL_MOVES_TO_FIND = 3    #malo by byt viac ale potom je pomaly
        self.MAX_POTENTIAL_MOVES_TO_PLAY = 3    #malo by byt viac ale potom je pomaly
        self.THRESHOLD = 0.4
        self.SCORE_WEIGHT = 2

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        print("ai turn start")

        # najprv vykona vsetky presuny na hranicne uzemia
        if self.stage == "TRANSFER":
            transfer = get_transfer_to_border(board, self.player_name)
            if transfer:
                if (nb_transfers_this_turn + 1 == self.max_transfers):
                    self.stage = "ATTACK"
                print("doing transfer")
                return TransferCommand(transfer[0], transfer[1])
            else:
                self.stage = "ATTACK"

        # potom vykona utoky
        if self.stage == "ATTACK":
            # ak nema cas, najde utok podla jednoduchsieho algoritmu
            if time_left <= self.TIME_THRESHOLD:
                best_move = possible_turns(self.player_name, board, self.SCORE_WEIGHT, self.THRESHOLD, self.MAX_POTENTIAL_MOVES_TO_FIND)
                if best_move:
                    source, target, dice_count = best_move[0]
                    print("doing fast attack: " + str(source.get_name()) + "--->" + str(target.get_name()))
                    self.action_count += 1
                    return BattleCommand(source.get_name(),target.get_name())


            # ak ma cas, najde utok podla podla maxn algoritmu
            else:
                maxn_move = self.maxn(board)
                if maxn_move:
                    print("doing maxn attack: " + str(maxn_move.source_name) + "--->" + str(maxn_move.target_name))
                    self.action_count += 1
                    return maxn_move

        # ak uz neexistuju vyhodne utoky, ukonci tah
        self.stage = "TRANSFER"
        print("ending turn, how many actions made this turn:" + str(self.action_count))
        self.action_count = 0
        return EndTurnCommand()

    """ funkcia vykona 1 utok a vrati modifikovanu hraciu plochu"""
    def show_board_after_attack(self, board, source, target, dice_count):
        source_area = board.get_area(source.get_name())
        target_area = board.get_area(target.get_name())
        source_area.set_dice(1)
        target_area.set_dice(dice_count-1)
        target_area.set_owner(source.get_owner_name())
        return board

    """ inicializacna cast algoritmu maxn"""
    def maxn(self, board):
        best_move = None
        best_heuristic = None
        current_player_index = self.players_order.index(self.player_name)
        last_player_index = current_player_index + len(self.players_order)-1
        current_move_index = 0

        # najdu sa vsetky vyhodne tahy
        turns = possible_turns(self.player_name, board, self.SCORE_WEIGHT, self.THRESHOLD, self.MAX_POTENTIAL_MOVES_TO_FIND)
        for source, target, dice_count in turns:
            #tah sa nasimuluje na hracej ploche
            new_board = self.show_board_after_attack(deepcopy(board), source, target, dice_count)

            # pre kazdy tah sa spocita heuristika
            heuristic = self.maxn_recursive(new_board, current_player_index, last_player_index, current_move_index + 1)

            # ak je tato heuristika lepsia ako doterajsie maximum, zapamatame si tento tah
            if self.is_better(best_heuristic, heuristic, self.player_name):
                best_move = BattleCommand(source.get_name(), target.get_name())
                best_heuristic = heuristic

        return best_move

    """ rekurzivna cast algoritmu maxn:
    Vytvori sa strom prehladavnia s hlbkou n*m, kde m je pocet hracov a n je priemerny pocet utokov ktore vykona 
    kazdy hrac. V listovych vrcholoch stromu sa stav hry ohodnoti manualne podla funkcie. V nelistovych vrcholoch 
    sa vyberie maximum z ohodnoteni priamych potomkov vrchola. (Maximum sa vybera z pohladu aktualneho hraca)"""
    def maxn_recursive(self, board, current_player_index, last_player_index, current_move_index):
        best_heuristic = None
        current_player = self.players_order[(current_player_index % len(self.players_order))]

        # ak je aktualny hrac mrtvy, preskoci sa na dalsieho hraca (ak uz nieje dalsi hrac, spocita sa heuristika)
        if len(board.get_player_areas(current_player)) == 0:
            if current_player_index == last_player_index:
                return self.calculate_heuristic(board)
            else:
                return self.maxn_recursive(board, current_player_index + 1, last_player_index, 0)

        # najdu sa vsetky vyhodne tahy
        turns = possible_turns(current_player, board, self.SCORE_WEIGHT, self.THRESHOLD, self.MAX_POTENTIAL_MOVES_TO_FIND)

        # ak aktualny hrac nema ziadne dobre tahy, prejde sa na dalsieho hraca (ak uz nieje dalsi hrac tak sa spocita heurisitka)
        if len(turns) == 0:
            if current_player_index == last_player_index:
                return self.calculate_heuristic(board)
            else:
                return self.maxn_recursive(board, current_player_index + 1, last_player_index, 0)

        for source, target, dice_count in turns:
            # tah sa nasimuluje na hracej ploche
            new_board = self.show_board_after_attack(deepcopy(board), source, target, dice_count)

            # pre kazdy tah sa spocita heuristika:
            if current_player_index == last_player_index and current_move_index == self.MAX_POTENTIAL_MOVES_TO_PLAY:
                # ak je toto posledny zo serie moznych utokov posledneho hraca, heuristika sa vyrata podla funkcie
                heuristic = self.calculate_heuristic(new_board)
            elif current_move_index == self.MAX_POTENTIAL_MOVES_TO_PLAY:
                # ak je toto posledny zo serie moznych utokov aktualneho hraca, prejde sa na dalsieho hraca
                heuristic = self.maxn_recursive(new_board, current_player_index + 1, last_player_index, 0)
            else:
                # prejde sa na dalsi zo serie moznych utokov aktualneho hraca
                heuristic = self.maxn_recursive(new_board, current_player_index, last_player_index, current_move_index + 1)

            # ak je tato heuristika lepsia ako doterajsie maximum, zapamatame si ju
            if self.is_better(best_heuristic, heuristic, current_player):
                best_heuristic = heuristic

        return best_heuristic

    """Heuristicka funkcia sa pocita zvlast pre kazdeho hraca a ulozi sa do listu.
     Kazdy hrac sa snazi zmaximalizovat svoju hodnotu.
     Tato heuristika berie do uvahy iba pocet uzemi patriacich hracovi"""
    def calculate_heuristic(self, board):
        heuristic = []
        for player in self.players_order:
            player_areas = board.get_player_areas(player)
            heuristic.append(len(player_areas))

        return heuristic

    """Tato heuristika berie do uvahy pocet uzemi patriacich hracovi minus pocet uzemi na hranici"""
    def calculate_heuristic2(self, board):
        heuristic = []
        for player in self.players_order:
            player_areas = board.get_player_areas(player)
            unstable_areas = board.get_player_border(player)
            heuristic.append(len(player_areas) - len(unstable_areas))

        return heuristic


    """Tu sa bude volat model.predict(), ked budeme mat model"""
    # def calculate_heuristic_withML(self,args):
    #     return model.predict(args)

    """Funkcia rozhodne ci je nova heuristika lepsia ako doterajsie maximum (pre konkretneho hraca)"""
    def is_better(self, best_heuristic, heuristic, current_player):
        if heuristic is None:
            return False
        elif best_heuristic is None:
            return True
        else:
            index = self.players_order.index(current_player)
            if heuristic[index] > best_heuristic[index]:
                return True
            else:
                return False
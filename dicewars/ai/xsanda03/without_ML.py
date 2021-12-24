import logging
from copy import deepcopy

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from ..utils import possible_attacks
#import funkcii z inych botov je zakazany, neskor tie funkcie treba dat do utils
from dicewars.ai.kb.move_selection import get_transfer_to_border
from dicewars.ai.dt import stei

class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.players_order = players_order
        self.player_name = player_name
        self.logger = logging.getLogger('AI-WITHOUT-ML                          ')
        self.max_transfers = max_transfers
        self.stei = stei.AI(player_name, board, players_order, max_transfers)
        self.stage = "TRANSFER"

        self.action_count = 0

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

        # potom vykona vsetky vyhodne utoky
        if self.stage == "ATTACK":
            # ak nema cas, utoci podla jednoduchsieho algoritmu
            if (time_left < 1):
                # print("no time, using stei")
                stei_move = self.stei.ai_turn(board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game,
                                          time_left)
                if isinstance(stei_move, BattleCommand):
                        print("doing stei attack: " + str(stei_move.source_name)+"--->"+str(stei_move.target_name))
                        self.action_count += 1
                        return stei_move

            # ak ma cas, utoci podla maxn algoritmu
            else:
                maxn_move = self.maxn(board)
                # maxn_move = self.maxn(board,player_name,players_order)
                if isinstance(maxn_move, BattleCommand):
                    print("doing attack: "+ str(maxn_move.source_name)+ "--->"+str(maxn_move.target_name))
                    self.action_count += 1
                    return maxn_move

        self.stage = "TRANSFER"
        print("ending turn, actions made this turn:" +str(self.action_count))
        self.action_count = 0
        return EndTurnCommand()

    """ funkcia vrati board tak ako bude vyzerat po vykonani 1 utoku zo source do target"""
    def show_board_after_attack(self,board,source,target,dice_count):
        source_area = board.get_area(source.get_name())
        target_area = board.get_area(target.get_name())
        source_area.set_dice(1)
        target_area.set_dice(dice_count-1)
        target_area.set_owner(source.get_owner_name())
        return board

    """ inicializacna cast algoritmu maxn"""
    def maxn(self,board):
        best_move = EndTurnCommand()
        best_heuristic = None
        current_player_index = self.players_order.index(self.player_name)
        last_player_index = current_player_index + len(self.players_order)-1

        current_player = self.players_order[(current_player_index % len(self.players_order))]
        # print("player orders: "+str(self.players_order))
        # print("maxn, CPI=" + str(current_player_index) + ", CP=" + str(current_player))

        for source,target in possible_attacks(board, current_player):
            new_board = self.show_board_after_attack(deepcopy(board),source,target,source.get_dice())
            heuristic = self.maxn_recursive(new_board, current_player_index + 1, last_player_index)

            if self.is_better(best_heuristic, heuristic, current_player):
                best_move = BattleCommand(source.get_name(), target.get_name())
                best_heuristic = heuristic

        #print("maxn end")
        return best_move

    """ rekurzivna cast algoritmu maxn"""
    def maxn_recursive(self,board,current_player_index, last_player_index):
        best_heuristic = None
        current_player = self.players_order[(current_player_index % len(self.players_order))]

        #print("maxn recursive, CPI=" + str(current_player_index) + ", CP=" + str(current_player))

        if len(board.get_player_areas(current_player)) == 0:
            return self.maxn_recursive(board, current_player_index + 1, last_player_index)

        for source, target in possible_attacks(board, current_player):
            new_board = self.show_board_after_attack(deepcopy(board), source, target, source.get_dice())

            if current_player_index == last_player_index:
                heuristic = self.calculate_heuristic(new_board)
            else:
                heuristic = self.maxn_recursive(new_board, current_player_index + 1, last_player_index)

            if self.is_better(best_heuristic, heuristic, current_player):
                best_heuristic = heuristic

        #print("best heuristic for player: "+str(current_player) + " is: "+str(best_heuristic))
        return best_heuristic

    """Funkcia ohodnoti poziciu kazdeho hraca v aktualnom stave hry a
    vrati list hodnot-pre kazdeho hraca jednu
    Momentalne sa heuristika pocita ako pocet uzemi patriacich danemu hracovi"""
    def calculate_heuristic(self,board):
        heuristic =[]
        for player in self.players_order:
            player_areas = board.get_player_areas(player)
            heuristic.append(len(player_areas))

        return heuristic

    def calculate_heuristic2(self,board):
        heuristic =[]
        for player in self.players_order:
            player_areas = board.get_player_areas(player)
            unstable_areas = board.get_player_border(player)
            heuristic.append(len(player_areas) - len(unstable_areas))

        return heuristic


    """Tu sa bude volat model.predict(), ked budeme mat model"""
    # def calculate_heuristic_withML(self,args):
    #     return model.predict(args)

    """Funkcia rozhodne ci je nova heuristika lepsia ako doterajsie maximum (pre konkretneho hraca)"""
    def is_better(self,best_heuristic,heuristic,current_player):
        if best_heuristic is None:
            return True
        elif heuristic is None:
            return False
        else:
            index = self.players_order.index(current_player)
            if heuristic[index] > best_heuristic[index]:
                return True
            else:
                return False

#PROBLEM: nas algoritmus neberie do uvahy uspesnost utoku (tak ako to robi stei)
#PROBLEM: nas algoritmus berie do uvahy iba 1 akciu kazdeho protihraca
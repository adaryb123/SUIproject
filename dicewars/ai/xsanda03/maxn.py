"""
Implementacia AI pomocou MaxN algoritmu

Autori:     Michal Sandanus   xsanda03
            David Pukanec     xpukan02
            Adam Rybansky     xryban00
            Filip Gutten      xgutte00

Nazov timu:     xsanda03
"""

import logging
from copy import deepcopy
import torch
from torch import nn
import numpy as np

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from .utils import possible_turns, get_transfer_to_border


class NN_heuristic(nn.Module):
    """
    Class representing NN model for calculating heuristic

    """
    def __init__(self):
        super().__init__()

        #Input layer
        self.input_layer = nn.Linear(in_features=633, out_features=128, bias=True)

        #Hidden layers
        self.hidden_layer1 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.hidden_layer2 = nn.Linear(in_features=64, out_features=32, bias=True)

        #Output layer
        self.output_layer = nn.Linear(in_features=32, out_features=4, bias=True)

        #Defining sigmoid activation function and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_vector):
        output = self.input_layer(input_vector)
        output = self.sigmoid(output)
        output = self.hidden_layer1(output)
        output = self.sigmoid(output)
        output = self.hidden_layer2(output)
        output = self.sigmoid(output)
        output = self.output_layer(output)
        output = self.softmax(output)

        return output

class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.players_order = players_order
        self.player_name = player_name
        self.logger = logging.getLogger('AI-maxn')
        self.max_transfers = max_transfers
        self.stage = "TRANSFER"     # The AI makes all the transfers first, then changes the stage to ATTACK and starts to attack
        self.action_count = 0
        self.board = board
        # since one round consists of several attack, here we store possible attack that we found while searching
        # the beast attack so we don't have to look for them when looking for the next best attack
        self.stored_turns_and_heuristics = []

        # various constans for maxn algorithm
        self.TIME_THRESHOLD = 1.00
        self.MAX_POTENTIAL_MOVES_TO_FIND = 3
        self.MAX_POTENTIAL_MOVES_TO_PLAY = 3
        self.THRESHOLD = 0.4
        self.SCORE_WEIGHT = 2

        # Initialize NN and adjacency areas for game
        self.adj_array = self.init_adj_board(board)
        self.network = NN_heuristic()
        # Model loading
        self.network.load_state_dict(torch.load(r"dicewars/ai/xsanda03/model-heuristic.pth", map_location='cpu'))
        self.network.eval()



    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        # at first, it makes all transfers to the border area
        if self.stage == "TRANSFER":
            transfer = get_transfer_to_border(board, self.player_name)
            if transfer:
                if nb_transfers_this_turn + 1 == self.max_transfers:
                    self.stage = "ATTACK"
                return TransferCommand(transfer[0], transfer[1])
            else:
                self.stage = "ATTACK"

        # then it performs attacks
        if self.stage == "ATTACK":
            maxn_move = self.maxn(board, time_left)
            if maxn_move is not None:
                self.action_count += 1
                return maxn_move

        # if there are no more advantageous attack, it ends the move
        self.stage = "TRANSFER"
        self.action_count = 0
        self.stored_turns_and_heuristics.clear()
        return EndTurnCommand()


    """The function simulates 1 attack and returns the modified game board"""
    def show_board_after_attack(self, board, source, target, dice_count):
        source_area = board.get_area(source.get_name())
        target_area = board.get_area(target.get_name())
        source_area.set_dice(1)
        target_area.set_dice(dice_count-1)
        target_area.set_owner(source.get_owner_name())
        return board

    """The function simulates 1 transfer and returns the modified game board"""
    def show_board_after_transfer(self, board, source, target):
        source_area = board.get_area(source)
        target_area = board.get_area(target)
        source_dice_count = source_area.get_dice()
        target_dice_count = target_area.get_dice()
        transfered_dice_count = min(source_dice_count - 1, 8 - target_dice_count)
        source_area.set_dice(source_dice_count - transfered_dice_count)
        target_area.set_dice(target_dice_count + transfered_dice_count)
        return board

    """The function decides whether the new heuristic is better than the current maximum (for a specific player)"""
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


    """The function deletes from saved attacks those than can no longer be performed"""
    def update_stored_turns(self, board):
        if len(self.stored_turns_and_heuristics) == 0:
            return

        index = 0
        for turn, heuristic in self.stored_turns_and_heuristics:
            source, target, dice_count = turn
            if board.get_area(target.get_name()).get_owner_name() != target.get_owner_name():
                self.stored_turns_and_heuristics[index] = -1
            elif board.get_area(source.get_name()).get_dice() != dice_count:
                self.stored_turns_and_heuristics[index] = -1

            index += 1

        self.stored_turns_and_heuristics = list(filter(lambda a: a != -1, self.stored_turns_and_heuristics))


    """The function finds the best of saved attacks"""
    def find_best_stored_turn(self, current_player):
        best_turn = None
        best_heuristic = None

        if len(self.stored_turns_and_heuristics) == 0:
            return None, None

        for turn, heuristic in self.stored_turns_and_heuristics:
            source, target, dice_count = turn
            if best_turn is None:
                best_turn = BattleCommand(source.get_name(), target.get_name())
                best_heuristic = heuristic
            elif self.is_better(best_heuristic, heuristic, current_player):
                best_turn = BattleCommand(source.get_name(), target.get_name())
                best_heuristic = heuristic

        return best_turn, best_heuristic


    """The function compares whether 2 attacks are the same"""
    def compare_turns(self, turn1, turn2):
        source1, target1, dice_count1 = turn1
        source2, target2, dice_count2 = turn2
        if source1.get_name() == source2.get_name() and target1.get_name() == target2.get_name():
            return True
        else:
            return False


    """Initialization part of the maxn algorithm"""
    def maxn(self, board, time_left):
        current_player_index = self.players_order.index(self.player_name)
        last_player_index = current_player_index + len(self.players_order)-1
        current_move_index = 0

        # it looks to see if we have saved any attacks from the past that are still valid
        self.update_stored_turns(board)
        best_move, best_heuristic = self.find_best_stored_turn(self.player_name)

        # all the good moves are found
        turns = possible_turns(self.player_name, board, self.SCORE_WEIGHT, self.THRESHOLD, self.MAX_POTENTIAL_MOVES_TO_FIND)
        for turn in turns:

            # if we have already saved this attack, we would not search it again
            can_skip = False
            for stored_turn, stored_heuristic in self.stored_turns_and_heuristics:
                if self.compare_turns(stored_turn, turn):
                    can_skip = True
            if can_skip:
                continue

            # the width and depth of the scan is determined according to the amount of time that is left
            if time_left <= 1.00:
                self.MAX_POTENTIAL_MOVES_TO_FIND = 1
                self.MAX_POTENTIAL_MOVES_TO_PLAY = 1
            elif time_left <= 5.00:
                self.MAX_POTENTIAL_MOVES_TO_FIND = 1
                self.MAX_POTENTIAL_MOVES_TO_PLAY = 3
            else:
                self.MAX_POTENTIAL_MOVES_TO_FIND = 3
                self.MAX_POTENTIAL_MOVES_TO_PLAY = 3

            source, target, dice_count = turn
            # the move is simulated on the playing field
            new_board = self.show_board_after_attack(deepcopy(board), source, target, dice_count)

            # the heuristic is calculated for every move
            heuristic = self.maxn_recursive(new_board, current_player_index, last_player_index, current_move_index + 1)

            self.stored_turns_and_heuristics.append([turn, heuristic])
            # if this heuristic is better than the current maximum, this move is remembered
            if self.is_better(best_heuristic, heuristic, self.player_name):
                best_move = BattleCommand(source.get_name(), target.get_name())
                best_heuristic = heuristic

        return best_move

    """ The recursive part of the maxn algorithm:
    A search tree with a depth of n * m is created, where m is the number of players and n is the average number
    attacks performed by every player. The state of the game is evaluated manually according to the function
    in the leaf tops of the tree. In the non-leaf tops of the tree the maximum is selected from the evaluation
    of the direct descendants of the vertex. (maximum is selected from the current player's view) """
    def maxn_recursive(self, board, current_player_index, last_player_index, current_move_index):
        best_heuristic = None
        current_player = self.players_order[(current_player_index % len(self.players_order))]

        if current_move_index == 0:
            for i in range(self.max_transfers):
                transfer = get_transfer_to_border(board, current_player)
                if transfer is None:
                    break
                else:
                    source, destination = transfer
                    board = self.show_board_after_transfer(board,source,destination)


        # if the current player is dead, there is jump to the next player (if there is no other player,
        # heuristic is calculated)
        if len(board.get_player_areas(current_player)) == 0:
            if current_player_index == last_player_index:
                return self.calculate_heuristic(board)
            else:
                return self.maxn_recursive(board, current_player_index + 1, last_player_index, 0)

        # all the good moves are found
        turns = possible_turns(current_player, board, self.SCORE_WEIGHT, self.THRESHOLD, self.MAX_POTENTIAL_MOVES_TO_FIND)

        # if the current player does not have any good moves, it moves to the next player (if there is no other player,
        # the heuristic is calculated
        if len(turns) == 0:
            if current_player_index == last_player_index:
                return self.calculate_heuristic(board)
            else:
                return self.maxn_recursive(board, current_player_index + 1, last_player_index, 0)

        for source, target, dice_count in turns:
            # the move is simulated on the playing field
            new_board = self.show_board_after_attack(deepcopy(board), source, target, dice_count)

            # the heuristic is calculated for every move:
            if current_player_index == last_player_index and current_move_index == self.MAX_POTENTIAL_MOVES_TO_PLAY:
                # if this is the last of a series of possible attacks, the heuristic is calculated according
                # to the function
                heuristic = self.calculate_heuristic(new_board)
            elif current_move_index == self.MAX_POTENTIAL_MOVES_TO_PLAY:
                # if this is the last of a series of possible attacks by the current player, it moves
                # to the next player
                heuristic = self.maxn_recursive(new_board, current_player_index + 1, last_player_index, 0)
            else:
                # it moves to the next in a series of possible attacks by the current player
                heuristic = self.maxn_recursive(new_board, current_player_index, last_player_index, current_move_index + 1)

            # if this heuristic is better than the current maximum, it is remembered
            if self.is_better(best_heuristic, heuristic, current_player):
                best_heuristic = heuristic

        return best_heuristic


    """The heuristic function is calculated separately for every player and stored in the list.
       Every player tries to maximize their value"""

    """This is the best heuristic. Others are in the file heuristics.py"""
    """This heuristic take into account the total number of territories of the player and a number of player's
       territories at the border that can attack"""
    """def calculate_heuristic(self, board):
        heuristic = []
        for player in self.players_order:
            player_areas = board.get_player_areas(player)

            can_attack = 0
            unstable_areas = board.get_player_border(player)
            for unstable_area in unstable_areas:
                if unstable_area.can_attack():
                    can_attack += 1

            heuristic.append(can_attack + 0.5 * len(player_areas))

        return heuristic
        """
    def init_adj_board(self, board):
        """
        Creates adjacency matrix and converts it into vector.

        Returns
        np.array: adjacency matrix
        """
        areas = {}
        # For every area on board get adjacent areas
        for a in board.areas:
            area = board.areas[a]
            areas[a] = area.get_adjacent_areas_names()
        # Calculate and init len of final array
        area_len = len(areas)
        adj_len = int((area_len * (area_len - 1)) / 2)
        adj_array = np.zeros(adj_len)
        # For every area
        for i in range(0,area_len):
            adj_areas = areas[str(i+1)]
            # For every adjacent area
            for j in adj_areas:
                # Add 1 to indicate adjacency
                if (j > i + 1):
                    offset = int((i / 2) * ((area_len - 1) + (area_len - i)))
                    adj_array[offset + (int(j) - 1 - (i+1) )] = 1
        return adj_array

    def get_game_state(self, board):

        """
        Function creates vector representing current game state.

        Returns
        np.array: adjacency matrix, owner and dice count, biggest area for player
        """
        final_array = np.array(deepcopy(self.adj_array))

        # add owner and dice count
        for a in board.areas:
            area = [board.areas[a].get_owner_name(), board.areas[a].get_dice()]
            final_array = np.append(final_array,area)
        # add the biggest area
        for p in range(len(self.players_order)):

            cmax = -1
            for i in board.get_players_regions(p+1):
                if len(i) > cmax:
                    cmax = len(i)
            final_array = np.append(final_array,cmax)

        return  final_array

    """ Function calculates heuristic using the trained model """
    def calculate_heuristic(self, board):
        # Extract all information needed for NN evaluation
        turn_info = self.get_game_state(board)
        # Convert torch tensor into proper shape and type
        turn_info = torch.from_numpy(turn_info)
        turn_info = turn_info.type(torch.float32)
        turn_info = torch.unsqueeze(turn_info,0)

        # Evaluation of game state
        with torch.no_grad():
            heuristic = self.network(turn_info).tolist()
        # Returns list of list with scores for every player
        heuristic = heuristic[0]
        return heuristic

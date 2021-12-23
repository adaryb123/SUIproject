import logging

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.kb.move_selection import get_transfer_to_border
from dicewars.ai.dt import stei

#this is the best possible AI that i was able to make from the already created agents
#first it transfers all forces to border and then attacks using stei from dt.stei.py

class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI-COPIED                             ')
        self.max_transfers = max_transfers
        self.stei = stei.AI(player_name, board, players_order, max_transfers)
        self.stage = "TRANSFER"  # Available stages: "TRANSFER", "ATTACK_BEST" , "ATTACK_FAST" , "END_TURN"

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        if self.stage == "TRANSFER":
            transfer = get_transfer_to_border(board, self.player_name)
            if transfer:
                if (nb_transfers_this_turn + 1 == self.max_transfers):
                    self.stage = "ATTACK_BEST"
                return TransferCommand(transfer[0], transfer[1])
            else:
                self.stage = "ATTACK_BEST"

        if self.stage == "ATTACK_BEST":
            stei_move = self.stei.ai_turn(board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game,
                                          time_left)
            if isinstance(stei_move, BattleCommand):
                return stei_move

        self.stage = "TRANSFER"
        return EndTurnCommand()
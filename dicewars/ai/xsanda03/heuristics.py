"""
Here are located all the heuristics we have tried
"""

"""This heuristic takes into account only the number of areas belonging to the player"""
def calculate_heuristic(self, board):
    heuristic = []
    for player in self.players_order:
        player_areas = board.get_player_areas(player)
        heuristic.append(len(player_areas))

    return heuristic

"""This heuristic takes into account difference of the number of areas belonging to the player
   and number of areas at the border"""
def calculate_heuristic2(self, board):
    heuristic = []
    for player in self.players_order:
        player_areas = board.get_player_areas(player)
        unstable_areas = board.get_player_border(player)
        heuristic.append(len(player_areas) - len(unstable_areas))

    return heuristic

"""This heuristic takes into account the total number of dice"""
def calculate_heuristic3(self, board):
    heuristic = []
    for player in self.players_order:
        dices = board.get_player_dice(player)
        heuristic.append(dices)

    return heuristic

"""This heuristic calculates the average number of dice in the border territory"""
def calculate_heuristic6(self, board):
    heuristic = []
    for player in self.players_order:
        avg_dices_on_borders = 0
        unstable_areas = board.get_player_border(player)
        borders = len(unstable_areas)
        for unstable_area in unstable_areas:
            avg_dices_on_borders += (unstable_area.get_dice()/borders)
        heuristic.append(avg_dices_on_borders)
    return heuristic

"""This heuristic calculates the total number of dice in the border territory"""
def calculate_heuristic6_B(self, board):
    heuristic = []
    for player in self.players_order:
        dices_on_borders = 0
        unstable_areas = board.get_player_border(player)
        for unstable_area in unstable_areas:
            dices_on_borders += unstable_area.get_dice()
        heuristic.append(dices_on_borders)
    return heuristic

"""This heuristic calculates the number of areas at the border that are able to fight"""
def calculate_heuristic7(self, board):
    heuristic = []
    for player in self.players_order:
        can_attack = 0
        unstable_areas = board.get_player_border(player)
        for unstable_area in unstable_areas:
            if unstable_area.can_attack():
                can_attack += 1
        heuristic.append(can_attack)
    return heuristic

"""This heuristic cumulates 1. and 7. heuristic"""
def calculate_heuristic1_7(self, board):
    heuristic1 = self.calculate_heuristic(board)
    heuristic2 = self.calculate_heuristic7(board)
    zipped = zip(heuristic1, heuristic2)
    return [x + y for (x, y) in zipped]

"""This heuristic cumulates the half weight of 1. and the whole weight of 7. heuristic"""
def calculate_heuristic1_7_B(self, board):
    heuristic1 = self.calculate_heuristic(board)
    heuristic2 = self.calculate_heuristic7(board)
    zipped = zip(heuristic1, heuristic2)
    return [x/2 + y for (x, y) in zipped]

"""This heuristic cumulates the one third weight of 1. and the whole weight of 7. heuristic"""
def calculate_heuristic1_7_C(self, board):
    heuristic1 = self.calculate_heuristic(board)
    heuristic2 = self.calculate_heuristic7(board)
    zipped = zip(heuristic1, heuristic2)
    return [x / 3 + y for (x, y) in zipped]

"""This heuristic cumulates the fifteen tenth weight of 1. and the whole weight of 7. heuristic"""
def calculate_heuristic1_7_D(self, board):
    heuristic1 = self.calculate_heuristic(board)
    heuristic2 = self.calculate_heuristic7(board)
    zipped = zip(heuristic1, heuristic2)
    return [x / 1.5 + y for (x, y) in zipped]


"""This heuristic takes into account only the number of areas belonging to the player 
    and the average number of dice in the border territory"""
def calculate_heuristic(self, board):
    heuristic = []
    for player in self.players_order:
        player_areas = board.get_player_areas(player)
        heuristic.append(len(player_areas))

        avg_dices_on_borders = 0
        unstable_areas = board.get_player_border(player)
        borders = len(unstable_areas)
        for unstable_area in unstable_areas:
            avg_dices_on_borders += (unstable_area.get_dice() / borders)
        heuristic.append(len(player_areas) * 1.7 + avg_dices_on_borders)

    return heuristic

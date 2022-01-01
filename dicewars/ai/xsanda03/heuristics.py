"""Tu sa nachadzaju vsetky heuristiky ktore sme skusali
   """


"""Tato heuristika berie do uvahy iba pocet uzemi patriacich hracovi"""
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

"""Tato heuristika berie do uvahy celkový počet kociek"""
def calculate_heuristic3(self, board):
    heuristic = []
    for player in self.players_order:
        dices = board.get_player_dice(player)
        heuristic.append(dices)

    return heuristic

"""Tato heuristika kalkuluje tzv. enemy index, počet rôznych dvojíc susediacich nepriateľov"""
"""padajúcich na 1 hraničnú oblasť hráča"""
"""nefunguje"""
def calculate_heuristic5(self, board):
    heuristic = []
    for player in self.players_order:
        enemy_index_of_area = 0
        unstable_areas = board.get_player_border(player)
        # print("\n1.cyklus")
        for unstable_area in unstable_areas:
            # print(unstable_area.get_owner_name())
            adjacent_areas = unstable_area.get_adjacent_areas_names()
            owner1 = -1
            # print("\n2.cyklus")
            for adjacent_area in adjacent_areas:
                # print("\n3.cyklus")
                # print(adjacent_area)
                owner2 = adjacent_area.get_owner_name()
                # print("\nprešlo?")
                if owner2 != player:
                    if owner2 != owner1:
                        enemy_index_of_area += 1
                        owner1 = owner2

        heuristic.append(enemy_index_of_area)

    return heuristic

"""Tato heuristika kalkuluje priemerný počet kociek na hraničnom území"""
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

"""Tato heuristika kalkuluje celkový počet kociek na hraničnom území"""
def calculate_heuristic6_B(self, board):
    heuristic = []
    for player in self.players_order:
        dices_on_borders = 0
        unstable_areas = board.get_player_border(player)
        for unstable_area in unstable_areas:
            dices_on_borders += unstable_area.get_dice()
        heuristic.append(dices_on_borders)
    return heuristic

"""Tato heuristika kalkuluje počet bojaschopných území na hranici"""
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

"""Tato heuristika kalkuluje počet bojaschopných území"""
"""nefunguje"""
def calculate_heuristic7_B(self, board):
    heuristic = []
    for player in self.players_order:
        can_attack = 0
        unstable_areas = board.get_player_border(player)
        for unstable_area in unstable_areas:
            if unstable_area.can_attack():
                can_attack += 1
        heuristic.append(can_attack)
    return heuristic

"""Tato heuristika kumuluje 1. a 7. heuristiku"""
def calculate_heuristic1_7(self, board):
    heuristic1 = self.calculate_heuristic(board)
    heuristic2 = self.calculate_heuristic7(board)
    zipped = zip(heuristic1, heuristic2)
    return [x + y for (x, y) in zipped]

"""Tato heuristika kumuluje polovičnú váhu 1. a plnú váhu 7. heuristiky"""
def calculate_heuristic1_7_B(self, board):
    heuristic1 = self.calculate_heuristic(board)
    heuristic2 = self.calculate_heuristic7(board)
    zipped = zip(heuristic1, heuristic2)
    return [x/2 + y for (x, y) in zipped]

"""Tato heuristika kumuluje tretinovú váhu 1. a plnú váhu 7. heuristiky"""
def calculate_heuristic1_7_C(self, board):
    heuristic1 = self.calculate_heuristic(board)
    heuristic2 = self.calculate_heuristic7(board)
    zipped = zip(heuristic1, heuristic2)
    return [x / 3 + y for (x, y) in zipped]

def calculate_heuristic1_7_D(self, board):
    heuristic1 = self.calculate_heuristic(board)
    heuristic2 = self.calculate_heuristic7(board)
    zipped = zip(heuristic1, heuristic2)
    return [x / 1.5 + y for (x, y) in zipped]

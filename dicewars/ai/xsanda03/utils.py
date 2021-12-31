"""
Tu sa nachadzaju funkcie, ktore sme prebrali z existujucich implementacii AI v projekte,
niekde sme pridali drobne upravy
"""


from ..utils import possible_attacks
from ..utils import probability_of_holding_area, probability_of_successful_attack

"""Funkcia vrati najlepsiu postupnost utokov ktore moze hrac vykonat v svojom tahu.
   Pocet utokov je obmedzeny parametrom ammount"""
def possible_turns(player_name, board, score_weight, treshold, ammount):

    largest_region = get_largest_region(board,player_name)
    all_turns = []
    for source, target in possible_attacks(board, player_name):
        atk_power = source.get_dice()
        atk_prob = probability_of_successful_attack(board, source.get_name(), target.get_name())
        hold_prob = atk_prob * probability_of_holding_area(board, target.get_name(), atk_power - 1, player_name)
        if hold_prob >= treshold or atk_power == 8:
            preference = hold_prob
            if source.get_name() in largest_region:
                preference *= score_weight
            all_turns.append([source, target, preference, hold_prob, source.get_dice()])

    all_turns = sorted(all_turns, key=lambda turn: turn[2], reverse=True)
    best_turns = []
    for i in range(0,min(ammount,len(all_turns))):
        source, target, preference, hold_prob, dice_count = all_turns[i]
        best_turns.append([source,target,dice_count])
    return best_turns

"""Funkcia najde najvacsie suvisle uzemie daneho hraca"""
def get_largest_region(board, player_name):
    players_regions = board.get_players_regions(player_name)
    max_region_size = max(len(region) for region in players_regions)
    max_sized_regions = [region for region in players_regions if len(region) == max_region_size]
    largest_region = max_sized_regions[0]
    return largest_region

"""funkcia najde mozny presun jednotiek z vnutorneho uzemia na hranicu"""
def get_transfer_to_border(board, player_name):
    border_names = [a.name for a in board.get_player_border(player_name)]
    all_areas = board.get_player_areas(player_name)
    inner = [a for a in all_areas if a.name not in border_names]

    for area in inner:
        if area.get_dice() < 2:
            continue

        for neigh in area.get_adjacent_areas_names():
            if neigh in border_names and board.get_area(neigh).get_dice() < 8:
                return area.get_name(), neigh

    return None
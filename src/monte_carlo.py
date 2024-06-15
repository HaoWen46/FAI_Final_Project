from game.players import BasePokerPlayer
from collections import Counter
import random
import copy

class Card:
    SUITS = 'CDHS'
    RANKS = '23456789TJQKA'
    SUITS_VAL = {s: i for i, s in enumerate(SUITS)}
    RANKS_VAL = {s: i for i, s in enumerate(RANKS)}
    
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.rank_val = self.RANKS_VAL[rank]
        self.suit_val = self.SUITS_VAL[suit]
        
class Deck:
    def __init(self):
        self.cards = [Card(rank, suit) for suit in Card.SUITS for rank in Card.RANKS]
        random.shuffle(self.cards)
    
    def draw(self, count=1):
        return [self.cards.pop() for _ in range(count)]
    
class Player(BasePokerPlayer):
    def evaluate(self, hand, community_cards):
        all_cards = hand + community_cards
        ranks = [card.rank for card in all_cards]
        rank_counts = Counter(ranks).values()
        if 4 in rank_counts:
            return 7  # Four of a kind
        elif 3 in rank_counts and 2 in rank_counts:
            return 6  # Full house
        elif 3 in rank_counts:
            return 3  # Three of a kind
        elif list(rank_counts).count(2) == 2:
            return 2  # Two pair
        elif 2 in rank_counts:
            return 1  # One pair
        else:
            return 0  # High card
        
    def declare_action(self, valid_actions, hole_card, round_state):
        N = 1000
        wins, ties = 0
        original_deck = Deck()
        known_cards = []
        player_hand = []
        community_cards = []
        for card in hole_card:
            player_hand.append(Card(card[1], card[0]))
            known_cards.append(player_hand[-1])
            
        for card in round_state['community_card']:
            community_cards.append(Card(card[1], card[0]))
            known_cards.append(community_cards[-1])
            
        original_deck.cards = [card for card in original_deck.cards if card not in known_cards]
        
        for _ in range(N):
            deck = copy.deepcopy(original_deck)
            random.shuffle(deck.cards)
            opponent_hand = deck.draw(2)
            
            remaining_community_cards = community_cards[:]
            while len(remaining_community_cards) < 5:
                remaining_community_cards.append(deck.draw()[0])
                
            player_score = self.evaluate(player_hand, community_cards)
            opponent_score = self.evaluate(opponent_hand, community_cards)
            
            if player_score > opponent_score:
                wins += 1
            elif player_score == opponent_score:
                ties += 1
        
        score = wins / N + (ties / N) / 2
        
        min_amount = valid_actions[2]['amount']['min']
        max_amount = valid_actions[2]['amount']['max']
        
        if score > 0.8:
            return 'raise', min_amount + (max_amount - min_amount) * 0.4
        elif score > 0.7:
            return 'raise', min_amount + (max_amount - min_amount) * 0.3
        elif score > 0.6:
            return 'raise', min_amount + (max_amount - min_amount) * 0.2
        elif score > 0.5:
            return 'raise', min_amount + (max_amount - min_amount) * 0.1
        elif score > 0.4:
            return 'call', 0
        else:
            return 'fold', 0
                
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
    
def setup_ai():
    return Player()
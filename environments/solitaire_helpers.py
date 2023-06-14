# credit: https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
def first_nonzero_index(array):
    """Return the index of the first non-zero element of array. If all elements are zero, return -1."""

    fnzi = -1 # first non-zero index
    indices = np.flatnonzero(array)

    if (len(indices) > 0):
        fnzi = indices[0]

    return fnzi


# gets suit number given card num (worried about it always returning 0, shouldnt it be -1?)
def get_suit(card_num):
    if card_num / 13 <= 1:
        return 0
    elif card_num / 26 <= 1:
        return 1
    elif card_num / 39 <= 1:
        return 2
    else:
        return 3
    return 0


# gets suit and card num, given card num
def get_suit_and_num(card_num):
    if card_num == 0 or card_num == 53:
        return [0, card_num]

    suit = 0
    num = card_num % 13
    if num == 0:
        num = 13

    if card_num / 13 <= 1:
        suit = 0
    elif card_num / 26 <= 1:
        suit = 1
    elif card_num / 39 <= 1:
        suit = 2
    else:
        suit = 3

    return [suit, num]


# gets character from the suit (for displaying images)
def get_suit_char_from_val(suit_val):
    if suit_val == 0:
        return "H"
    elif suit_val == 1:
        return "D"
    elif suit_val == 2:
        return "S"
    elif suit_val == 3:
        return "C"


# gets character from the card num (for displaying images)
def get_card_char(card_num):
    if card_num == 1:
        return "A"
    elif card_num == 10:
        return "T"
    elif card_num == 11:
        return "J"
    elif card_num == 12:
        return "Q"
    elif card_num == 13:
        return "K"
    else:
        return str(int(card_num))


# returns an np array size 52 with the numbers 1-53(non-inc), False-resampled
def get_shuffled_deck(np_random):
    return np_random.choice(range(1,53), 52, False)


# checks if can move card from deck to suit
def deck_to_suit_check(deck_cards_param, suit_cards_param, highest_nonzero_deck):
    # -1 is encoded in this variable as the row is entirely empty, so cant move anything from empty deck top row
    if highest_nonzero_deck > -1:
        # make sure it can slot into the pile properly
        active_deck_card = deck_cards_param[0, highest_nonzero_deck]
        active_deck_card_suit_and_num = get_suit_and_num(active_deck_card)
        a_suit = active_deck_card_suit_and_num[0]
        a_num = active_deck_card_suit_and_num[1]
        # if suit cards is one below active deck card number, then can add to suits
        if suit_cards_param[a_suit] + 1 == a_num:
            return True, a_suit
    return False, False


# checks if can move from deck to pile
def deck_to_pile_check(deck_cards_param, pile_cards_param, pile_i, highest_nonzero_deck):
    # -1 is encoded in this variable as the row is entirely empty, so cant move anything from empty deck top row
    if highest_nonzero_deck > -1:
        # get suit and num of the deck card
        c_suit_and_num = get_suit_and_num(deck_cards_param[0, highest_nonzero_deck])
        c_suit = c_suit_and_num[0]
        c_num = c_suit_and_num[1]

        # check if deck card is king, if so just check if pile bottom most is empty
        if c_num == 13:
            if pile_cards_param[0, pile_i] == 0:
                return True
        else:

            # check pile bottom most is not empty
            if pile_cards_param[0, pile_i] > 0:

                # check the pile you are trying to move to's bottom most card is one lower and different suit
                t_suit_and_num = get_suit_and_num(pile_cards_param[0,pile_i])

                # check opposite suits and current card is one lower than target
                if ((c_suit <= 1 and t_suit_and_num[0] >= 2) or (c_suit >= 2 and t_suit_and_num[0] <= 1)) and (c_num + 1 == t_suit_and_num[1]):
                    return True

    return False


# checks if can move from suit down to pile
def suit_to_pile_check(suit_cards_param, pile_cards_param, pile_i, suit_j):
    # check suit card you are trying to move exists, i.e. greater than zero
    # and also greater than one because why consider moving ace down?
    c_num = suit_cards_param[suit_j]
    if c_num > 1:

        # first, check it isnt a king edge case
        if c_num == 13:

            # check if pile trying to move to is empty
            if pile_cards_param[0, pile_i] == 0:
                return True

        else:

            # check pile bottom most is not empty
            if pile_cards_param[0, pile_i] > 0:

                # check the pile you are trying to move to's bottom most card is one lower and different suit
                t_suit_and_num = get_suit_and_num(pile_cards_param[0,pile_i])

                # check opposite suits and current card is one lower than target
                if ((suit_j <= 1 and t_suit_and_num[0] >= 2) or (suit_j >= 2 and t_suit_and_num[0] <= 1)) and (c_num + 1 == t_suit_and_num[1]):
                    return True

    return False


# check if can move from pile to suit
def pile_to_suit_check(pile_cards_param, suit_cards_param, pile_i):
    # check there is a card in the pile
    if pile_cards_param[0,pile_i] > 0:
        # get suit and number of bottom most pile card
        a_suit_and_num = get_suit_and_num(pile_cards_param[0,pile_i])
        # check that the card is indeed one higher than what is current in its suit
        if a_suit_and_num[1] - 1 == suit_cards_param[a_suit_and_num[0]]:
            return True, a_suit_and_num[0]
    return False, False


# check that agent can move some card index (and all the cards below it) to
# another pile
# index 0 is lowest card
# TODO: consider removing ability to move ace, again when would you ever move ace pile to pile?
# the reason this is happening is because yes a card can be at index 0 in the pile and not be
# an ace, for example on reset every card spawns. We need to do a
def pile_to_pile_check(pile_cards_param, pile_i, pile_to_move_to_j, card_k, highest_nonzero_pile_i):
    # check current pile is not empty
    if highest_nonzero_pile_i > -1:
        # check that there is a card which is a number
        # of cards below the top one you want to move
        # i.e. if the pile is ace,2,3,4, and card_k=7, that is out of bounds
        # max would be card_k = 3, which would move ace. card_k = 0 moves card 4
        # BELOW MAYBE DIFFERENT? where do I see 12?
        # only 12 actions because in k,q,j,10,9,8,7,6,5,4,3,2,ace, it would only
        # ever make sense to go down to the 2 and move that, otherwise
        # send the ace to the suits
        # highest nonzero pile i here should be 3
        if card_k <= highest_nonzero_pile_i:
            # get the index of the current card you want to move. In the above example
            # c_k = 3 would mean we move index 0, the ace, exactly. c_k=0 means we move
            # index 3, the 4, exactly
            index_to_move = highest_nonzero_pile_i - card_k
            # get the current top card trying to move's suit and num
            c_suit_and_num = get_suit_and_num(pile_cards_param[index_to_move,pile_i])

            if c_suit_and_num[1] == 1:
                return False

            # if the current card we want to move is a king, check the target pile is empty
            if c_suit_and_num[1] == 13:
                if not pile_cards_param[:,pile_to_move_to_j].any():
                    return True
            else:
                # else its some other card, so check the pile we want to move to has
                # card at index 0, and that it is one lower and different suit than current
                if pile_cards_param[0,pile_to_move_to_j] > 0:
                    t_suit_and_num = get_suit_and_num(pile_cards_param[0,pile_to_move_to_j])
                    # check diff suit and one higher
                    if ((c_suit_and_num[0] <= 1 and t_suit_and_num[0] >= 2) or (c_suit_and_num[0] >= 2 and t_suit_and_num[0] <= 1)) and (c_suit_and_num[1] + 1 == t_suit_and_num[1]):
                        return True

    return False

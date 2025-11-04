from numpy.random import choice

def weighted_shuffle(ls_elems, ls_weights):
    return choice(ls_elems, 
                  len(ls_elems),
                  replace = False,
                  p = [weight/sum(ls_weights) for weight in ls_weights])
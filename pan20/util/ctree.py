"""Constituency Tree features."""
import benepar
import numpy as np


# these are internal nodes, i.e. not POS tags
wanted = ['NP', 'VP', 'SBAR', 'S', 'PP']


class GetTree:

    def __init__(self):
        self.parser = benepar.Parser('benepar_en2')

    def __call__(self, sent):
        return self.parser.parse(sent)


def avg_branch_factor(tree):
    branching_factors = []
    stack = [tree]
    while len(stack) > 0:
        node = stack[-1]
        if len(node) > 0:  # is internal node
            branching_factors.append(len(node))
            for child in node:
                if child.label() in wanted:
                    stack.append(child)
        stack.remove(node)
    return np.mean(branching_factors)


def height(tree):
    return tree.height()


def max_const_height(tree, const):
    h = 0
    stack = [tree]
    while len(stack) > 0:
        node = stack[-1]
        if node.label() == const:
            if node.height() > h:
                h = node.height()
        for child in node:
            if child.label() in wanted:
                stack.append(child)
        stack.remove(node)
    return h

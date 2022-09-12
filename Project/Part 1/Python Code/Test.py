import numpy as np

def advance_out(instance):
    instance.c += 1


class Test:
    c = 0
    def __init__(self, c):
        self.c = c

    @classmethod
    def advance(self):
        self.c += 1

x = np.array([Test(np.array([0, 2, 1, 5])), Test(np.array([5, 1, 8, 3]))])
map(Test.advance, x)
x.advance()
print(x[0].c)

from copy import copy


class AlphabetMonitor:
    alphabet = []
    value = []

    def __init__(self, signs):
        self.alphabet = []
        self.value = []
        for sign in signs:
            if not sign in self.alphabet:
                self.alphabet.append(sign)
                self.value.append(0)

    def __Monitoring__(self, learned):
        value = copy(self.value)
        for learn in learned:
            index = self.alphabet.index(learn)
            value[index] += 1
        return value

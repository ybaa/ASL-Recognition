from copy import copy


class AlphabetMonitor:
    alphabet = []
    value = []

    def __init__(self, signs):
        for sign in signs:
            if not sign in self.alphabet:
                self.alphabet.append(sign)
                self.value.append(0)

    def Monitoring(self, learned):
        value = copy(self.value)
        sum = len(learned)
        for learn in learned:
            index = self.alphabet.index(learn)
            value[index] += 1
        for val in value:
            val = val/sum
        return value
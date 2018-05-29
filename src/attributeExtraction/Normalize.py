from sklearn.preprocessing import normalize


def __Normalize__(descriptors):
    newDescriptors = normalize(descriptors, 'l2')
    return newDescriptors

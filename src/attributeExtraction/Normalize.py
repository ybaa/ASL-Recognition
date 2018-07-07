from sklearn.preprocessing import normalize as norm


def normalize(descriptors):
    newDescriptors = norm(descriptors, 'l2')
    return newDescriptors

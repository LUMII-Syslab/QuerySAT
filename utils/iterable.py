def elements_to_str(inputs: iter):
    return [str(x) for x in inputs]


def flatten(array: iter):
    while isinstance(array[0], list) or isinstance(array[0], tuple):
        array = [x for c in array for x in c]

    return array


def elements_to_int(array: iter):
    return [int(x) for x in array]

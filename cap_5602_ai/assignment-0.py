# CAP-5602
# Michael Cordero
# Assignment-0

def min_list(L: [float]) -> float:
    """
    Question 1 - minimum value of a list
    Write a function min_list(L) that takes in a non-empty list of real numbers L and returns its minimum value.
    You must use loops in your code.
    :param L: list
    :return min_value: smallest value
    """
    min_value = float('inf')
    for i in L:
        if i < min_value:
            min_value = i
    return min_value


def min_matrix(M: [float]) -> float:
    """
    Question 2 - minimum value of a matrix
    Write a function min_matrix(M) that takes in a non-empty matrix of real numbers M and returns its minimum value.
    You must use loops in your code.
    :param M:
    :return min_value: smallest value
    """
    min_value = float('inf')
    for row in range(len(M)):
        for col in range(len(M[row])):
            if (M[row][col]) < min_value:
                min_value = M[row][col]
    return min_value


def freq_to_prob(c: [int]) -> [int]:
    """
    Question 3 - convert vector of frequencies to probabilities
    Given a vector of frequencies c = (c1, c2, ..., cn), where all n >= 1 and all c_i are non-negative integers. We can
    convert this vector into a probability vector p = (p1, p2, ..., pn) using the formulation p_i = c_i/sum(c), for all
    i. Write a function freq_to_prob(c) that takes a frequency vector c as input and returns the corresponding
    probability vector p.
    :param c: input vector
    :return p: output probability vector
    """
    summation: int = sum(c)
    p: [int] = [i / summation for i in c]
    return p


if __name__ == '__main__':
    print('Hello CAP-5602-AI!')
    # Question 3
    # frequencies: [int] = [15, 10, 16, 2]
    # probabilities: [int] = freq_to_prob(frequencies)
    # print(f'Probabilities: ', probabilities)
    # Question 2
    # matrix: [int] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # print(f'Min Value of Matrix: ', min_matrix(matrix))
    # Question 1
    # arr: [int] = [0, 1, 2, 4, 5, 6]
    # print(f'Min Value of List: ', min_list(arr))

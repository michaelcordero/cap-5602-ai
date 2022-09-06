import math
import numpy as np
from scipy.special import softmax


def vec_to_prob(r: np.array) -> np.array:
    """
    Question 1: Convert vector of real numbers to probabilities (4% total grade)
Given a vector of real numbers  𝑟=(𝑟1,𝑟2,...,𝑟𝑛) . We can convert this vector into a probability vector  𝑝=(𝑝1,𝑝2,...,𝑝𝑛)  using the formulation:  𝑝𝑖=𝑒𝑟𝑖/(∑𝑛𝑖=1𝑒𝑟𝑖) , for all  𝑖 .
Write a Python function vec_to_prob(r) that takes the vector  𝑟  as input and returns the vector  𝑝 . Both  𝑟  and  𝑝  will be numpy arrays. You can assume  𝑟  is non-empty.
Sample inputs and outputs:
Input: np.array([4, 6]), output: [0.11920292, 0.88079708]
Input: np.array([3.4, 6.2, 7.1, 9.8]), output: [0.00151576, 0.02492606, 0.06130823, 0.91224995]
"""
    e = math.e
    summation = 0
    for i in r:
        summation += e ** i
    results = [e ** i / summation for i in r]
    p = np.array(results)
    return p


def mat_to_prob(R: np.array) -> np.array:
    """
    Question 2: Convert matrix of real numbers to probabilities (6% total grade)
Now we will extend Question 1 to matrices. Given a matrix of real numbers  𝑅=(𝑅1,𝑅2,...,𝑅𝑚) , where  𝑅𝑖=(𝑟1,𝑟2,...,𝑟𝑛)  is the i-th row of the matrix.
Question 2a (4% total grade):

Write a Python function mat_to_prob(R) that returns the matrix  𝑃=(𝑃1,𝑃2,...,𝑃𝑚)  where  𝑃𝑖  is the i-th row of matrix  𝑃 , and  𝑃𝑖  is the probability vector obtained from  𝑅𝑖  using the formulation in Question 1. In other words, convert each row of the input matrix into a probability vector.
Sample inputs and outputs:
Input: np.array([[4, 6], [3.5, 9.1]])
Output: [[0.11920292, 0.88079708], [0.00368424, 0.99631576]]
Input: np.array([[2, 3.1, 5], [10, 3.7, 12], [4, 5.5, 0]]))
Output: [[4.15115123e-02, 1.24707475e-01, 8.33781013e-01], [1.19176835e-01, 2.18844992e-04, 8.80604320e-01], [1.81818026e-01, 8.14851861e-01, 3.33011331e-03]]
    :param R:
    :return:
    """
    p = np.array([])
    for i in R:
        p = np.append(p, vec_to_prob(i))
    q = np.asmatrix(p)
    return q


def scipy_mat_to_prob(R: np.array) -> np.array:
    """
    Question 2b (2% total grade):
In fact, the function above is called the softmax function, and scipy has an implementation for it. First, read the API at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html.
Then write code to apply this version of the softmax function on the sample matrices above and print out the results.
    :param R:
    :return:
    """
    p = np.array([])
    for i in R:
        p = np.append(p, softmax(i))
    q = np.asmatrix(p)
    return q


def scale_vec(r: np.array) -> np.array:
    """
    Question 3: Vector scaling (5% total grade)
Given a vector of real numbers  𝑟=(𝑟1,𝑟2,...,𝑟𝑛) . We can standardize the vector using the formulation:  𝑣𝑖=𝑟𝑖−𝑚𝑠 , where  𝑚  is the mean of the vector  𝑟 , and  𝑠  is the standard deviation of  𝑟 . The vector  𝑣=(𝑣1,𝑣2,...,𝑣𝑛)  will be the scaled vector.
Write a Python function scale_vec(r) that takes the vector  𝑟  as input and returns the scaled vector  𝑣 .
Sample inputs and outputs:
Input: np.array([1, 3, 5]), output: [-1.22474487, 0., 1.22474487]
Input: np.array([3.3, 1.2, -2.7, -0.6]), output: [1.35457092, 0.40637128, -1.35457092, -0.40637128]
    :param r:
    :return:
    """
    q = np.array([])
    m = np.mean(r)
    s = np.std(r)
    for i in r:
        q = np.append(q, (i - m) / s)
    return q


if __name__ == '__main__':
    # t = np.array([4, 6])
    # u = vec_to_prob(t)
    v = np.array([[4, 6], [3.5, 9.1]])
    w = mat_to_prob(v)
    x = scipy_mat_to_prob(v)
    y = np.array([1, 3, 5])
    z = scale_vec(y)
    print(z)

import os
import sys
import argparse

import pandas as pd

import numpy as np
import numpy.typing as npt

from matplotlib import use
from matplotlib import pyplot as plt


use("TkAgg")


class Segment:

    def __init__(self,
        X: npt.ArrayLike, Y: npt.ArrayLike, order: int = 1, **kwargs,
    ) -> None:

        self.X = X
        self.Y = Y
        self.order = order

        self.__dict__.update(**kwargs)

    @property
    def Xe(self) -> npt.ArrayLike:
        # Add bias
        ones = (np.ones(self.X.shape),)
        # Expand feature vector
        cols = tuple(self.X ** (i+1) for i in range(self.order))\
                if not self.__dict__.get('func') else (self.func(self.X),)
        return np.column_stack(ones + cols)

    def mle(self, reg=None) -> npt.ArrayLike:
        # Maximum likelihood estimate
        return np.linalg.inv(
            self.Xe.T @ self.Xe + (reg*np.eye(self.Xe.shape[1]) if reg else 0.0)
        ) @ self.Xe.T @ self.Y

    def est(self, coeffs: npt.ArrayLike) -> float:
        # Return estimated 
        return sum((coef * self.X ** i for i, coef in enumerate(coeffs)))

    def fit(self, reg=0.0) -> float:
        # Fit function using maximum-likelihood/least squares
        coefs = self.mle(reg)
        Y_est = self.est(coefs)
        
        if self.__dict__.get('ax'): self.ax.plot(self.X, Y_est, 'r')

        return squared_err(self.Y, Y_est)


# Sum of the squared errors
squared_err = lambda y, y_est: np.sum((y - y_est) ** 2) 

# Try unknown function
fn = lambda x: np.sin(x)

# Fit and return total reconstruction error
fit_segment = lambda xs, ys, degree, ax=None: sum(
    (Segment(xs[i:i+20], ys[i:i+20], degree, ax=ax, func=None).fit()\
        for i in range(0, len(xs), 20))
)

def cross_validate(X_train, Y_train, X_test, Y_test, order, reg=None):
    train = Segment(X_train, Y_train, order)
    coefs = train.mle(reg)

    test = Segment(X_test, Y_test, order)
    Y_est = test.est(coefs)

    return ((Y_test - Y_est) ** 2).mean()

def optimizer(X, Y, min_overfit):
    # Minimise error
    order = best_order(X, Y, min_overfit)

    # Measure overfitting
    cve = k_fold(X, Y, order)

    return order

def k_fold(X, Y, order, k=10):
    assert 20 % k == 0, "Specify k-value that divides segments of 20 equally"   
    size = int(len(X) / k)
    cv_errors = []
    for i in range(0, len(X), size):
        start = i
        end = start+size
        # Test/train split
        X_test, Y_test = X[start:end], Y[start:end]
        X_train, Y_train = np.concatenate((X[:start],X[end:])),\
                            np.concatenate((Y[:start],Y[end:]))
        # Cross validate
        err = cross_validate(X_train, Y_train, X_test, Y_test, order)
        cv_errors.append(err)
    
    return np.array(cv_errors).mean()

def best_order(
    X: npt.ArrayLike, Y: npt.ArrayLike, min_overfit: bool = False
) -> int:
    # Find polynomial degree where error is minimised
    min_error = None
    max_degree = 1
    for degree in range(1, 10):
        error = k_fold(X, Y, degree) if min_overfit\
                    else fit_segment(X, Y, degree)
        if not min_error:
            min_error = error
        elif error < min_error:
            min_error = error
            max_degree = degree
        elif error > min_error:
            break
    return max_degree

def main(**vars) -> None:
    # Handle input arguments
    try:
        xs, ys = load_points_from_file(f"{vars['file']}")
    except FileNotFoundError:
        return print(f"Invalid filename: {vars['file']}")

    _, ax = plt.subplots()
    
    # Find polynomial order where error is minimised
    order = optimizer(xs, ys, vars['minimise_overfitting'])

    # Fit and plot with best polynomial order
    total_err = fit_segment(xs, ys, order, ax=ax)

    print(total_err)

    if vars.get('plot'): view_data_segments(xs, ys)


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('file')
    p.add_argument('--plot', dest='plot', action='store_true', default=False)
    p.add_argument('--minimise-overfitting', dest='minimise_overfitting', action='store_true', default=False)
    
    args = p.parse_args()
    main(**vars(args))
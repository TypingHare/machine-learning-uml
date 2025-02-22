'''
This is Homework 4 in COMP4220-Machine Learning 
University of Massachusetts Lowell
'''
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools


def create_synthetic_data(add_outliers=False, add_class=False):
    '''This function creates a synthetic 2D data set for classification.
    It can be used to add outliers and well as more classes.'''
    x0 = np.random.normal(size=50).reshape(-1,2)-1
    x1 = np.random.normal(size=50).reshape(-1,2)+1
    if add_outliers:
        x_1 = np.random.normal(size=10).reshape(-1,2)+np.array([5.0, 10.])
        return np.concatenate([x0, x1, x_1]),np.concatenate([np.zeros(25), np.ones(30)]).astype(int)
    if add_class:
        x_2 = np.random.normal(size=50).reshape(-1,2) + 3.
        return np.concatenate([x0,x1,x_2]), np.concatenta([np.zeros(25), np.ones(25), 2+np.zeros(25)]).astype(int)
    return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(int) 


class PolynomialFeature(object):
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        if x.ndim == 1:
            x = x[:, None]  #ensure size (:,1) instead of (:,)
        x_t = x.transpose() #return the data to its initial shape
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x,y: x*y, items))
        return np.asarray(features).transpose()


class Classifier(object):
    pass


class LogisticRegression(Classifier):
    def __init__(self):
        pass
    @staticmethod
    def _sigmoid(a):
        #--- your code here ---#
        # part (a)
        pass

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, max_iter: int=100):
        w = np.zeros(np.size(x_train, 1))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            #--- your code here ---#
            # part (b)
            y = []

            #--- your code here ---#
            # part (c)
            grad = []

            #--- your code here ---#
            # part (d)            
            hessian = []

            #--- your code here ---#
            # part (e)            
            w -= 0

        self.w = w

    def proba(self, x: np.ndarray):
        #--- your code here ---#
        # part (f)        
        return 0.0

    def classify(self, x: np.ndarray, threshold: float=0.5):
        #--- your code here ---#
        # part (g)        
        return 1


def test_LogisticRegression():
    #--- your code here ---#
    # part (h)    
    x_train, t_train = [], []

    #--- your code here ---#
    # part (i)    
    x1_test, x2_test = [], []
    x_test = []

    #--- your code here ---#
    # part (j)    
    # don't forget the bias term
    logistic_regression = LogisticRegression()


    #--- your code here ---#
    # part (k)    
    y_lr = []

    #--- your code here ---#
    # part (l)
    plt.scatter(0, 0)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Logistic Regression")
    plt.show()    


if __name__=='__main__':
    test_LogisticRegression()

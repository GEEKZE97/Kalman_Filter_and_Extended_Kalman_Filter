import numpy as np

class KalmanFilter:
    def __init__(self, x=None, A=None, B=None, C=None, Q=None, R=None, P=None):
        if x is not None and A is not None and B is not None and C is not None and Q is not None and R is not None and P is not None:
            self.init(x, A, B, C, Q, R, P)

    def init(self, x, A, B, C, Q, R, P):
        if 0 in x.shape or 0 in A.shape or 0 in B.shape or 0 in C.shape or 0 in Q.shape or 0 in R.shape or 0 in P.shape:
            return False
        self.x = x
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.P = P
        return True

    def setA(self, A):
        self.A = A

    def setB(self, B):
        self.B = B

    def setC(self, C):
        self.C = C

    def setQ(self, Q):
        self.Q = Q

    def setR(self, R):
        self.R = R

    def getX(self):
        return self.x

    def getP(self):
        return self.P

    def getXelement(self, i):
        return self.x[i]

    def predict(self, u=None, A=None, B=None, Q=None):
        if A is None: A = self.A
        if Q is None: Q = self.Q
        if B is None: B = self.B

        if u is not None:
            x_next = A @ self.x + B @ u
        else:
            x_next = A @ self.x

        if x_next.shape != self.x.shape or A.shape[1] != self.P.shape[0] or Q.shape != (Q.shape[1], Q.shape[0]):
            return False

        self.x = x_next
        self.P = A @ self.P @ A.T + Q
        return True

    def update(self, y, C=None, R=None):
        if C is None: C = self.C
        if R is None: R = self.R

        y_pred = C @ self.x

        if self.P.shape[1] != C.shape[0] or R.shape != (R.shape[0], R.shape[1]) or y.shape[0] != y_pred.shape[0]:
            return False

        PCT = self.P @ C.T
        K = PCT @ np.linalg.inv(R + C @ PCT)

        if np.isnan(K).any() or np.isinf(K).any():
            return False

        self.x = self.x + K @ (y - y_pred)
        self.P = self.P - K @ C @ self.P
        return True

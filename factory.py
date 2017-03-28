import GPy

class ModelFactory(object):

    def __init__(self, x, ):
        self.x = x

    def predictionKernel(self, k):
        """The kernel for predicting underlying function."""
        return k

    def buildKernel(self,):
        return GPy.kern.RBF(1)

    def batchTrain(self, y, size=3):
        best = None

        trained = []

        for s in range(size):
            m = self.train(y)

            trained.append(m)

            if best is None or best.log_likelihood() < m.log_likelihood():
                best = m

        return best, trained

    def train(self, y):
        k = self.buildKernel()

        m = GPy.models.GPRegression(self.x, y.T.reshape(y.shape[0]*y.shape[1] ,1), k)
        m.randomize()
        m.optimize()

        return m

class HierarchicalFactory(ModelFactory):

    def __init__(self, x, levels=2):
        ModelFactory.__init__(self,x)
        self.levels = levels

    def buildKernel(self):
        names = ['base'] + ['level%d'%i for i in range(self.levels)]
        kerns = [GPy.kern.RBF(1, name=n) for n in names]

        return GPy.kern.Hierarchical(kerns)

    def predictionKernel(self, k):
        return k.base

class OneLevelFactory(ModelFactory):

    def predictionKernel(self, k):
        return k.rbf

    def buildKernel(self,):
        return GPy.kern.Hierarchical([GPy.kern.RBF(1), GPy.kern.RBF(1)])

class TwoLevelFactory(OneLevelFactory):

    def buildKernel(self,):
        return GPy.kern.Hierarchical([GPy.kern.RBF(1), GPy.kern.RBF(1), GPy.kern.RBF(1)])

import GPy

class ModelFactory(object):

    def __init__(self, x, ):
        self.x = x

    def predictionKernel(self, k):
        """The kernel for predicting underlying function."""
        return k

    def parameters(self, gp):

        ret = {}
        ret['y_sigma']  = m.likelihood.sigma

        return ret

    def buildKernel(self,):
        return GPy.kern.RBF(1, name='kern.base')

    def batchTrain(self, y, size=3, callback=None, selectionCriteria=None):
        best = None

        if selectionCriteria is None:
            selectionCriteria = lambda x, y: y.log_likelihood() < x.log_likelihood()

        trained = []

        for s in range(size):
            m = self.train(y,callback=callback)

            trained.append(m)

            # if best is None or best.log_likelihood() < m.log_likelihood():
            if best is None or selectionCriteria(m, best):
                best = m

        return best, trained

    def reshape(self, y):
        return y.T.reshape(y.shape[0]*y.shape[1] ,1)

    def train(self, y, max_iter=100, callback=None):
        i = 0
        while i < max_iter:
            i += 1

            try:
                k = self.buildKernel()

                m = GPy.models.GPRegression(self.x, y.T.reshape(y.shape[0]*y.shape[1] ,1), k)

                if not callback is None:
                    m = callback(m)

                m.randomize()
                m.optimize()

                return m
            except:
                continue

        raise Exception('could not fit the model!')

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

    def parameters(self, gp):
        ret = ModelFactory.parameters(self, gp)

        for n in ['base'] + ['level%d'%i for i in range(self.levels)]:
            ret["%s_sigma"] = gp.kern.__dict__[n].sigma
            ret["%s_lengthscale"] = gp.kern.__dict__[n].lengthscale

        return ret

class OneLevelFactory(ModelFactory):

    def predictionKernel(self, k):
        return k.rbf

    def buildKernel(self,):
        return GPy.kern.Hierarchical([GPy.kern.RBF(1), GPy.kern.RBF(1)])

class TwoLevelFactory(OneLevelFactory):

    def buildKernel(self,):
        return GPy.kern.Hierarchical([GPy.kern.RBF(1), GPy.kern.RBF(1), GPy.kern.RBF(1)])

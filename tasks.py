import luigi
import os
import popmachine

DATADIR = '.data'
PLATES = [u'20150517 PQ 3',
          u'20150607 PQ 4',
          u'20150630 PQ 5',
          u'20150702 PQ 6',
          u'20150704 PQ 7',
          u'20150715 PQ 8',
          u'20150717 PQ 9',
          u'20161010_PQ_osmo',
          u'20161107_PQ_osmo_combo']


class Dataset(luigi.Task):

    name = luigi.Parameter()
    label = luigi.Parameter()
    search = luigi.DictParameter()
    scaleBy = luigi.ListParameter()

    def dataFile(self):
        return os.path.join(DATADIR, self.label, self.name, 'data.csv')

    def metaFile(self):
        return os.path.join(DATADIR, self.label, self.name, 'meta.csv')

    def run(self):
        machine = popmachine.Machine(
            'sqlite:///../popmachine_local/.popmachine.db')

        search = dict(self.search)
        for k in search.keys():
            if type(search[k]) == tuple:
                search[k] = list(search[k])

        ds = machine.search(plates=[self.name], **search)

        ds.log()
        ds.filter()
        ds.trim(5)
        ds.poly_scale(2, groupby=self.scaleBy)

        ds.data = ds.data.iloc[::3, :]

        luigi.LocalTarget(self.dataFile()).makedirs()
        ds.data.to_csv(self.dataFile())
        ds.meta.to_csv(self.metaFile())

    def output(self):
        return [luigi.LocalTarget(self.dataFile()),
                luigi.LocalTarget(self.metaFile())]


class StandardData(luigi.Task):

    def requires(self):
        return [
            Dataset(name=p,
                    label='standard',
                    search={
                        'Strain': 'ura3', 'mM PQ': [0.0], 'M NaCl':[4.2, None]},
                    scaleBy=['plate'])
            for p in PLATES]


class LowData(luigi.Task):

    def requires(self):
        return [
            Dataset(name=p,
                    label='low-oxidative',
                    search={
                        'Strain': 'ura3', 'mM PQ': [0.0, 0.083], 'M NaCl':[4.2, None]},
                    scaleBy=['plate', 'mM PQ'])
            for p in PLATES]


class HiData(luigi.Task):

    def requires(self):
        return [
            Dataset(name=p,
                    label='hi-oxidative',
                    search={
                        'Strain': 'ura3', 'mM PQ': [0.0, 0.333], 'M NaCl':[4.2, None]},
                    scaleBy=['plate', 'mM PQ'])
            for p in PLATES]


class AllData(luigi.Task):

    def requires(self):
        return [StandardData(), LowData(), HiData()]


class Posterior(luigi.Task):

    label = luigi.Parameter()
    plates = luigi.ListParameter()

    condition = luigi.Parameter(default=None)
    control = luigi.Parameter(default=None)
    batch = luigi.Parameter(default=None)

    def requires(self):
        return AllData()

    def run(self):
        data = pd.DataFrame()
        meta = pd.DataFrame()

        for p in self.plates:
            d = pd.read_csv(os.path.join(DATADIR, self.label, p, 'data.csv'))
            m = pd.read_csv(os.path.join(DATADIR, self.label, p, 'meta.csv'))

            data = pd.concat((data, d), 1)
            meta = pd.concat((meta, m), 0)

        ds = popmachine.Dataset(data=data, meta=meta)

        effects = []
        if self.condition:
            effects.append(self.condition)
        if self.batch:
            effects.append(self.batch)

        xraw, y, design, labels = ds.build(effects, scale=True)
        ymean, ystd = y.mean(), y.std()
        y = (y - y.mean()) / y.std()
        x = (xraw - xraw.mean()) / xraw.std()

        k = 1
        b = 1
        if self.condition:
            k += 1
            b = 2
        if self.batch:
            k *= (1 + ds.meta[self.batch].unique().shape[0])

        dm = np.zeros((y.shape[1], k))
        dm[:, 0] = 1

        if self.condition:
            dm[:, 1] = 1 - 2 * \
                (design[self.condition] == labels[0].index(self.control))

        if self.batch:
            for i in range(design.plate.unique().shape[0]):
                dm[:, b + i * b:b * 2 + i * b] = dm[:, :b] * \
                    (design[self.batch] == i).values[:, None]

        p = dm.shape[1]
        priors = [1, 2] + [3, 4] * ds.meta.plate.unique().shape[0]
        train_data = {
            'N': x.shape[0],
            'P': y.shape[1],
            'K': dm.shape[1],
            'L': 4,
            'prior': priors,
            'length_scale': [1, .5, .3],
            'alpha': [1, .4, .3],
            'sigma': .2,
            'design': dm,
            'x': x[:, 0],
            'y': y.T
        }

        if self.condition:
        train_data['alpha_prior'] = [[1, 1], [1, 1], [.1, 1], [.1, 1]]

        train_data['length_scale_prior'] = [[1.5, 2]] * k
        train_data['sigma_prior'] = [.1, 1]

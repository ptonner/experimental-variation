from popmachine import Machine, DataSet
import pandas as pd
import os

if '.popmachine.db' in os.listdir('.'):
    print 'Database already exists, delete and re-create?'
    ans = raw_input()
    if ans.lower() == 'y':
        os.remove('.popmachine.db')

machine = Machine()

#################
# od-cfu
#################

od = pd.read_excel("data/cfu-od-raw.xlsx")
melt = pd.melt(od,id_vars=['strain'],var_name='time', value_name='OD600')
cfu = pd.read_excel("data/cfu-od-raw.xlsx",sheetname=1)
merge = pd.merge(melt, cfu, 'outer', ['strain', 'OD600'])

d1 = merge.pivot('time', 'strain', 'OD600')
d2 = merge.pivot('time', 'strain', 'CFUs/ml')

meta = pd.DataFrame({'strain':d1.columns.tolist() + d2.columns.tolist()})

m1 = pd.DataFrame(d1.columns)
m2 = pd.DataFrame(d2.columns)

m1['measurment'] = 'od'
m2['measurment'] = 'cfu'

d1.columns = range(d1.shape[1])
d2.columns = range(d1.shape[1])

data = pd.merge(d1,d2,left_index=True, right_index=True)
meta = pd.concat((m1,m2))

machine.createPlate('cfu-od-1', data=data, experimentalDesign=meta)

#################
# pq-osmo
#################

ds = DataSet.fromDirectory("data/pq-osmo-control/")
machine.createPlate('pq-osmo-control', data=ds.data, experimentalDesign=ds.meta)

ds = DataSet.fromDirectory("data/pq-osmo-combo/")
machine.createPlate('pq-osmo-combo', data=ds.data, experimentalDesign=ds.meta)

#################
# ura3-replicate
#################

ds = DataSet.fromDirectory("data/normalized/ura3-pq-replicate/")
machine.createPlate('ura3-replicate', data=ds.data, experimentalDesign=ds.meta)

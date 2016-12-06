

if __name__ == "__main__":

    import argparse, ConfigParser, os

    parser = argparse.ArgumentParser(description='Generate single-level datasets')

    parser.add_argument('p', help='number of replicates', type=int)
    parser.add_argument('theta1', help='ratio of function to noise variance', type=float)
    parser.add_argument('theta2', help='ratio of replicate noise to iid noise variance', type=float)

    args = parser.parse_args()

    config = ConfigParser.ConfigParser()
    config.read('single-level/base.cfg')
    config.set("main",'nrep',str(args.p))
    config.set("k1",'sigma',str(args.theta1))
    config.set("k2",'sigma',str((1-args.theta1)*args.theta2))
    config.set("yKernel",'sigma',str((1-args.theta1)*(1-args.theta2)))

    label = '%d-%.1lf-%.1lf' % (args.p, args.theta1, args.theta2)

    if not label in os.listdir('single-level'):
        os.mkdir(os.path.join('single-level',label))

    config.write(open(os.path.join('single-level',label,'config.cfg'),'w'))

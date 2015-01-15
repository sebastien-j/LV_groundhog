import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sgmfile", type=str)
parser.add_argument("txtfile", type=str)
args = parser.parse_args()

with open(args.sgmfile, 'r') as f:
    with open(args.txtfile, 'w') as g:
        for line in f:
            if line.startswith('<seg id='):
                pos = line.find('>')
                g.write(line[pos+1:-7]+'\n')

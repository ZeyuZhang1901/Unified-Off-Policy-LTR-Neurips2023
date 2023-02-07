# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--list', type=str, nargs='+')
# args = parser.parse_args()
# print(args.list)

import torch
def func(a):
    print(a)
    a = a+1
    print(a)
    a[:, 2] = 0.0
    print(a)
    return

a = torch.rand(2,5)
func(a)
print(a)

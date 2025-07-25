'''
Description: 
Author:  
Date: 2024-07-08 17:39:49
LastEditTime: 2024-08-01 22:51:43
LastEditors:  
'''

import datetime
import os
import threading
import sys

main_node_path = '/home/huangxiaoyang/AAAI2026_our/main.py' # todo
python_path = '/home/huangxiaoyang/miniconda3/envs/dl1/bin/python'

def execCmd(cmd):
    print("Python interpreter path:", sys.executable)

    try:
        os.system(cmd)
    except:
        print('%s\t Runtime Error' % (cmd))

if __name__ == '__main__':
    if_parallel = False # False

    # 
    cmds = [
        f"python {main_node_path} --lr 4e-7 --seed 43 --model our --lam 2 --task tra",
        f"python {main_node_path} --lr 4e-7 --seed 43 --model our --lam 2 --task npr",
        f"python {main_node_path} --lr 4e-7 --seed 43 --model our --lam 2 --task roe",

    ]

    if if_parallel:
        threads = []
        for cmd in cmds:
            th = threading.Thread(target=execCmd, args=(cmd,))
            th.start()
            threads.append(th)

        for th in threads:
            th.join()
    else:
        for cmd in cmds:
            try:
                os.system(cmd)
            except:
                print('%s\t Runtime Error' % (cmd))
                exit()




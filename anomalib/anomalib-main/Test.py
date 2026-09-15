# 定义Test.py文件按如下代码进行测试

import argparse
if __name__ == "__main__":
    print ('start')
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputArgs', nargs='+')
    args = parser.parse_args()
    print(args.InputArgs)


# 在cmd窗口运行如下代码:
# //解释器                     脚本文件所在目录                   接收参数变量         参数
# python                      D:/Pycharm_Code/.../Test.py     --InputArgs          111 2355 222
# 程序输出结果
# ['111', '2355', '222']

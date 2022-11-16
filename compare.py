import ast
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import merge_txt

# ratios = ['23', '55', '78', '1010', '1213'] # default: 23
ratios = ['23']
# steps = range(100, 1000, 100) # default: 100
steps = [100]
filepath = '01_pro' # default: 0


for step in steps:
    print("Current step:", step)
    for ratio in ratios:
        merge_txt(ratio, str(step), filepath)
        print("- Current ratio:", ratio)
        path = f'results/{filepath}/{str(step)}/content'
        readfiles = open(path+'/compare'+ratio, 'r')
        readlines = readfiles.readlines()
        # i = 0
        ONMF = []
        TDF = []
        SNMF = []
        PD = []
        for i in range(len(readlines)):
            line = readlines[i]
            if 'ONMF' in line:
                for j in range(1,5):
                    i = i+1
                    line=readlines[i]
                    line=line.strip('\n')
                    line_list = line.split(' ')
                    del line_list[0]
                    # del line_list[-1]
                    ONMF.append([float(x) for x in line_list])
                i = i + 1
            elif 'PD' in line:
                for j in range(1, 5):
                    i = i+1
                    line = readlines[i]

                    line = line.strip('\n')
                    line_list = line.split(' ')
                    del line_list[0]
                    # del line_list[-1]
                    PD.append([float(x) for x in line_list])
                i = i + 1

            elif 'TDF' in line:
                for j in range(1, 5):
                    i = i+1
                    line = readlines[i]
                    line = line.strip('\n')
                    line_list = line.split(' ')
                    del line_list[0]
                    # del line_list[-1]
                    TDF.append([float(x) for x in line_list])
                i = i + 1
            elif 'SNMF' in line:
                for j in range(1, 5):
                    i = i + 1
                    line = readlines[i]
                    line = line.strip('\n')
                    line_list = line.split(' ')
                    del line_list[0]
                    # del line_list[-1]
                    SNMF.append([float(x) for x in line_list])
                i = i + 1
            else:
                pass
            # print(line)



            # i=i+1
            # line=readlines[i]
            # if 'JPP' in line:
            #     for j in range(1,5):
            #         i=i+1
            #         line=readlines[i]
            #         line=line.strip('\n')
            #         line_list=ast.literal_eval(line)
            #         del line_list[-1]
            #         JPP.append([float(x) for x in line_list])





        plt.rcParams.update({'font.size': 14})


        ONMF = [x[:47] for x in ONMF]
        TDF = [x[:47] for x in TDF]
        SNMF = [x[:47] for x in SNMF]
        PD = [x[:47] for x in PD]
        print(TDF)
        x = range(len(TDF[0]))

        methods = ['joint ONMF', 'Pseudo-Deflation', 'ONMF', 'SNMF']
        TIME = COMMON = DIFFER = True
        RE = True
        if TIME == True:
            fig = plt.figure()
            fig.set_size_inches(7.5, 5)
            plt.ylim(0, 10)
            plt.plot(x, np.log(TDF[0]), marker='+', linewidth=1.5, color='tab:red')
            plt.plot(x, np.log(PD[0]), marker='x', linewidth=1.5, color='tab:purple', markersize=4, linestyle='dashed')
            plt.plot(x, np.log(ONMF[0]), marker='v', linewidth=1.5, color='tab:blue', markersize=4, markerfacecolor='tab:blue')
            # plt.plot(x, np.log(JPP[0][:len(x)]), marker='o', color='green', markerfacecolor='green', markersize=4,linewidth=1.5, linestyle='dashed')
            #plt.plot(x, np.log(OLDA[0][:len(x)]), marker='d', markerfacecolor='orange', markersize=4, color='orange', linewidth=1.5)
            plt.plot(x, np.log(SNMF[0]), marker='x', linewidth=1.5,color='tab:orange',  markersize=4, markerfacecolor='tab:orange')
            #plt.title()
            plt.xlabel('Time Stamp')
            plt.ylabel('Training Time (s) (log)')
            plt.legend(methods, ncol=len(methods), loc='upper center', prop={'size': 10})
            # plt.show()
            plt.savefig(path+'Time_log'+ratio+'.png')

        if COMMON == True:
            # fig = plt.figure()
            # fig.set_size_inches(7.5, 5)
            # plt.plot(x, TDF[1], marker='+',  linewidth=1.5, linestyle='dashed', color='tab:red')
            # plt.plot(x, PD[1], marker='o', linewidth=1.5, linestyle='dashed', color='tab:purple', markerfacecolor='tab:purple', markersize=4)
            # plt.plot(x, ONMF[1], marker='v', linewidth=1.5, linestyle='dashed', color='tab:blue', markerfacecolor='tab:blue', markersize=4)
            # # plt.plot(x, OLDA[1], marker='d', markerfacecolor='orange', markersize=4, color='orange', linewidth=1.5,linestyle='dashed')
            # plt.plot(x, SNMF[1], marker='x', linewidth=1.5, linestyle='dashed', color='tab:orange', markerfacecolor='tab:orange', markersize=4)
            # plt.plot(x, [np.mean(TDF[1])]*len(TDF[0]), linewidth=2, color='tab:red')
            # plt.plot(x, [np.mean(PD[1])]*len(TDF[0]), linewidth=2, color='tab:purple')
            # plt.plot(x, [np.mean(ONMF[1])]*len(TDF[0]), linewidth=2, color='tab:blue')
            # # plt.plot(x,[np.mean(JPP[1])]*len(RTT[0]),linewidth=2,color='green')
            # # plt.plot(x,[np.mean(OLDA[1])]*len(RTT[0]),linewidth=2,color='orange')
            # plt.plot(x, [np.mean(SNMF[1])]*len(TDF[0]), linewidth=2, color='tab:orange')
            # plt.xlabel('Time')
            # plt.ylabel('Commonness Score')
            # plt.legend(methods, ncol=len(methods))
            # plt.savefig(path+'common_line'+ratio+'.png')

            fig = plt.figure()
            fig.set_size_inches(7.5, 5)
            plt.plot(x, TDF[1], marker='+', linewidth=1.5, color='tab:red')
            plt.plot(x, PD[1], marker='x',  linewidth=1.5, color='tab:purple', markersize=4, linestyle='dashed')
            plt.plot(x, ONMF[1], marker='o', linewidth=1.5, markersize=4, color='tab:blue', markerfacecolor='tab:blue')
            # plt.plot(x[:-1], JPP[2], marker='', color='olive', linewidth=1, linestyle='dashed')
            # plt.plot(x[:-1], OLDA[2], marker='d', markerfacecolor='orange', markersize=4, color='orange', linewidth=1)
            plt.plot(x, SNMF[1], marker='x', linewidth=1.5, markersize=4, color='tab:orange', markerfacecolor='tab:orange')
            #plt.title()
            plt.xlabel('Time')
            plt.ylabel('Commonness Score')
            plt.legend(methods, ncol=len(methods), prop={'size': 10})
            plt.savefig(path+'common'+ratio+'.png')

        if DIFFER == True:
            fig = plt.figure()
            fig.set_size_inches(7.5, 5)
            plt.plot(x, TDF[2], marker='+', color='tab:red', linewidth=1.5)
            plt.plot(x, PD[2], marker='x', linewidth=1.5, color='tab:purple', markersize=4, linestyle='dashed')
            plt.plot(x, ONMF[2], marker='o', markerfacecolor='tab:blue', markersize=4, color='tab:blue', linewidth=1.5)
            # plt.plot(x[:-1], JPP[3], marker='', color='olive', linewidth=1, linestyle='dashed')
            # plt.plot(x[:-1], OLDA[3], marker='d', markerfacecolor='orange', markersize=4, color='orange', linewidth=1)
            plt.plot(x, SNMF[2], marker='x', markerfacecolor='tab:orange', markersize=4, color='tab:orange', linewidth=1.5)
            #plt.title()
            plt.xlabel('Time')
            plt.ylabel('Difference Score')
            plt.legend(methods, ncol=len(methods), prop={'size': 10})
            plt.savefig(path+'differ'+ratio+'.png')

        if RE == True:
            fig = plt.figure()
            fig.set_size_inches(7.5, 5)
            plt.plot(x, np.log(TDF[3]), marker='+', color='tab:red', linewidth=1.5)
            plt.plot(x, np.log(PD[3]), marker='x', linewidth=1.5, color='tab:purple', markersize=4, linestyle='dashed')
            plt.plot(x, np.log(ONMF[3]), marker='o', markerfacecolor='tab:blue', markersize=4, color='tab:blue', linewidth=1.5)
            # plt.plot(x, JPP[4], marker='', color='olive', linewidth=1, linestyle='dashed')
            # plt.plot(x, OLDA[4], marker='d', markerfacecolor='orange', markersize=4, color='orange', linewidth=1)
            plt.plot(x, np.log(SNMF[3]), marker='x', markerfacecolor='tab:orange', markersize=4, color='tab:orange', linewidth=1.5)
            #plt.title()
            plt.xlabel('Time')
            plt.ylabel('Reconstruction Error (log)')
            plt.legend(methods, ncol=len(methods), prop={'size': 10})
            plt.savefig(path+'error_log'+ratio+'.png')





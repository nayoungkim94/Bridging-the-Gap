import matplotlib.pyplot as plt
import numpy as np
file=open('results/harvey/nc/Harvey.txt','r')
lines=file.readlines()
beta_list=[]
common_list_final=[]
differ_list_final=[]
for line in lines:
    line.strip('\n')
    if 'nc' in line:
        alist=line.split(' ')
        beta_list.append(float(alist[1]))
    elif 'common' in line:
        clist=line.split(' ')
        del clist[0]
        common_list_final.append([float(x) for x in clist])
    elif 'differ' in line:
        dlist=line.split(' ')
        del dlist[0]
        differ_list_final.append([float(x) for x in dlist])
num=0
comm_ls=[]
diff_ls=[]
n=len(common_list_final)
for column in range(n):
    num += 1
    common_score=common_list_final[column]
    differ_score=differ_list_final[column]
    comm_ls.append(np.mean(common_score))
    diff_ls.append(np.mean(differ_score))

print comm_ls
print diff_ls
plt.rcParams.update({'font.size': 14})
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel(r'$k_c$')
ax1.set_ylabel('CScore', color=color)
#ax1.set_ylim(0,0.0005)
ax1.set(xticks=range(n),xticklabels=['1','3','5','7','9'])
ax1.plot(range(n), comm_ls, 'rD-')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('DScore', color=color)  # we already handled the x-label with ax1
ax2.plot(range(n), diff_ls, 'b+--')
ax2.set_ylim(10,15)
ax2.tick_params(axis='y', labelcolor=color)
fig.legend(('CScore','DScore'),loc='upper center',ncol=2)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
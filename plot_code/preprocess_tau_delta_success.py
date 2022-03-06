import numpy as np 
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Serif"   # Font
plt.rcParams["font.size"] = 10                 # Size of word

for i in range(14):
    data_nopreprocess = np.loadtxt('tau_delta_nopreprocess/success'+str(i)+'.csv',skiprows=1,delimiter=",")
    data_nopreprocess_temp = data_nopreprocess[:,1]
    data_nopreprocess_temp = np.array([data_nopreprocess_temp])
    data_proposed_method = np.loadtxt('tau_delta_preprocess/success'+str(i)+'.csv',skiprows=1,delimiter=",")
    data_proposed_method_temp = data_proposed_method[:,1]
    data_proposed_method_temp = np.array([data_proposed_method_temp])

    # Martix is [num of learning curve, steps]
    if i > 0:
        data_nopreprocess_matrix = np.concatenate([data_nopreprocess_matrix, data_nopreprocess_temp])
        data_proposed_method_matrix = np.concatenate([data_proposed_method_matrix, data_proposed_method_temp])
    else:
        data_nopreprocess_matrix = data_nopreprocess_temp
        data_proposed_method_matrix = data_proposed_method_temp

data_len = len(data_proposed_method_temp[0])
print(data_len)
steps = []
for i in range(data_len):
    steps.append((i+1)*10000)

train_nopreprocess_scores_mean = np.mean(data_nopreprocess_matrix, axis=0)
train_nopreprocess_scores_std = np.std(data_nopreprocess_matrix, axis=0)
train_proposed_method_scores_mean = np.mean(data_proposed_method_matrix, axis=0)
train_proposed_method_scores_std = np.std(data_proposed_method_matrix, axis=0)

plt.figure()
plt.xlabel("Learning Steps")
plt.ylabel("Success Rate")


# Plot Traing score and Test score 
plt.plot(steps, train_nopreprocess_scores_mean,color="r", label="Without Preprocessing")
plt.plot(steps, train_proposed_method_scores_mean,color="b", label="With Preprocessing")

# Plot standard distribution
plt.fill_between(steps, train_nopreprocess_scores_mean - train_nopreprocess_scores_std, train_nopreprocess_scores_mean + train_nopreprocess_scores_std, color="r", alpha=0.15)
plt.fill_between(steps, train_proposed_method_scores_mean - train_proposed_method_scores_std, train_proposed_method_scores_mean + train_proposed_method_scores_std, color="b", alpha=0.15)

plt.xlim(0, 600000)
plt.ylim(0., 1.01)
plt.legend(loc="best")
plt.grid()
#plt.show()
plt.savefig("success_rates_preprocess.png")

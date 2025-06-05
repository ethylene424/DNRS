import scipy.stats as sta
from result import *
from scipy.stats import chi2_contingency


dataset_list = ["Ionosphere", "Isolet", "TWbankrupt", "Segmentation", "Waveform", "Yeast", "Madelon", "Swarm", "Communities"]


data_result = []
for m in dataset_list:
    temp = dim_red[m][1:]
    # for i in range(8):
    #     temp.append(float(classifier[m][i][2][:5]))
        # for j in range(3):
    data_result.append(temp)


stats, pvalue = sta.friedmanchisquare(data_result[0], data_result[1], data_result[2],
                                      data_result[3], data_result[4], data_result[5],
                                      data_result[6], data_result[7], data_result[8],)

kt = chi2_contingency(data_result)

print(stats, pvalue, kt)
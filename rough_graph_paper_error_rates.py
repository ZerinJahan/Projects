# ***********   WITH OUT  SCALING *********
 # *** ONLY FOR GRAPH ****
 
from numpy.random import seed
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('xx-small')


font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 24}
plt.rc('font', **font)



url="E:/ML/WEATHER FORECAST WITH JP_ZERIN/DOWNLOADS & ROUGH WRITING/next latex file/updated_2/New Microsoft Excel Worksheet.csv"
dataset = pd.read_csv(url)

df = dataset.copy()

df.plot(x="Parameter", y=['MSE', 'RMSE', 'R2', 'MAE', 'MAPE'], kind = "barh")
plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
plt.savefig('E:/ML/WEATHER FORECAST WITH JP_ZERIN/DOWNLOADS & ROUGH WRITING/next latex file/updated_2/figures/barh.png', dpi = 300)
plt.show()


# # training_set = train_set.iloc[:, 18:19].values
# # testing_set = test_set.iloc[:, 18:19].values

# y_test_descaled_windspeedKmph = dataset.iloc[0:720, 0:1].values
# y_predicted_descaled_windspeedKmph = dataset.iloc[0:720, 1:2].values
# y_predicted_descaled_windspeedKmph = y_predicted_descaled_windspeedKmph

# y_test_descaled_humidity = dataset.iloc[0:720, 2:3].values
# y_predicted_descaled_humidity= dataset.iloc[0:720, 3:4].values
# y_predicted_descaled_humidity= dataset.iloc[0:720, 3:4].values

# y_test_descaled_tempC= dataset.iloc[0:720, 4:5].values
# y_predicted_descaled_tempC= dataset.iloc[0:720, 5:6].values
# y_predicted_descaled_tempC= dataset.iloc[0:720, 5:6].values


# y_test_descaled_pressure= dataset.iloc[0:720, 6:7].values
# y_predicted_descaled_pressure= dataset.iloc[0:720, 7:8].values
# y_predicted_descaled_pressure= dataset.iloc[0:720, 7:8].values


# y_test_descaled_DewPointC= dataset.iloc[0:720, 8:9].values
# y_predicted_descaled_DewPointC= dataset.iloc[0:720, 9:10].values
# y_predicted_descaled_DewPointC= dataset.iloc[0:720, 9:10].values


# #
# # Show results
# #
# plt.figure(figsize=(14,14))


# plt.subplot(5, 2, 1)
# plt.plot(y_test_descaled_DewPointC[0:720], color = 'black', linewidth=1)
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.ylabel("Dew Point")
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")

# plt.subplot(5, 2, 2)
# plt.plot(y_predicted_descaled_DewPointC[0:720], color = 'red')
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")


# plt.subplot(5, 2, 3)
# plt.plot(y_test_descaled_humidity[0:720], color = 'black', linewidth=1)
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.ylabel("Humidity")
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")

# plt.subplot(5, 2, 4)
# plt.plot(y_predicted_descaled_humidity[0:720], color = 'red')
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")


# plt.subplot(5, 2, 5)
# plt.plot(y_test_descaled_tempC[0:720], color = 'black', linewidth=1)
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.ylabel("Temperature")
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")

# plt.subplot(5, 2, 6)
# plt.plot(y_predicted_descaled_tempC[0:720], color = 'red')
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")


# plt.subplot(5, 2, 7)
# plt.plot(y_test_descaled_windspeedKmph[0:720], color = 'black', linewidth=1)
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.ylabel("Wind Speed")
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")

# plt.subplot(5, 2, 8)
# plt.plot(y_predicted_descaled_windspeedKmph[0:720], color = 'red')
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")



# plt.subplot(5, 2, 9)
# plt.plot(y_test_descaled_pressure[0:720], color = 'black', linewidth=1)
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.ylabel("Pressure")
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")

# plt.subplot(5, 2, 10)
# plt.plot(y_predicted_descaled_pressure[0:720], color = 'red')
# # plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
# plt.xlabel("Hours")
# # plt.title("Predicted data (first 30 days)")



# # plt.subplot(3, 3, 9)
# # plt.scatter(y_predicted_descaled, y_test_descaled, s=2, color='black')
# # plt.ylabel("Y true")
# # plt.xlabel("Y predicted")
# # plt.title("Scatter plot")


# plt.subplots_adjust(hspace = 1, wspace=.25)
# plt.savefig('E:/ML/RESULTS/FIGURES/ROUGH.png', dpi = 300)
# plt.show()


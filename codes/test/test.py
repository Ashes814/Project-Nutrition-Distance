import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# df = pd.read_csv('p.csv')

# df['buildingP'].values

from CNTD import CalNutritionTransformDistance as cndt

testObj = cndt()
df = testObj.readData(r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\t.csv')
twoDArray, supplyIndex, demandIndex = testObj.calcGridIndex(df['gridResult'].values)
print('————————————————————输送前————————————————————————')
testObj.showGridNumbers(twoDArray)
testObj.showMatrix(twoDArray, name= r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\输送前.jpg')
s_to_d_distance = testObj.calcIndexDistance(supplyIndex, demandIndex)
sortedDistanceDict = testObj.sortDisDict(s_to_d_distance)
indexWithValueSupply = testObj.calcIndexWithValueSupply(twoDArray, sortedDistanceDict)
distance_array, finalGrids = testObj.iterateCalculation(twoDArray, indexWithValueSupply, sortedDistanceDict)
print('————————————————————输送后————————————————————————')
testObj.showGridNumbers(finalGrids)
print('最远输送距离为网格{0}至{1}，距离为{2}单位'.format(distance_array[0][0], distance_array[0][1], distance_array[0][2]))
testObj.showMatrix(finalGrids, name= r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\输送后.jpg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class calNutritionTransformDistance(object):

    def readData(self, file):
        """read the supply and demand data

        Args:
            file (string): file path

        Returns:
            DataFrame: pandas DF for further calculation
        """
        df = pd.read_csv(file)
        df = df.fillna(0)
        df['pneed'] = df['grid_code'] * 0.9 * 100
        df['gridResult'] = df['buildingP'] - df['pneed']
        return df

    def calEucDistance(self, value1, value2):
        """The Euculidian Distance between two grid

        Args:
            value1 (tuple): coordinates of grid1
            value2 (tuple): coordinates of grid2

        Returns:
            int: the distance between grid1 and grid2
        """
        return ((value1[0] - value2[0])**2 + (value1[1] - value2[1])**2)**0.5


    def showMatrix(self, value, name = r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\test.jpg', shape=(100, 100)):
        """Visualize the Nutrition Supply and Demand situation

        Args:
            value (Array): supply - demand of all grids
            shape (tuple, optional): the grid resolution of research area. Defaults to (100, 100).
        """
        v = value.reshape(shape[0], shape[1])
        fixValues = np.zeros(shape)
        for i in range(shape[0]):
            fixValues[shape[0] - i - 1] = v[i]
        plt.imshow(fixValues, cmap='magma')
        plt.savefig(name, dpi=300)


    def showGridNumbers(self, value):
        """Show the number of supply and demand grids respectively

        Args:
            value (Array): supply - demand of all grids
        """
        demandGridNumbers = (value < 0).sum()
        supplyGridNumbers = (value > 0).sum()
        supplyTotal = value[value > 0].sum()
        demandTotal = value[value < 0].sum()

        print("供给网格数量为{0}, 总量为{1}KG".format(supplyGridNumbers, supplyTotal))
        print("需求网格数量为{0}, 总量为{1}KG".format(demandGridNumbers, demandTotal))
        print("总网格数量为{}".format(value.shape[0] * value.shape[1]))

    def calcGridIndex(self, value, shape=(100, 100)):
        """计算供给与需求在网格矩阵中的索引

        Args:
            value (Array): supply - demand of all grids
            shape (tuple, optional): 网格分辨率. Defaults to (100, 100).

        Returns:
            list: 返回三个列表，第一个是二维矩阵，包含所有的供给量信息，第二和第三个分别是需求的索引
        """
        twoDArray = value.reshape(shape[0], shape[1])
        supplyIndex = []
        demandIndex = []
        for i in range(shape[0]):
            for j in range(shape[0]):
                if twoDArray[i][j] > 0:
                    supplyIndex.append((i,j))
                elif twoDArray[i][j] < 0:
                    demandIndex.append((i,j))
        return twoDArray, supplyIndex, demandIndex


    def calcIndexDistance(self, supplyIndex, demandIndex):
        """计算每个供给网格到所有需求网格的距离

        Args:
            supplyIndex (list): 所有供给网格的索引（即坐标）
            demandIndex (list): 所有需求网格的索引（即坐标）

        Returns:
            dict: 字典的键为索引号，值为
        """
        s_to_d_distance = {}
        for s in supplyIndex:
            s_to_d_distance[s] = []
            for d in demandIndex:
                s_to_d_distance[s].append([d, self.calEucDistance(s, d)])
        return s_to_d_distance

    def sortDisDict(self, s_to_d_distance):
        sortedDistanceDict = {}
        for i in s_to_d_distance:
            sortedDistanceDict[i] = sorted(s_to_d_distance[i], key=lambda x: x[1])
        return sortedDistanceDict


    def calcIndexWithValueSupply(self, twoDArray, sortedDistanceDict):
        indexWithValueSupply = []
        for index in sortedDistanceDict.keys():
            supply = twoDArray[index]
            indexWithValueSupply.append([index, supply])
        indexWithValueSupply.sort(key=lambda x : x[1], reverse=True)
        return indexWithValueSupply
        

    def iterateCalculation(self, twoDArray, indexWithValueSupply, sortedDistanceDict):
        distance_array = []
        testTwoDArray = twoDArray.copy()
        for supply in indexWithValueSupply:
            tempSupply = testTwoDArray[supply[0]]
            for demand in sortedDistanceDict[supply[0]]:
                tempDemand = testTwoDArray[demand[0]]

                if tempSupply <= tempDemand:
                    tempSupply = 0
                    tempDemand += tempSupply
                    testTwoDArray[supply[0]] = tempSupply
                    testTwoDArray[demand[0]] = tempDemand
                    # distance_array.append([supply[0], demand[0], calEucDistance(supply[0], demand[0])])
                    
                elif tempSupply > tempDemand:
                    tempSupply += tempDemand
                    tempDemand = 0
                    testTwoDArray[supply[0]] = tempSupply
                    testTwoDArray[demand[0]] = tempDemand
                else:
                    pass
                if tempSupply == 0:
                    distance_array.append([supply[0], demand[0], self.calEucDistance(supply[0], demand[0])])
                    break
        distance_array.sort(key=lambda x: x[2], reverse=True)
        return distance_array, testTwoDArray

testObj = calNutritionTransformDistance()
        

if __name__ == "__main__":
    df = testObj.readData(r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\t.csv')
    twoDArray, supplyIndex, demandIndex = testObj.calcGridIndex(df['gridResult'].values)
    print('————————————————————输送前————————————————————————')
    testObj.showGridNumbers(twoDArray)
    testObj.showMatrix(twoDArray)
    s_to_d_distance = testObj.calcIndexDistance(supplyIndex, demandIndex)
    sortedDistanceDict = testObj.sortDisDict(s_to_d_distance)
    indexWithValueSupply = testObj.calcIndexWithValueSupply(twoDArray, sortedDistanceDict)
    distance_array, finalGrids = testObj.iterateCalculation(twoDArray, indexWithValueSupply, sortedDistanceDict)
    print('————————————————————输送后————————————————————————')
    testObj.showGridNumbers(finalGrids)
    print('最远输送距离为网格{0}至{1}，距离为{2}单位'.format(distance_array[0][0], distance_array[0][1], distance_array[0][2]))
    testObj.showMatrix(finalGrids, name= r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\test2.jpg')


    
    
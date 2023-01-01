import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dis import dis
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

class CalNutritionTransformDistance(object):

    # def __init__(self, DataFile):
    #     self.

    def readData(self, file):
        """read the supply and demand data and preprocessing

        Args:
            file (string): file path

        Returns:
            DataFrame: pandas DF for further calculation
        """
        df = pd.read_csv(file)
        df = df.fillna(0)

        #计算需求值
        df['pneed'] = df['grid_code'] * 30

        #将供给与需求值合并
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
        plt.imshow(fixValues, cmap='magma', norm=matplotlib.colors.Normalize(value.min(), value.max()))
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
        return supplyGridNumbers, supplyTotal, demandGridNumbers, demandTotal, value.shape[0] * value.shape[1]


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
            dict: 字典的键为供给索引号，值为一个二维列表，其中第一个值为需求索引，值为距离
        """
        s_to_d_distance = {}
        for s in supplyIndex:
            s_to_d_distance[s] = []
            for d in demandIndex:
                s_to_d_distance[s].append([d, self.calEucDistance(s, d)])
        return s_to_d_distance

    def sortDisDict(self, s_to_d_distance):
        """对距离字典进行排序

        Args:
            s_to_d_distance (dict): 各供给点至需求点距离

        Returns:
            dict: 排序后的距离字典
        """
        sortedDistanceDict = {}
        for i in s_to_d_distance:
            sortedDistanceDict[i] = sorted(s_to_d_distance[i], key=lambda x: x[1])
        return sortedDistanceDict


    def calcIndexWithValueSupply(self, twoDArray, sortedDistanceDict):
        """为每个供给点索引赋予供给值，并从大到小进行排列

        Args:
            twoDArray (list): 包括所有供需值的二维列表
            sortedDistanceDict (dict): 排序后的供需距离字典

        Returns:
            list: 从大到小排列的供给索引及攻击值
        """
        indexWithValueSupply = []
        for index in sortedDistanceDict.keys():
            supply = twoDArray[index]
            indexWithValueSupply.append([index, supply])
        indexWithValueSupply.sort(key=lambda x : x[1], reverse=True)
        return indexWithValueSupply
        

    

    def iterateCalculation(self, twoDArray, indexWithValueSupply, sortedDistanceDict, resPerGrid):
        """算法的主要实现步骤，计算与迭代

        Args:
            twoDArray ([type]): [description]
            indexWithValueSupply ([type]): [description]
            sortedDistanceDict ([type]): [description]

        Returns:
            [type]: [description]
        """
        transformMass = []
        transformDistance = []
        distance_array = []
        distancePerSupplyList = []
        massPerSupplyList = []
        supplyindex = 0
        testTwoDArray = twoDArray.copy()
        for supply in indexWithValueSupply:
            distancePerSupply = []
            massPerSupply = []
            tempSupply = testTwoDArray[supply[0]]
            for demand in sortedDistanceDict[supply[0]]:
                distancePerSupply.append(demand[1]* resPerGrid)
                tempDemand = testTwoDArray[demand[0]]

                if tempSupply <= -tempDemand:
                    massPerSupply.append(abs(tempSupply))
                    transformMass.append(abs(tempSupply))
                    tempDemand += tempSupply
                    tempSupply = 0
                    testTwoDArray[supply[0]] = tempSupply
                    testTwoDArray[demand[0]] = tempDemand

                    
                    transformDistance.append(demand[1]* resPerGrid)

                    
                elif tempSupply > tempDemand:
                    massPerSupply.append(abs(tempDemand))
                    transformMass.append(abs(tempDemand))
                    tempSupply += tempDemand
                    tempDemand = 0
                    testTwoDArray[supply[0]] = tempSupply
                    testTwoDArray[demand[0]] = tempDemand

                    
                    transformDistance.append(demand[1]* resPerGrid)
                else:
                    pass
                if tempSupply == 0:
                    distance_array.append([supply[0], demand[0], self.calEucDistance(supply[0], demand[0])* resPerGrid])
                    break
            distancePerSupplyList.append(distancePerSupply)
            massPerSupplyList.append(massPerSupply)
            supplyindex += 1
            # self.showMatrix(testTwoDArray, name = r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\show{0}.jpg'.format(supplyindex), shape=(50, 50))
        distance_array.sort(key=lambda x: x[2], reverse=True)
        
        return distance_array, testTwoDArray, transformMass, transformDistance,distancePerSupplyList, massPerSupplyList


        
def calDisResult(fileTitle = 'initGrid', resoulution = 5):
    mwdis = 0
    resPerGrid = 100 / resoulution
    fileName = 'initGrid5'
    calObj = CalNutritionTransformDistance()
    df = calObj.readData(r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\DataWithoutSludge\{0}.csv'.format(fileTitle + str(resoulution)))
    twoDArray, supplyIndex, demandIndex = calObj.calcGridIndex(df['gridResult'].values, (int(df.shape[0]**0.5),int(df.shape[0]**0.5)))
    supplyGridNumbers, supplyTotal, demandGridNumbers, demandTotal, totalGrids = calObj.showGridNumbers(twoDArray)
    s_to_d_distance = calObj.calcIndexDistance(supplyIndex, demandIndex)
    sortedDistanceDict = calObj.sortDisDict(s_to_d_distance)
    indexWithValueSupply = calObj.calcIndexWithValueSupply(twoDArray, sortedDistanceDict)
    max_distance_array, finalGrids, transformMass1, transformDistance1, distancePerSupplyList, massPerSupplyList = calObj.iterateCalculation(twoDArray, indexWithValueSupply, sortedDistanceDict, resPerGrid)
    mwdis = np.dot(np.array(transformMass1), np.array(transformDistance1).T) / np.sum(transformMass1) 
    max_transfer_distance = max_distance_array[0][-1]
    dataDict = {
    'res': resoulution,
    'MWDIS': mwdis,
    'TotalDistance':np.sum(transformDistance1),
    'MaxTraDis':max_transfer_distance,
    'MaxDisPerSupply':max_distance_array,
    'MassPerSupply': massPerSupplyList,
    'DisPerSupply': distancePerSupplyList,
    'CountsSupplyGrids':supplyGridNumbers,
    'ValueSupplyGrids':supplyTotal,
    'CountsDemandGrids':demandGridNumbers,
    'ValueDemandGrids':demandTotal,
    'TotalGrids':totalGrids,
    'SingleTraMass': transformMass1,
    'SingleTraDis': transformDistance1,
    }

    return dataDict

dataSet = []
for r in [5, 10, 20,30,40,50,60,70,80,90,100, 110,120, 130,140,150,160,170,180,190,200]:

    dataSet.append(calDisResult(fileTitle = 'initGrid', resoulution = r))

with open('dataSet2.txt', 'w') as f:
    f.write(str(dataSet))
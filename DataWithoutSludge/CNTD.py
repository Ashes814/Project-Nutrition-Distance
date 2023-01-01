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
        print(1)
        return supplyGridNumbers, supplyTotal, demandGridNumbers, demandTotal


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
        

    

    def iterateCalculation(self, twoDArray, indexWithValueSupply, sortedDistanceDict):
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
        supplyindex = 0
        testTwoDArray = twoDArray.copy()
        for supply in indexWithValueSupply:
            distancePerSupply = []
            tempSupply = testTwoDArray[supply[0]]
            for demand in sortedDistanceDict[supply[0]]:
                distancePerSupply.append()
                tempDemand = testTwoDArray[demand[0]]

                if tempSupply <= -tempDemand:
                    transformMass.append(abs(tempSupply))
                    tempDemand += tempSupply
                    tempSupply = 0
                    testTwoDArray[supply[0]] = tempSupply
                    testTwoDArray[demand[0]] = tempDemand

                    
                    transformDistance.append(demand[1])

                    
                elif tempSupply > tempDemand:
                    transformMass.append(abs(tempDemand))
                    tempSupply += tempDemand
                    tempDemand = 0
                    testTwoDArray[supply[0]] = tempSupply
                    testTwoDArray[demand[0]] = tempDemand

                    
                    transformDistance.append(demand[1])
                else:
                    pass
                if tempSupply == 0:
                    i += 1;
                    distance_array.append([supply[0], demand[0], self.calEucDistance(supply[0], demand[0])])
                    break
            supplyindex += 1
            # self.showMatrix(testTwoDArray, name = r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\show{0}.jpg'.format(supplyindex), shape=(50, 50))
        distance_array.sort(key=lambda x: x[2], reverse=True)
        
        return distance_array, testTwoDArray, transformMass, transformDistance


        













        


# 测试
if __name__ == "__main__":
    def main(fileName,i):
        mwdis = 0
        testObj = CalNutritionTransformDistance()
        df = testObj.readData(r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\{0}.csv'.format(fileName))
        # df.to_csv(r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\draw_100.csv', encoding='utf_8_sig');
        twoDArray, supplyIndex, demandIndex = testObj.calcGridIndex(df['gridResult'].values, (int(df.shape[0]**0.5),int(df.shape[0]**0.5)))
        print(twoDArray[:1])
        print('————————————————————输送前————————————————————————')
        testObj.showGridNumbers(twoDArray)
        # testObj.showMatrix(twoDArray, 
        #                     name = r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\输送前{0}.jpg'.format(fileName), 
        #                     shape = (int(df.shape[0]**0.5),int(df.shape[0]**0.5)))
        s_to_d_distance = testObj.calcIndexDistance(supplyIndex, demandIndex)
        sortedDistanceDict = testObj.sortDisDict(s_to_d_distance)
        indexWithValueSupply = testObj.calcIndexWithValueSupply(twoDArray, sortedDistanceDict)
        distance_array, finalGrids, transformMass1, transformDistance1 = testObj.iterateCalculation(twoDArray, indexWithValueSupply, sortedDistanceDict)
        print('————————————————————输送后————————————————————————')
        # print(transformMass1)
        # print(transformDistance1)
        # print(np.sum(transformMass1))
        mwdis = np.dot(np.array(transformMass1), np.array(transformDistance1).T) / np.sum(transformMass1) 
        print('mwdis : {}'.format(mwdis))
        testObj.showGridNumbers(finalGrids)
        maxOutGrid = distance_array[0][0]
        maxInGrid = distance_array[0][1]
        maxGridNumber = distance_array[0][2]
        print('最远输送距离为网格{0}至{1}，距离为{2}单位'.format(distance_array[0][0], distance_array[0][1], distance_array[0][2] * (100/i)))

        # print(len(transformMass))
        # print(len(transformDistance))
        # print(sum(transformMass))
        # print(transformMass[0:10])
        # print(transformDistance[0:10])

        # print(twoDArray[18][30])
        # testObj.showMatrix(finalGrids, name= r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\输送后{0}.jpg'.format(fileName), 
        #                 shape=(int(df.shape[0]**0.5),int(df.shape[0]**0.5)))
        return maxOutGrid, maxInGrid, maxGridNumber, distance_array, finalGrids, twoDArray, sortedDistanceDict, indexWithValueSupply, mwdis
    # main('RESG5')
    maxOutGridList = []
    maxInGridList = []
    maxGridNumberList = []
    averageLength = []
    mwdisList = []
    fileArray = [100]#[5, 10, 20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
                #  250, 300]
    for i in fileArray:
        # fileName = 'Grid_DRAW_' + str(i)
        fileName = 'NoSludgeGrid100'
        maxOutGrid, maxInGrid, maxGridNumber, distance_array, finalGrids, initGrids, sortedDistanceDict, indexWithValueSupply, mwdis = main(fileName,i)
        maxOutGridList.append(maxOutGrid)
        maxInGridList.append(maxInGrid)
        maxGridNumberList.append(maxGridNumber * (100 / i))
        mwdisList.append(mwdis * (100 / i) )
        sum = 0
        for j in distance_array:
            sum += j[2] * (100 / i)
        averageLength.append(sum/len(distance_array))
        # print(finalGrids[:1])
        reshapeGrids = finalGrids.reshape((-1))
        # print(reshapeGrids[:20])
    print(averageLength)
    df = pd.DataFrame(reshapeGrids, columns=['finalres'])
    # df.to_csv(r'C:\Users\oo\Desktop\Project-Nutrition-Distance\codes\test\draw_final_100.csv', encoding='utf_8_sig');
        # print(pd.DataFrame(finalGrids))
        # print(initGrids)
        # print(sum/len(distance_array))
        # print(indexWithValueSupply[0])
        # print(sortedDistanceDict[(22, 28)][0:10])
    # print(finalGrids)

    # plt.figure()
    # plt.plot(fileArray, mwdisList)
    # plt.show()
    # plt.figure()
    # plt.plot(fileArray, averageLength)
    # plt.show()
    # df = pd.DataFrame([maxOutGridList, maxInGridList, maxGridNumberList])
    # df.to_csv('resg_result.csv')
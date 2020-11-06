#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# <font color='magenta'>Metrics</font>

# <font color='orange'>0. shortestDistance</font>

# In[ ]:


def shortestDistance(x,y,G):
    shortestDistance = []
    if( y not in G.neighbors(x) ):
        score = 0
    else:
        score = nx.dijkstra_path_length(G,x,y)
    return score


# <font color='orange'>1. commonNeighbour</font>

# In[ ]:


def commonNeighbour(x,y,G):
    neighboursNode1 = []
    node1 = x
    node2 = y
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    commonNeighbours = list(set(neighboursNode1) & set(neighboursNode2))
    score = len(commonNeighbours)
    return score


# <font color='orange'>2. jaccardCoefficient</font>

# In[ ]:


def jaccardCoefficient(x,y,G):
    neighboursNode1 = []
    node1 = x
    node2 = y
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    neighboursNode1 = set(neighboursNode1)
    neighboursNode2 = set(neighboursNode2)
    commonNeighbours = neighboursNode1.intersection(neighboursNode2)
    totalNeighbours = neighboursNode1.union(neighboursNode2)
    score = len(commonNeighbours)/len(totalNeighbours)
    return score


# <font color='orange'>3. adamicAdar</font>

# In[ ]:


def adamicAdar(x,y,G):
    import math
    node1 = x
    node2 = y
    neighboursNode1 = []
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    neighboursNode1 = set(neighboursNode1)
    neighboursNode2 = set(neighboursNode2)
    commonNeighbours = neighboursNode1.intersection(neighboursNode2)
    totalNeighbours = neighboursNode1.union(neighboursNode2)
    if( len(commonNeighbours) == 0 ):
        score = 0
    else:
        score = 1.0/math.log(len(commonNeighbours))
    return score


# <font color='orange'>4. preferentialAttachment</font>

# In[ ]:


def preferentialAttachment(x,y,G):
    visitedNodeDict = {}
    node1 = x
    node2 = y
    neighboursNode1 = []
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    score = len(neighboursNode1) * len(neighboursNode2)
    return score


# <font color='orange'>5. katz</font>

# In[ ]:


def katz(x,y,G):
    import math
    beta = 1
    node1 = x
    node2 = y
    neighboursNode1 = []
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    maxLength = 0
    for path in nx.all_simple_paths(G, source=node1, target=node2):
        if( len(path) > maxLength ):
            maxLength = len(path)
    sumPath = 0
    for i in range(1,maxLength+1):
        count = 0
        for path in nx.all_simple_paths(G, source=node1, target=node2):
            if( len(path) == i ):
                count+=1
        sumPath += math.pow(beta,i) * count
    score = sumPath
    return score


# <font color='orange'>6. hittingTime</font>

# In[ ]:


def hittingTime(x,y,G,A):
    import scipy as sp
    import math
    import numpy as np
    
    node1 = x
    node2 = y
    beta = 1
    
    index = (node1-1,node2-1)
    C = A.copy()
    C += C.T
    C[index[1],:]=0
    C[index[1],index[1]]=1
    C = (C.T/C.sum(axis=1)).T
    B = C.copy()
    Z = []
    for n in range(100):
        Z.append( B[index] )
        B = np.dot(B,C)
    Z = np.array(Z)
    score = np.mean(Z)
    return score


# <font color='orange'>7. commuteTime</font>

# In[ ]:


def commuteTime(x,y,G,A):
    score = hittingTime(x,y,G,A) + hittingTime(y,x,G,A)
    return score


# <font color='orange'>8. simRank</font>

# In[ ]:


def simRank(x,y,gamma,G):
    score = 0
    i = 0
    j = 0
    if( x == y ):
        return 1
    for node1 in G.neighbors(x):
        for node2 in G.neighbors(y):
            score +=simRank(node1,node2,gamma,G)
            j+=1
        i+=1
    score = score/(i*j)
    return score


# <font color='orange'>9. Rank Approximation</font>

# In[ ]:


import numpy as np
def matrixDecomposition(A,r):
    u,s,v = np.linalg.svd(A)
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar


# In[ ]:


def lowRankApproximationKatz(x,y,G,A):
    import math
    beta = 1
    node1 = x
    node2 = y
    Ad = matrixDecomposition(A,r)
    maxLen = 0
    for path in nx.all_simple_paths(G, source=node1, target=node2):
        if( len(path) > maxLength ):
            maxLength = len(path)
    for i in range(1,maxLength+1):
        sumPath += math.pow(beta,i) * Ad[node1][node2]
        Ad = np.dot(Ad,Ad)
    score = sumPath
    return score


# In[ ]:


def lowRankApproximationCommonNeighbours(x,y,G,A):
    import math
    beta = 1
    node1 = x
    node2 = y
    Ad = matrixDecomposition(A,r)
    v1 = Ad[x]
    v2 = Ad[y]
    score = np.dot(Ad[x],Ad[y].T)
    return score


# In[ ]:


def lowRankApproximationBasic(x,y,G,A):
    import math
    beta = 1
    node1 = x
    node2 = y
    Ad = matrixDecomposition(A,r)
    score = np.dot(Ad[x,y])
    return score


# <font color='orange'>10. Unseen Bi-grams</font>

# In[ ]:


def unseenBigrams(x,y,G,A):
    import scipy as sp
    import math
    import numpy as np
    
    node1 = x
    node2 = y
    beta = 1
    
    index = (node1-1,node2-1)
    C = A.copy()
    C += C.T
    C[index[1],:]=0
    C[index[1],index[1]]=1
    C = (C.T/C.sum(axis=1)).T
    B = C.copy()
    Z = []
    for n in range(100):
        Z.append( B[index] )
        B = np.dot(B,C)
    Z = np.array(Z)
    score = np.mean(Z)
    return score


# In[ ]:


H = nx.read_edgelist("M.Tech/Data2/socfb-Caltech36.txt",create_using = nx.Graph(),nodetype=int)


# In[ ]:


H.number_of_nodes()


# In[ ]:


H.number_of_edges()


# In[ ]:


G = nx.Graph()
edges = nx.min_edge_cover(H)


# In[ ]:


G.add_nodes_from(H.nodes())
G.add_edges_from(edges)


# In[ ]:


G.number_of_nodes()


# In[ ]:


G.number_of_edges()


# In[ ]:


import warnings
warnings.simplefilter('ignore')


# In[ ]:


#G = H
shortestDistanceL = []
commonNeighbourL = []
jaccardCoefficientL = []
adamicAdarL = []
preferentialAttachmentL = []
katzL = []
hittingTimeL = []
commuteTimeL = []
rootedPageRankL = []
simRankL = []

A = nx.adjacency_matrix(G)
#A = A.todense()
#A = np.array(A)
count = 0
#create a visited dictionary for every node of the graph
visitedNodeDict = {}
for node in G:
    visitedNodeDict[node] = 0
for node1 in G:
    for node2 in G:
        #print("(",node1,node2,")")
        if( node1 == node2 or visitedNodeDict[node2] == 1 ):
                continue
        #score = shortestDistance(node1,node2,G)
        #shortestDistanceL.append([(node1,node2),score])
        #commonNeighbourL.append([(node1,node2),commonNeighbour(node1,node2,G)])
        #jaccardCoefficiend3a88f9259af78be8cdba74ef6dae0941efbb9d657bebd9atL.append([(node1,node2),jaccardCoefficient(node1,node2,G)])
        #adamicAdarL.append([(node1,node2),adamicAdar(node1,node2,G)])
        #preferentialAttachmentL.append([(node1,node2),preferentialAttachment(node1,node2,G)])
        #katzL.append([(node1,node2),katz(node1,node2,G)])
        #hittingTimeL.append([(node1,node2),hittingTime(node1,node2,G,A)])
        #print("Count: ",count)
        count+=1
        #commuteTime.append([x,y,commuteTime(x,y,G,A)])
        #rootedPageRank.append([x,y,rootedPageRank(x,y,G)])
        gamma = 1 #gamma belongs to [0,1]
        simRankL.append([(node1,node2),simRank(node1,node2,gamma,G)])
    visitedNodeDict[node1] = 1

#shortestDistanceL = np.array(shortestDistanceL)
#shortestDistanceSortedL = shortestDistanceL[shortestDistanceL[:,1].argsort()]
           
#commonNeighbourL = np.array(commonNeighbourL)
#commonNeighbourSorted = commonNeighbourL[commonNeighbourL[:,1].argsort()]

#jaccardCoefficientL = np.array(jaccardCoefficientL)
#jaccardCoefficientSorted = jaccardCoefficientL[jaccardCoefficientL[:,1].argsort()]
           
#adamicAdarL = np.array(adamicAdarL)
#adamicAdarSorted = adamicAdarL[adamicAdarL[:,1].argsort()]

#hittingTimeL = np.array(hittingTimeL)
#hittingTimeSorted = hittingTimeL[hittingTimeL[:,1].argsort()]

#katz = np.array(katz)
#katzSorted = katz[katz[:,2].argsort()]

#hittingTime = np.array(hittingTime)
#hittingTimeSorted = hittingTime[hittingTime[:,2].argsort()]

#katz = np.array(katz)
#katzSorted = katz[katz[:,2].argsort()]

#hittingTime = np.array(hittingTime)
#hittingTimeSorted = hittingTime[hittingTime[:,2].argsort()]
           
#commuteTime = np.array(commuteTime)
#commuteTimeSorted = commuteTime[commuteTime[:,2].argsort()]

simRankL = np.array(simRankL)
simRankSorted = simRankL[simRankL[:,2].argsort()]
           
#lowRankApproximationKatz = np.array(lowRankApproximationKatz)
#lowRankApproximationKatzSorted = lowRankApproximationKatz[lowRankApproximationKatz[:,2].argsort()]

#lowRankApproximationCommonNeighbours = np.array(lowRankApproximationCommonNeighbours)
#lowRankApproximationCommonNeighboursSorted = lowRankApproximationCommonNeighbours[lowRankApproximationCommonNeighbours[:,2].argsort()]

#lowRankApproximationBasic = np.array(lowRankApproximationBasic)
#lowRankApproximationBasicSorted = lowRankApproximationBasic[lowRankApproximationBasic[:,2].argsort()]


# In[ ]:


len(hittingTimeL)


# In[ ]:


#Taking the top-K values
len(shortestDistanceSortedL)


# In[ ]:


complete_edge_list = H.edges()


# In[ ]:


complete_edge_list = list(complete_edge_list)


# In[ ]:


count=0
for item in shortestDistanceSortedL:
    if( ( item[0] in complete_edge_list or ((item[0][1],item[1][0]) in complete_edge_list) ) and (item[0] not in edges) ):
        count+=1


# In[ ]:


count


# In[ ]:


count/(len(complete_edge_list) - len(edges))


# In[ ]:


#Common Neighbour
print("Edges: ",len(commonNeighbourSorted))
count=0
for item in commonNeighbourSorted:
    if( (item[0] in complete_edge_list) and (item[0] not in edges) ):
        count+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[ ]:


#jaccardCoefficientSorted
print("Edges: ",len(jaccardCoefficientSorted))
count=0
for item in jaccardCoefficientSorted:
    if( (item[0] in complete_edge_list) and (item[0] not in edges) ):
        count+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[ ]:


#preferentialAttachmentSorted
normalizedScores = np.array(preferentialAttachmentSorted)
normalizedScores = normalizedScores[:,1]
normalizedScores = normalizedScores/normalizedScores.max()
print("Edges: ",len(normalizedScores))
count=0
index = 0
for item in preferentialAttachmentSorted:
    if( ( item[0] in complete_edge_list or ((item[0][1],item[0][0]) in complete_edge_list)  ) and (item[0] not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    index+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[ ]:


#preferentialAttachmentSorted
normalizedScores = np.array(katzSorted)
normalizedScores = normalizedScores[:,1]
normalizedScores = normalizedScores/normalizedScores.max()
print("Edges: ",len(normalizedScores))
count=0
index = 0
for item in katzSorted:
    if( ( item[0] in complete_edge_list or ((item[0][1],item[0][0]) in complete_edge_list)  ) and (item[0] not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    index+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[ ]:


#preferentialAttachmentSorted
normalizedScores = np.array(katzSorted)
normalizedScores = normalizedScores[:,1]
normalizedScores = normalizedScores/normalizedScores.max()
print("Edges: ",len(normalizedScores))
count=0
index = 0
for item in katzSorted:
    if( ( item[0] in complete_edge_list or ((item[0][1],item[0][0]) in complete_edge_list)  ) and (item[0] not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    index+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[ ]:





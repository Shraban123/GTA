#!/usr/bin/env python
# coding: utf-8

# In[4]:


import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# <font color='magenta'>Metrics</font>

# <font color='orange'>0. shortestDistance</font>

# In[5]:


def shortestDistance(x,y,G):
    shortestDistance = []
    if( y not in G.neighbors(x) ):
        score = 0
    else:
        score = nx.dijkstra_path_length(G,x,y)
    return score


# <font color='orange'>1. commonNeighbour</font>

# In[6]:


def commonNeighbour(x,y,z,G):
    neighboursNode1 = []
    node1 = x
    node2 = y
    node3 = z
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    neighboursNode3 = []
    for item in G.neighbors(node3):
        neighboursNode3.append(item)
    commonNeighbours = list(set(neighboursNode1) & (set(neighboursNode2).union(set(neighboursNode3))))
    score = len(commonNeighbours)
    return score


# <font color='orange'>2. jaccardCoefficient</font>

# In[7]:


def jaccardCoefficient(x,y,z,G):
    neighboursNode1 = []
    node1 = x
    node2 = y
    node3 = z
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    neighboursNode3 = []
    for item in G.neighbors(node3):
        neighboursNode3.append(item)
    neighboursNode1 = set(neighboursNode1)
    neighboursNode2 = set(neighboursNode2)
    neighboursNode3 = set(neighboursNode3)
    commonNeighbours = neighboursNode1.intersection(neighboursNode2.union(neighboursNode3))
    totalNeighbours = neighboursNode1.union(neighboursNode2.union(neighboursNode3))
    score = len(commonNeighbours)/len(totalNeighbours)
    return score


# <font color='orange'>3. adamicAdar</font>

# In[28]:


def adamicAdar(x,y,z,G):
    import math
    node1 = x
    node2 = y
    node3 = z
    neighboursNode1 = []
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    neighboursNode3 = []
    for item in G.neighbors(node3):
        neighboursNode3.append(item)
    neighboursNode1 = set(neighboursNode1)
    neighboursNode2 = set(neighboursNode2)
    neighboursNode3 = set(neighboursNode3)
    commonNeighbours = neighboursNode1.intersection(neighboursNode2.union(neighboursNode3))
    #totalNeighbours = neighboursNode1.union(neighboursNode2.union(neighboursNode3))
    if( len(commonNeighbours) == 0 ):
        score = 100
    else:
        if(len(commonNeighbours) == 1):
            score = 100 
        else:
            score = 1.0/math.log(len(commonNeighbours))
    return score


# <font color='orange'>4. preferentialAttachment</font>

# In[9]:


def preferentialAttachment(x,y,z,G):
    visitedNodeDict = {}
    node1 = x
    node2 = y
    node3 = z
    neighboursNode1 = []
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    neighboursNode3 = []
    for item in G.neighbors(node3):
        neighboursNode3.append(item)
    
    neighboursNode1 = set(neighboursNode1)
    neighboursNode2 = set(neighboursNode2)
    neighboursNode3 = set(neighboursNode3)
    
    score = len(neighboursNode1) * len(neighboursNode2.union(neighboursNode3))
    return score


# <font color='orange'>5. katz</font>

# In[36]:


def katz(x,y,z,G):
    import math
    beta = 1
    node1 = x
    node2 = y
    node3 = z
    neighboursNode1 = []
    for item in G.neighbors(node1):
        neighboursNode1.append(item)
    neighboursNode2 = []
    for item in G.neighbors(node2):
        neighboursNode2.append(item)
    neighboursNode3 = []
    for item in G.neighbors(node3):
        neighboursNode3.append(item)
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

# In[11]:


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

# In[12]:


def commuteTime(x,y,G,A):
    score = hittingTime(x,y,G,A) + hittingTime(y,x,G,A)
    return score


# <font color='orange'>8. rootedPageRank</font>

# <font color='orange'>8. simRank</font>

# In[13]:


def simRank(x,y,gamma,G):
    score = 0
    i = 0
    j = 0
    if( x == y ):
        return 1
    for node1 in G.neighbors(x):
        for node2 in G.neighbors(y):
            score +=simRank(node1,node2)
            j+=1
        i+=1
    score = score/(i*j)
    return score


# <font color='orange'>9. Rank Approximation</font>

# In[14]:


import numpy as np
def matrixDecomposition(A,r):
    u,s,v = np.linalg.svd(A)
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar


# In[15]:


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


# In[16]:


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


# In[17]:


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





# In[18]:


H = nx.read_edgelist("M.Tech/Data2/socfb-Caltech36.txt",create_using = nx.Graph(),nodetype=int)


# In[19]:


H.number_of_nodes()


# In[20]:


H.number_of_edges()


# In[21]:


G = nx.Graph()
edges = nx.min_edge_cover(H)


# In[22]:


G.add_nodes_from(H.nodes())
G.add_edges_from(edges)


# In[23]:


G.number_of_nodes()


# In[24]:


G.number_of_edges()


# In[25]:


import warnings
warnings.simplefilter('ignore')


# In[37]:


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

#create a visited dictionary for every node of the graph
visitedNodeDict = {}
for node in G:
    visitedNodeDict[node] = 0
for node1 in G:
    for item in G.edges():
        #print("(",node1,node2,")")
        node2 = item[0]
        node3 = item[1]
        if( node1 == node2 or visitedNodeDict[node2] == 1 ):
                continue
        #score = shortestDistance(node1,node2,G)
        #shortestDistanceL.append([(node1,node2),score])
        commonNeighbourL.append([(node1,node2,node3),commonNeighbour(node1,node2,node3,G)])
        jaccardCoefficientL.append([(node1,node2,node3),jaccardCoefficient(node1,node2,node3,G)])
        adamicAdarL.append([(node1,node2,node3),adamicAdar(node1,node2,node3,G)])
        preferentialAttachmentL.append([(node1,node2,node3),preferentialAttachment(node1,node2,node3,G)])
        katzL.append([(node1,node2,node3),katz(node1,node2,node3,G)])
        #hittingTimeL.append([(node1,node2),hittingTime(node1,node2,G,A)])
        #commuteTime.append([x,y,commuteTime(x,y,G,A)])
        #rootedPageRank.append([x,y,rootedPageRank(x,y,G)])
        #gamma = 1 #gamma belongs to [0,1]
        #simRank.append([x,y,simRank(x,y,gamma,G)])
    visitedNodeDict[node1] = 1

#shortestDistanceL = np.array(shortestDistanceL)
#shortestDistanceSortedL = shortestDistanceL[shortestDistanceL[:,1].argsort()]
           
commonNeighbourL = np.array(commonNeighbourL)
commonNeighbourSorted = commonNeighbourL[commonNeighbourL[:,1].argsort()]

jaccardCoefficientL = np.array(jaccardCoefficientL)
jaccardCoefficientSorted = jaccardCoefficientL[jaccardCoefficientL[:,1].argsort()]
           
adamicAdarL = np.array(adamicAdarL)
adamicAdarSorted = adamicAdarL[adamicAdarL[:,1].argsort()]

preferentialAttachmentL = np.array(preferentialAttachmentL)
preferentialAttachmentSorted = preferentialAttachmentL[preferentialAttachmentL[:,1].argsort()]

katzL = np.array(katzL)
katzSorted = katzL[katzL[:,1].argsort()]

#hittingTimeL = np.array(hittingTimeL)
#hittingTimeSorted = hittingTimeL[hittingTimeL[:,1].argsort()]

#katz = np.array(katz)
#katzSorted = katz[katz[:,2].argsort()]

#hittingTime = np.array(hittingTime)
#hittingTimeSorted = hittingTime[hittingTime[:,2].argsort()]
           
#commuteTime = np.array(commuteTime)
#commuteTimeSorted = commuteTime[commuteTime[:,2].argsort()]

#simRank = np.array(simRank)
#simRankSorted = simRank[simRank[:,2].argsort()]
           
#lowRankApproximationKatz = np.array(lowRankApproximationKatz)
#lowRankApproximationKatzSorted = lowRankApproximationKatz[lowRankApproximationKatz[:,2].argsort()]

#lowRankApproximationCommonNeighbours = np.array(lowRankApproximationCommonNeighbours)
#lowRankApproximationCommonNeighboursSorted = lowRankApproximationCommonNeighbours[lowRankApproximationCommonNeighbours[:,2].argsort()]

#lowRankApproximationBasic = np.array(lowRankApproximationBasic)
#lowRankApproximationBasicSorted = lowRankApproximationBasic[lowRankApproximationBasic[:,2].argsort()]


# In[38]:


complete_edge_list = H.edges()


# In[39]:


complete_edge_list = list(complete_edge_list)


# In[43]:


#Common Neighbour
normalizedScores = np.array(commonNeighbourSorted)
normalizedScores = normalizedScores[:,1]
normalizedScores = normalizedScores/normalizedScores.max()
print("Edges: ",len(normalizedScores))
count=0
index = 0
for item in commonNeighbourSorted:
    edge = (item[0][0],item[0][1])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    edge = (item[0][0],item[0][2])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    index+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[44]:


#jaccardCoefficientSorted
normalizedScores = np.array(jaccardCoefficientSorted)
normalizedScores = normalizedScores[:,1]
normalizedScores = normalizedScores/normalizedScores.max()
print("Edges: ",len(jaccardCoefficientSorted))
count=0
index = 0
for item in jaccardCoefficientSorted:
    edge = (item[0][0],item[0][1])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    edge = (item[0][0],item[0][2])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    index+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[ ]:


#adamicAdarSorted
normalizedScores = np.array(adamicAdarSorted)
normalizedScores = normalizedScores[:,1]
normalizedScores = normalizedScores/normalizedScores.max()
print("Edges: ",len(adamicAdarSorted))
count=0
index = 0
for item in adamicAdarSorted:
    edge = (item[0][0],item[0][1])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    edge = (item[0][0],item[0][2])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    index+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[48]:


#preferentialAttachmentSorted
normalizedScores = np.array(preferentialAttachmentSorted)
normalizedScores = normalizedScores[:,1]
normalizedScores = normalizedScores/normalizedScores.max()
print("Edges: ",len(preferentialAttachmentSorted))
count=0
index = 0
for item in preferentialAttachmentSorted:
    edge = (item[0][0],item[0][1])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    edge = (item[0][0],item[0][2])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    index+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[49]:


#katzSorted
normalizedScores = np.array(katzSorted)
normalizedScores = normalizedScores[:,1]
normalizedScores = normalizedScores/normalizedScores.max()
print("Edges: ",len(katzSorted))
count=0
index = 0
for item in katzSorted:
    edge = (item[0][0],item[0][1])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    edge = (item[0][0],item[0][2])
    if( ( edge in complete_edge_list or ((edge[1],edge[0]) in complete_edge_list)  ) and (edge not in edges) ):
        if(normalizedScores[index] <= 0.4):
            count+=1
    index+=1
print("New edges: ",count)
print("Accuracy: ",count/(len(complete_edge_list) - len(edges)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[111]:


(len(complete_edge_list) - len(edges))


# In[55]:


G.number_of_edges()


# In[56]:


preferentialAttachmentSorted


# In[74]:


if ((5,12) not in complete_edge_list):
    print("trueee")


# In[75]:


772-386


# ### Changing the functions

# In[ ]:





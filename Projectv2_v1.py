#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import matplotlib.pyplot as plt
import networkx as nx


# In[7]:


import os
directory = r'M.Tech/Data'
dicList = []
for file in os.listdir(directory):
    fp = open(directory+'/'+file,errors='ignore')
    for line in fp:
        res = json.loads(line)
        dicList.append(res)
    fp.close()


# In[8]:


G = nx.Graph()


# In[9]:


authorIdDic = {}
i = 0
for item in dicList:
    authorList = item['authors']
    authorNodes = []
    for author in authorList:
        #print("-----",author)
        if( len(author['ids']) == 0 ):
            #print("skipped")
            continue
        if author['ids'][0] not in authorIdDic.keys():
            authorIdDic[author['ids'][0]] = i
            i+=1
        authorNodes.append(authorIdDic[author['ids'][0]])
    for k in range(len(authorNodes)):
        for j in range(len(authorNodes)):
            if( k == j ):
                continue
            G.add_edge(authorNodes[k],authorNodes[j])


# In[23]:


nx.number_of_edges(G)


# In[24]:


nx.number_of_nodes(G)


# In[28]:


3334674 + i


# In[25]:


i


# In[26]:


len(authorIdDic)


# In[27]:


3981735 - 3334674


# In[ ]:


#Break the graph into train and test based on year
#Remove all the nodes after a particular year. Take 5 years as the training set
#Calculate the number of links in the train set and the test set


# <font color='orange'>0. shortestDistance</font>

# In[50]:


import numpy as np
a =np.array([[1,2,3],[1,2,2],[1,2,1]])


# In[52]:


b = a[a[:,2].argsort()]


# In[51]:


a


# In[53]:


b


# In[ ]:


def shortestDistance(x,y,G):
    shortestDistance = []
    visitedNodeDict = {}
    for node in G:
        visitedNodeDict[node] = 0
    for node1 in G:
        for node2 in G:
            if( node1 == node2 || visitedNodeDict[node2] == 1 || node2 in G.neighbors(node1)):
                continue
            shortestDistance.append([node1,node2,nx.dijkstra_path_length(G,node1,node2)])
        visitedNodeDict[node1] = 1
    
    #This portion will be common for every function
    shortestDistance = np.array(shortestDistance)
    shortestDistanceSorted = shortestDistance[shortestDistance[:,2].argsort()]
    
    #return the top k items of shortestDistanceSorted


# In[3]:


t=nx.balanced_tree(2, 4)


# In[4]:


l = nx.dijkstra_path_length(t,1,2)


# In[5]:


l = nx.dijkstra_path_length(G,1,2)


# In[6]:


nx.draw(t)


# In[9]:


adj = nx.adjacency_matrix(t)


# In[26]:


import scipy as sp
print(adj)


# In[27]:


adj = adj.todense()


# In[30]:


adj


# In[32]:


import numpy as np
adj = np.array(adj)


# In[33]:


adj


# In[37]:


for node1 in t:
    for node2 in t:
        if( node1 == node2 ):
            continue
        (nx.dijkstra_path_length(t,node1,node2))


# In[34]:


for item in t.neighbors(0):
    print(item)


# In[55]:


if(2 in t.neighbors(1)):
    print('true')
else:
    print('false')


# In[22]:


for node1 in G:
    for node2 in G:
        if( node1 == node2 ):
            continue
        if(node2 in G.neighbors(node1)):
            print(nx.dijkstra_path(G,node1,node2))
        else:
            print("skipped")


# In[12]:


for item in l:
    print(item)
    break


# <font color='orange'>1. commonNeighbour</font>

# In[ ]:


def commonNeighbour(x,y,G):
    commonNeighbour = []
    visitedNodeDict = {}
    for node in G:
        visitedNodeDict[node] = 0
    for node1 in G:
        for node2 in G:
            if( node1 == node2 || visitedNodeDict[node2] == 1 || node2 in G.neighbors(node1)):
                continue
            neighboursNode1 = []
            for item in G.neighbours(node1):
                neighboursNode1.append(item)
            neighboursNode2 = []
            for item in G.neighbours(node2):
                neighboursNode2.append(item)
            commonNeighbours = list(set(neighboursNode1) & set(neighboursNode2))
            score = len(commonNeighbours)
            commonNeighbour.append([node1,node2,score])
        visitedNodeDict[node1] = 1
    
    #This portion will be common for every function
    commonNeighbour = np.array(commonNeighbour)
    commonNeighbourSorted = commonNeighbour[commonNeighbour[:,2].argsort()]
    
    #return the top k items of shortestDistanceSorted


# <font color='orange'>2. jaccardCoefficient</font>

# In[62]:


def intersection(lst1, lst2): 
    lst1 = set(lst1)
    lst2 = set(lst2)
    z = lst1.intersection(lst2)
    return z
  
# Driver Code 
lst1 = [15, 9, 10, 56, 23, 78, 5, 4, 9] 
lst2 = [9, 4, 5, 36, 47, 26, 10, 45, 87] 
print(intersection(lst1, lst2)) 


# In[ ]:


def jaccardCoefficient(x,y,G):
    commonNeighbour = []
    visitedNodeDict = {}
    for node in G:
        visitedNodeDict[node] = 0
    for node1 in G:
        for node2 in G:
            if( node1 == node2 || visitedNodeDict[node2] == 1 || node2 in G.neighbors(node1)):
                continue
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
            score = len(commonNeighbours)/len(totalNeighbours)
            commonNeighbour.append([node1,node2,score])
        visitedNodeDict[node1] = 1
    
    #This portion will be common for every function
    commonNeighbour = np.array(commonNeighbour)
    commonNeighbourSorted = commonNeighbour[commonNeighbour[:,2].argsort()]
    
    #return the top k items of shortestDistanceSorted


# <font color='orange'>3. adamicAdar</font>

# In[ ]:


def adamicAdar(x,y,G):
    import math
    commonNeighbour = []
    visitedNodeDict = {}
    for node in G:
        visitedNodeDict[node] = 0
    for node1 in G:
        for node2 in G:
            if( node1 == node2 || visitedNodeDict[node2] == 1 || node2 in G.neighbors(node1)):
                continue
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
            score = 1.0/math.log(len(commonNeighbours),2)
            commonNeighbour.append([node1,node2,score])
        visitedNodeDict[node1] = 1
    
    #This portion will be common for every function
    commonNeighbour = np.array(commonNeighbour)
    commonNeighbourSorted = commonNeighbour[commonNeighbour[:,2].argsort()]
    
    #return the top k items of shortestDistanceSorted


# <font color='orange'>4. preferentialAttachment</font>

# In[ ]:


def preferentialAttachment(x,y,G):
    commonNeighbour = []
    visitedNodeDict = {}
    for node in G:
        visitedNodeDict[node] = 0
    for node1 in G:
        for node2 in G:
            if( node1 == node2 || visitedNodeDict[node2] == 1 || node2 in G.neighbors(node1)):
                continue
            neighboursNode1 = []
            for item in G.neighbors(node1):
                neighboursNode1.append(item)
            neighboursNode2 = []
            for item in G.neighbors(node2):
                neighboursNode2.append(item)
            score = len(neighboursNode1) * len(neighboursNode2)
            commonNeighbour.append([node1,node2,score])
        visitedNodeDict[node1] = 1
    
    #This portion will be common for every function
    commonNeighbour = np.array(commonNeighbour)
    commonNeighbourSorted = commonNeighbour[commonNeighbour[:,2].argsort()]
    
    #return the top k items of shortestDistanceSorted


# <font color='orange'>5. katz</font>

# In[ ]:


def katz(x,y,G):
    import math
    commonNeighbour = []
    visitedNodeDict = {}
    beta = 1
    for node in G:
        visitedNodeDict[node] = 0
    for node1 in G:
        for node2 in G:
            if( node1 == node2 || visitedNodeDict[node2] == 1 || node2 in G.neighbors(node1)):
                continue
            neighboursNode1 = []
            for item in G.neighbors(node1):
                neighboursNode1.append(item)
            neighboursNode2 = []
            for item in G.neighbors(node2):
                neighboursNode2.append(item)
            maxLen = 0
            for path in nx.all_simple_paths(G, source=node1, target=node2):
                if( len(path) > maxLength ):
                    maxLength = path
            sumPath = 0
            for i in range(1,l+1):
                count = 0
                for path in nx.all_simple_paths(G, source=node1, target=node2):
                if( len(path) == i ):
                    count+=1
                sumPath += math.pow(beta,i) * count
            score = sumPath
            commonNeighbour.append([node1,node2,score])
        visitedNodeDict[node1] = 1
    
    #This portion will be common for every function
    commonNeighbour = np.array(commonNeighbour)
    commonNeighbourSorted = commonNeighbour[commonNeighbour[:,2].argsort()]
    
    #return the top k items of shortestDistanceSorted


# <font color='orange'>6. hittingTime</font>

# In[29]:


adj.shape


# In[ ]:


def hittingTime(x,y,G):
    import scipy as sp
    import math
    import numpy as np
    commonNeighbour = []
    visitedNodeDict = {}
    beta = 1
    for node in G:
        visitedNodeDict[node] = 0
    for node1 in G:
        for node2 in G:
            if( node1 == node2 || visitedNodeDict[node2] == 1 || node2 in G.neighbors(node1)):
                continue
            neighboursNode1 = []
            for item in G.neighbors(node1):
                neighboursNode1.append(item)
            neighboursNode2 = []
            for item in G.neighbors(node2):
                neighboursNode2.append(item)
            maxLen = 0
            for path in nx.all_simple_paths(G, source=node1, target=node2):
                if( len(path) > maxLength ):
                    maxLength = path
            sumPath = 0
            index = (node1,node2)
            A = nx.adjacency_matrix(G)
            A = A.todense()
            A = np.array(A)
            A += A.T
            A[index[1],:]=0
            A[index[1],index[1]]=1
            A = (A.T/A.sum(axis=1)).T
            B = A.copy()
            Z = []
            for n in xrange(100):
                Z.append( B[index] )
                B = dot(B,A)
            Z = np.array(Z)
            score = np.mean(Z)
            commonNeighbour.append([node1,node2,score])
        visitedNodeDict[node1] = 1
    
    #This portion will be common for every function
    commonNeighbour = np.array(commonNeighbour)
    commonNeighbourSorted = commonNeighbour[commonNeighbour[:,2].argsort()]
    
    #return the top k items of shortestDistanceSorted


# <font color='orange'>7. commuteTime</font>

# In[ ]:


def commuteTime(x,y,G):
    import scipy as sp
    import math
    import numpy as np
    commonNeighbour = []
    visitedNodeDict = {}
    beta = 1
    for node in G:
        visitedNodeDict[node] = 0
    for node1 in G:
        for node2 in G:
            if( node1 == node2 || visitedNodeDict[node2] == 1 || node2 in G.neighbors(node1)):
                continue
            neighboursNode1 = []
            for item in G.neighbors(node1):
                neighboursNode1.append(item)
            neighboursNode2 = []
            for item in G.neighbors(node2):
                neighboursNode2.append(item)
            maxLen = 0
            for path in nx.all_simple_paths(G, source=node1, target=node2):
                if( len(path) > maxLength ):
                    maxLength = path
            sumPath = 0
            index = (node1,node2)
            A = nx.adjacency_matrix(G)
            A = A.todense()
            A = np.array(A)
            A += A.T
            A[index[1],:]=0
            A[index[1],index[1]]=1
            A = (A.T/A.sum(axis=1)).T
            B = A.copy()
            Z = []
            for n in xrange(100):
                Z.append( B[index] )
                B = dot(B,A)
            Z = np.array(Z)
            score = np.mean(Z)
            commonNeighbour.append([node1,node2,score])
        visitedNodeDict[node1] = 1
    
    #This portion will be common for every function
    commonNeighbour = np.array(commonNeighbour)
    commonNeighbourSorted = commonNeighbour[commonNeighbour[:,2].argsort()]
    
    #return the top k items of shortestDistanceSorted


# <font color='orange'>8. rootedPageRank</font>

# In[35]:


len(t.neighbors(1))


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
            score +=simRank(node1,node2)
            j+=1
        i+=1
    score = score/(i*j)
    return score


# In[4]:


l = nx.all_pairs_shortest_path_length(G)


# In[3]:


def commonNeighbour(x,y,G):
    all_pairs_shortest_path_length(G[, cutoff])


# In[ ]:


shortestDistance = []
commonNeighbour = []
jaccardCoefficient = []
adamicAdar = []
preferentialAttachment = []
katz = []
hittingTime = []
commuteTime = []
rootedPageRank = []
simRank = []
#create a visited dictionary for every node of the graph
for node1 in G:
    for node2 in G:
        if( node1 == node2 ):
            continue
        if( )
        shortestDistance.append(x,y,shortestDistance(x,y,G))
        commonNeighbour.append(x,y,commonNeighbour(x,y,G))
        jaccardCoefficient.append(x,y,commonNeighbour(x,y,G))
        adamicAdar.append(x,y,adamicAdar(x,y,G))
        preferentialAttachment.append(x,y,preferentialAttachment(x,y,G))
        katz.append(x,y,katz(x,y,G))
        hittingTime.append(x,y,hittingTime(x,y,G))
        commuteTime.append(x,y,commuteTime(x,y,G))
        rootedPageRank.append(x,y,rootedPageRank(x,y,G))
        simRank.append(x,y,simRank(x,y,G))


# In[ ]:


authorIdDic = {}
i = 0
for item in dicList:
    authorList = item['authors']
    authorNodes = []
    for author in authorList:
        #print("-----",author)
        if( len(author['ids']) == 0 ):
            #print("skipped")
            continue
        if author['ids'][0] not in authorIdDic.keys():
            authorIdDic[author['ids'][0]] = i
            i+=1
        authorNodes.append(authorIdDic[author['ids'][0]])
    for k in range(len(authorNodes)):
        for j in range(len(authorNodes)):
            if( k == j ):
                continue
            G.add_edge(authorNodes[k],authorNodes[j])


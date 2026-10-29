import numpy as np

import math
import random

#This file should provide the implementation of a neural network
#The built-in methods are:
#
# init - creates a neural network with inputs of each layer side
# forward - passes inputs through the neural network and returns the output
# mutate - mutates neural network based on a normal distribution, takes in inputs of range 0-1 of how likely a given weight is to mutate, and the standard deviation
#
# sigmoid
# relu
# tanH

# Input Neurons are:
# (Normalised) angle to nearest creature
# (Normalised) angle to nearest food
# proximity to nearest creature
# proximity to nearest food (of plant/meat type)
# proximity to nearest wall
# Random
# Constant

# Output Neurons are:
# new speed
# angle change
# biting state

# Functionality includes mutations for input/output nodes, and for biases

class Neural_Network:
    defaultWeight = 1 #The default value of a weight when a connection is mutated
    inputNeuronChance = 0.5 #The chance of an input neuron being created if a new node is being added
    outputNeuronChance = 0.5 #The same but for an output neuron


    #This indicates the ratios of the probabilites of each mutation happening
    mutationRatios = {"Update Existing Weight":6, "Update Existing Bias":3, "Add Connection":2, "Remove Connection":2, "Add Node":1, "Remove Node":1}
    #mutationRatios = {"Update Existing Weight":12, "Update Existing Bias":1, "Add Connection":2, "Remove Connection":2, "Add Node":1, "Remove Node":1} #This is another example of a mutation ratio.
    #i.e. in the example above, updating a weight has a 12x higher chance of occuring than adding a node

    def __init__(self,nodeBlueprint:str):

        #This is just a string version of the network
        self.__nodeBlueprint = nodeBlueprint

        #Turns the string version into an evaluatable structure
        nodeDictionary:dict = eval(nodeBlueprint)

        #The nodeDictionary needs to be formatted in this way:
        # A dictionary which stores unique numbers starting from 0 with increments of 1, which corresponds to each node in the network. Note that if it does not start at 0 or if there are gaps, then it breaks, but the number does not mean anything.
        # Each key in this dictionary needs to store another dictionary which contains attributes for the corresponding node
        #
        # The current defined attributes are:
        # "Purpose" : str -> indicates the purpose of the node, used for defining specific input/output nodes
        # "Input" : bool -> indicates whether the node is an input node or not
        # "Output" : bool -> indicates whether the node is an output or not
        # "Bias" : float -> assigns a bias to the node
        # "ActivationFunction" : activation function -> assigns one of the activation functions defined in Neural_Network to the node
        # x : int -> any integer number that corresponds to a node, (note that the above restrictions apply)
        #
        # Note that the specific names for purposes have not been defined (as of writing this), and so error correction may need to be implemented if an invalid purpose is inserted

        #These attributes indicate which data needs to be collected
        self.requiresInputOfFood = False
        self.requiresInputOfCreature = False
        self.requiresInputOfWall = False
        
        self.requiresInputOfConstant = False
        self.requiresInputOfRandom = False

        #These are the names of the different input purposes that a neuron can have
        self.__unusedInputPurposeList = [
        "Closest Plant Proximity",
        "Closest Plant Angle",

        "Closest Meat Proximity",
        "Closest Meat Angle",

        "Closest Creature Proximity",
        "Closest Creature Angle",

        "Closest Wall Proximity",

        "Constant",
        "Random"
        ]

        #These are the different output purposes a node can have
        #This has to be stored as a dictionary as the activation functions of each purpose is fixed
        self.__unusedOutputPurposeDict = {
            "Speed":Neural_Network.TanH,
            "Angle Change":Neural_Network.TanH,
            "Biting State":Neural_Network.BinaryStep
        }

        self.nodes:list[Node] = []
        self.inputNodes:list[Node] = []
        self.outputNodes:list[Node] = []

        self.__createNetworkFromDictionary(nodeDictionary)

    def __createNetworkFromDictionary(self, nodeDictionary): #This method creates empty node objects, and assigns them to allNodes/inputNodes/outputNodes
        for i in range(len(nodeDictionary)):
            self.nodes.append(Node())
            if "Input" in nodeDictionary[i]:
                self.inputNodes.append(self.nodes[i])

            elif "Output" in nodeDictionary[i]:
                self.outputNodes.append(self.nodes[i])

        #Now assigns all the weights/biases to the nodes
        for i in nodeDictionary.keys():
            nodeAttributes = nodeDictionary[i] #This should be a dict containing the attributes
            for key in nodeAttributes:
                #loops through all the attributes and assigns them to corresponding node

                if key == "Input": #Updates the input state if it is an input
                    self.nodes[i].input = True
                    
                elif key == "Output": #Skips if it is an output
                    continue

                elif key == "Bias": #Assigns the bias if it is the bias
                    self.nodes[i].bias = nodeAttributes[key]

                elif key == "Activation Function": #Assigns the activation function to the node
                    self.nodes[i].activationFunction = Neural_Network.stringToActivationFunction[nodeAttributes[key]]

                elif key == "Purpose": #Assigns the purpose to the node
                    self.nodes[i].purpose = nodeAttributes[key]
                    #Removes the purpose from the unused list as it is now used
                    if nodeAttributes[key] in self.__unusedInputPurposeList:
                        self.__updateNetworkRequirements(self.nodes[i].purpose)
                        self.__unusedInputPurposeList.remove(nodeAttributes[key])
                    elif nodeAttributes[key] in self.__unusedOutputPurposeDict:
                        self.nodes[i].activationFunction = self.__unusedOutputPurposeDict[nodeAttributes[key]] #Updates the node's activation function to be the required one
                        self.__unusedOutputPurposeDict.pop(nodeAttributes[key])


                elif isinstance(key, int): #Assigns the values to the node.weights dictionary if corresponds to a weight (ensures that the weight is a float)
                    self.nodes[i].weights[self.nodes[key]] = float(nodeAttributes[key])

                else: #If there is a key which is not defined, it raises an exception
                    raise Exception(f"Neural Network blueprint is invalid, the following key has undefined behaviour: {key}")

    def __updateNetworkRequirements(self, purpose:str): #This method updates the values that network states it requires, which takes an input parameter of a node's purpose
        if purpose == "Closest Plant Proximity" or purpose == "Closest Plant Angle" or purpose == "Closest Meat Proximity" or purpose == "Closest Meat Angle": #if it requires any type of food as an input
            self.requiresInputOfFood = True

        elif purpose == "Closest Creature Proximity" or purpose == "Closest Creature Angle":
            self.requiresInputOfCreature = True

        elif purpose == "Closest Wall Proximity" or purpose == "Closest Wall Angle":
            self.requiresInputOfWall = True

        elif purpose == "Constant":
            self.requiresInputOfConstant = True

        elif purpose == "Random":
            self.requiresInputOfRandom = True

        else:
            raise ValueError(f"Invalid node purpose of {purpose} was attempted to be added to a Neural Network.")

    def __str__(self): #This method returns a string of the neural network, which will instantiate a copy if eval(network.repr()) is run
        nodeToNumberDict = {} #This is a dictionary which converts node objects into numbers
        nodeBlueprint = {} #This is a dictionary which represents the structure of the neural network

        #This assigns numbers to all the nodes
        for i,node in enumerate(self.nodes):
            nodeToNumberDict[node] = i

        #key is a node object
        for key in nodeToNumberDict:
            nodeBlueprint[nodeToNumberDict[key]] = {} #Assigns an empty dictionary to each node
            
            if key.input == True:
                nodeBlueprint[nodeToNumberDict[key]]["Input"] = True
            if key in self.outputNodes:
                nodeBlueprint[nodeToNumberDict[key]]["Output"] = True
            
            if key.activationFunction != Node.defaultActivationFunction:
                nodeBlueprint[nodeToNumberDict[key]]["Activation Function"] = key.activationFunction.__name__ #The __name__ magic method just gets a string representation of the name of the method

            if key.purpose != "Perceptron":
                nodeBlueprint[nodeToNumberDict[key]]["Purpose"] = key.purpose
            
            if key.bias != 0.0:
                nodeBlueprint[nodeToNumberDict[key]]["Bias"] = key.bias
            
            #weight is a node object (which are the connections)
            for weight in key.weights: #loops through all weights in the node and assigns it
                nodeBlueprint[nodeToNumberDict[key]][nodeToNumberDict[weight]] = key.weights[weight]

        return str(nodeBlueprint)
    
    @classmethod
    def __getMutationAmounts(cls, mutationAmount:int, mutationProbability:float) -> tuple[int,int,int,int,int,int]: #This method returns the number of mutations for each attribute
        
        #This generates the total number of mutations to happen
        totalMutationAmount = np.random.binomial(mutationAmount, mutationProbability)
        
        amountOfUpdatedConnections = 0
        amountOfUpdatedBiases = 0

        amountOfNewConnections = 0
        amountOfRemovedConnections = 0

        amountOfNewNodes = 0
        amountOfRemovedNodes = 0

        total = sum(cls.mutationRatios.values()) #This gathers the total value of the probability ratios

        for i in range(totalMutationAmount):
            randomNumber = random.randint(1,total)
            for key in Neural_Network.mutationRatios: #loops through each possible mutation and finds which the random number fell on
                randomNumber -= Neural_Network.mutationRatios[key]
                if randomNumber > 0: #If it hasn't reached the key it has been assigned to, then it continues
                    continue

                #If it has fallen on the key, then it updates the value accordingly
                if key == "Update Existing Weight":
                    amountOfUpdatedConnections += 1
                
                elif key == "Update Existing Bias":
                    amountOfUpdatedBiases += 1
                
                elif key == "Add Connection":
                    amountOfNewConnections += 1

                elif key == "Remove Connection":
                    amountOfRemovedConnections += 1

                elif key == "Add Node":
                    amountOfNewNodes += 1

                elif key == "Remove Node":
                    amountOfRemovedNodes += 1

                break #stops the loop once the key has been assigned

        #Returns the number of mutations for each variable
        return amountOfUpdatedConnections, amountOfUpdatedBiases, amountOfNewConnections, amountOfRemovedConnections, amountOfNewNodes, amountOfRemovedNodes

    def returnMutatedNetwork(self, mutationAmount:int, mutationProbability:float, mutationStandardDeviation:float) -> str:
        #This creates a new network with the same arrangement as itself
        newNeural_Network = Neural_Network(self.__nodeBlueprint)

        #Gathers the number of attributes to update
        amountOfUpdatedConnections, amountOfUpdatedBiases, amountOfNewConnections, amountOfRemovedConnections, amountOfNewNodes, amountOfRemovedNodes = Neural_Network.__getMutationAmounts(mutationAmount, mutationProbability)

        #Calls methods to update the network

        if amountOfNewNodes > 0: #adds nodes
            newNeural_Network.__addRandomNode(amountOfNewNodes)

        if amountOfRemovedNodes > 0: #removes nodes
            newNeural_Network.__removeRandomNode(amountOfRemovedNodes)

        if amountOfNewConnections > 0: #adds connections (weights)
            newNeural_Network.__addRandomConnection(amountOfNewConnections)

        if amountOfRemovedConnections > 0: #removes connections
            newNeural_Network.__removeRandomConnection(amountOfRemovedConnections)

        if amountOfUpdatedConnections > 0: #updates existing connections
            newNeural_Network.__updateRandomConnection(amountOfUpdatedConnections, mutationStandardDeviation)

        if amountOfUpdatedBiases > 0:
            newNeural_Network.__updateRandomBias(amountOfUpdatedBiases, mutationStandardDeviation)

        #removes any unused nodes
        newNeural_Network.__pruneNodes()

        #returns the string version of the new network
        return str(newNeural_Network)

    def __addNewInputNode(self): #This method adds a new input node from the unused list
        newNode = Node()
        newNode.input = True #Marks the node as an input node

        self.nodes.append(newNode)
        self.inputNodes.append(newNode)

        randomNode = self.nodes[random.randint(0,len(self.nodes)-1)]
        if randomNode != newNode and randomNode.input == False:
            randomNode.weights[newNode] = Neural_Network.defaultWeight * (-1)**(random.randint(0,1))

        #This removes a random purpose from the list of unused purposes, and assigns it to the new neuron
        randomPurpose = self.__unusedInputPurposeList[random.randint(0, len(self.__unusedInputPurposeList)-1)]
        newNode.purpose = randomPurpose
        self.__unusedInputPurposeList.remove(randomPurpose)

    def __addNewOutputNode(self): #This method adds a new output node from the unused dictionary
        newNode = Node()

        self.nodes.append(newNode)
        self.outputNodes.append(newNode)

        randomNode = self.nodes[random.randint(0,len(self.nodes)-1)]
        if randomNode != newNode:
            newNode.weights[randomNode] = Neural_Network.defaultWeight * (-1)**(random.randint(0,1))

        #This removes a random purpose from the list of unused purposes, and assigns it to the new neuron
        randomIndex = random.randint(0, len(self.__unusedOutputPurposeDict)-1) #This gets a random index which is used to select a purpose
        randomPurpose = list(self.__unusedOutputPurposeDict.keys())[randomIndex] #This gets a random purpose from the usassigned list
        newNode.activationFunction = self.__unusedOutputPurposeDict[randomPurpose] #This assigns the corresponding activation function to the node
        newNode.purpose = randomPurpose #This assigns the purpose to the node
        self.__unusedOutputPurposeDict.pop(randomPurpose) #This removes the purpose from the unused dictionary, as it is now used

    def __addRandomNode(self, amount:int) -> None: #This method inserts a new node into the neural network, with a random activation function

        if len(self.__unusedInputPurposeList) > 0: #If the network still has unused input nodes, then it generates a random number to see if it should create one
            randomNumber = random.uniform(0.0,1.0)
            if randomNumber <= Neural_Network.inputNeuronChance: #Creates an input neuron if the random number was low enough
                self.__addNewInputNode()
                amount -= 1
        
        if amount > 0 and len(self.__unusedOutputPurposeDict) > 0: #Does the same but for the output neuron
            randomNumber = random.uniform(0.0,1.0)
            if randomNumber <= Neural_Network.outputNeuronChance:
                self.__addNewOutputNode()
                amount -= 1

        inserting = False
        if amount > 0:
            inserting = True
        while inserting:
            #Order of getValues is randomEnd -> newNode -> randomBeginning
            if len(self.nodes) < 2: #If the network contains less than two nodes, then it creates one with no connections
                newNode = Node()
                self.nodes.append(newNode) #This adds the node to the list of all nodes, to ensure the removeSpecificNode method works
                newNode.activationFunction = random.choice(Neural_Network.activationFunctions)
                amount -= 1

            else:
                randomBeginning, randomEnd = self.__getTwoRandomNodes()
                newNode = Node()
                self.nodes.append(newNode) #This adds the node to the list of all nodes, to ensure the removeSpecificNode method works
                if randomBeginning not in self.outputNodes: #Only gathers values from nodes which are not output nodes
                    newNode.weights[randomBeginning] = Neural_Network.defaultWeight * (-1)**(random.randint(0,1)) #This assigns a node for the current node to get values from
                if not randomEnd.input: #Only assigns the end weight if the selected node is not an input
                    randomEnd.weights[newNode] = Neural_Network.defaultWeight * (-1)**(random.randint(0,1)) #This assings a node which gathers values from the current one

                #If the network is valid after inserting, then it continues
                if randomEnd.checkForInfiniteRecursion(): #Checks to ensure that the adding of the node is valid
                    activationFunction = random.choice(Neural_Network.activationFunctions)
                    if activationFunction != Node.defaultActivationFunction: #If it is the default activation function, then it does not need to be stored
                        newNode.activationFunction = activationFunction #Selects a random activation function
                    amount -= 1
                else: #If the network is not valid, then it removes the node and tries again
                    self.nodes.remove(newNode)
                    del randomEnd.weights[newNode]

            if amount <= 0: #Once all nodes have been inserted, it stops the loop
                inserting = False
        
        return

    def __removeRandomNode(self, amount:int): #Deletes a specified amount of nodes from the network
        while amount > 0:
            if len(self.nodes) == 0: #If it attempts to remove a node that does not exist, then it stops
                return
            nodeToBeRemoved = self.nodes[random.randint(0, len(self.nodes)-1)]
            self.__removeSpecificNode(nodeToBeRemoved) #removes the node
            amount -= 1

        return

    def __addRandomConnection(self, amount:int): #This method should add a connection between 2 nodes (should ensure that nodes cannot be connected backwards)

        #This ensures that hidden nodes exist to form connections
        if len(self.nodes) == len(self.inputNodes):
            return
        
        if len(self.nodes) < 2: #If there are less than two nodes, then a connection cannot be formed
            return

        while amount > 0:

            #This block ensures that an input node cannot be assigned connections
            endIsInput = True
            while endIsInput:
                randomBeginning, randomEnd = self.__getTwoRandomNodes()
                endIsInput = randomEnd.input
            
            if (randomBeginning not in randomEnd.weights.values()) and (randomBeginning not in self.outputNodes): #If it attempts to add a connection when one already exists, then it doesn't override it. Also performs additional checks if the neuron is an output node
                randomEnd.weights[randomBeginning] = Neural_Network.defaultWeight * (-1)**(random.randint(0,1)) #It randomly chooses whether it is a negative or positive weight

            if randomEnd.checkForInfiniteRecursion(): #Checks if the new connection is valid by ensuring that starting from the end node does not lead to any infinite recursions
                amount -= 1

            else: #Deletes the connection if it is not valid and retries
                del randomEnd.weights[randomBeginning]

        return
  
    def __removeRandomConnection(self, amount:int): #This method should delete a specified amount of random connections
        connectionAmount = 0
        for node in self.nodes:
            connectionAmount += len(node.weights)
        
        amount = min(amount, connectionAmount) #ensures that it cannot remove more connections than what exists

        while amount > 0:
            randomNode = self.nodes[random.randint(0, len(self.nodes)-1)] #Selects a random node
            
            if len(randomNode.weights) == 0: #skips the node if it does not have any connections
                continue
            
            randomConnection = random.choice(list(randomNode.weights.keys())) #Gathers a random connection from the selected node
            del randomNode.weights[randomConnection]
            amount -= 1
        
        return

    def __updateRandomConnection(self, amount:int, mutationStandardDeviation:float): #This method should update the connections between 2 nodes (if the newWeight is 0 then the connection should be removed)
        connectionAmount = 0
        for node in self.nodes:
            connectionAmount += len(node.weights)
        
        amount = min(amount, connectionAmount) #ensures that it cannot update more connections than what exists
        

        while amount > 0:
            randomNode = self.nodes[random.randint(0, len(self.nodes)-1)] #Selects a random node
            
            if len(randomNode.weights) == 0: #skips the node if it does not have any connections
                continue

            randomConnection = random.choice(list(randomNode.weights.keys())) #Gathers a random connection from the selected node
            
            value = randomNode.weights[randomConnection] #gathers the value
            newValue = float(np.random.normal(value, mutationStandardDeviation, size=1)[0]) #generates a new value from the normal distribution
            randomNode.weights[randomConnection] = newValue #updates the weight

            amount -= 1
        
        return

    def __updateRandomBias(self, amount:int, mutationStandardDeviation:float): #This method updates the value of the bias on nodes that are not input nodes
        if len(self.nodes) == 0: #If there are no nodes to update, then it returns
            return
        if len(self.inputNodes) +  len(self.outputNodes) == len(self.nodes): #If there are only input or output nodes, then it returns
            return
        
        while amount > 0:
            randomNode = self.nodes[random.randint(0, len(self.nodes)-1)]
            if randomNode.input or randomNode in self.outputNodes: #This skips the node if it is an input
                amount -= 1
                continue
            value = randomNode.bias #gathers the value
            newValue = float(np.random.normal(value, mutationStandardDeviation, size=1)[0]) #generates a new value from the normal distribution
            randomNode.bias = newValue
            amount -= 1
        
        return

    def __pruneNodes(self): #This method removes nodes which have no connections in either direction
        #If it has no connections, and if no other nodes connect to it, and it is not an input node, then it can be removed
        if len(self.nodes) < 3: #Does not remove any nodes if there are less than three, to ensure networks do not get stuck with no nodes
            return
        nodesToBeDeleted:list[Node] = list()
        for node in self.nodes: #loops through all nodes
            found = False

            if node.input: #skips all input nodes
                continue
            if node in self.outputNodes: #skips all output nodes
                continue

            for secondNode in self.nodes: #checks for references in other nodes
                if node == secondNode: #skips the node if it is itself
                    continue

                if node in secondNode.weights: #if the node is its references
                    found = True
                    break

            if found == True: #If the neuron was found to be connected to another node, then it skips it
                continue

            else: #if it has not found any connections, then it adds it to the list of nodes to be removed
                if len(node.weights) == 0:
                    nodesToBeDeleted.append(node)

        for node in nodesToBeDeleted: #deletes the unconnected nodes
            self.nodes.remove(node)
            
        return

    def __getTwoRandomNodes(self): #This method returns two unique nodes, used for adding connections
        if len(self.nodes) < 2:
            raise Exception("Can't gather two unique nodes from network as there are less than two nodes")
        else:
            indexes = np.random.choice(range(0, len(self.nodes)), size=2, replace=False)
            return self.nodes[indexes[0]], self.nodes[indexes[1]]

    def __removeSpecificNode(self, nodeToBeRemoved): #Removes a specific node from the network
        for node in self.nodes: #loops through each node
            for weight in node.weights: #loops through each weight in each node
                if weight == nodeToBeRemoved:
                    del node.weights[weight] #deletes the connecton if found
                    break

        #now all connections should have been deleted, so it deletes the node
        #Deletes it from the input/output lists if it is an input/output node
        #Also reassigns the purpose to the unused list if it removed an input/output node
        if nodeToBeRemoved.input:
            self.__unusedInputPurposeList.append(nodeToBeRemoved.purpose)
            self.inputNodes.remove(nodeToBeRemoved)
        elif nodeToBeRemoved in self.outputNodes:
            self.__unusedOutputPurposeDict[nodeToBeRemoved.purpose] = nodeToBeRemoved.activationFunction 
            self.outputNodes.remove(nodeToBeRemoved)

        self.nodes.remove(nodeToBeRemoved)

        return

    def forward(self): #This method should assign each node in self.outputNodes the correct values, given the input values
        for n in self.nodes: #shows that values haven't been updated yet
            #n.value = 0
            n.updated = False
        for n in self.inputNodes: #assigns the correct inputs to the input neurons
            n.updated = True
        for n in self.outputNodes: #collects the relevant values for each output node
            n.value = n.getValues()

    @property
    def isValidConfiguration(self) -> bool: #This returns a True/False depending on whether the neural network is a valid configuration
        for nodeToBeChecked in self.nodes:
            for node in self.nodes: #Resets all the 'visited' values to False
                node.updated = False

            if nodeToBeChecked.checkForInfiniteRecursion() == False: #If an infinite recursion has occurred
                return False
        
        #If it passes through all the nodes without an infinite recursion occurring, it is valid and so returns true
        return True

    @staticmethod
    def Identity(x:float) -> float:
        return x

    @staticmethod
    def Sigmoid(x:float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def TanH(x:float) -> float:
        return math.tanh(x)
    
    @staticmethod
    def ReLU(x:float) -> float:
        return max(0,x)
    
    @staticmethod
    def BinaryStep(x:float) -> float:
        if x >= 0:
            return 1.0
        else:
            return 0.0

    #Below are class attributes that require methods from the class, so they need to be stated after the functions have been defined

    #This stores a list of all the activation functions, used for when needing to randomly pick one
    activationFunctions = [Identity.__func__, Sigmoid.__func__, TanH.__func__, ReLU.__func__, BinaryStep.__func__]
    
    #The reason why __func__ has to be used is because without it, it stores the function with the wrapper, but the wrapper is not accessed by other classes and so causes issues when trying to make comparisons

    #The reason why I have functions to convert between strings is because every time the program is run, the memory locations of the functions change and so it will not be compatible with saving/loading
    stringToActivationFunction = {"Identity":Identity.__func__, "Sigmoid":Sigmoid.__func__, "TanH":TanH.__func__, "ReLU":ReLU.__func__, "BinaryStep":BinaryStep.__func__}

class Node: #This represents a single node in the neural network

    #The default activation function in this case is the identity function
    defaultActivationFunction = Neural_Network.Identity

    def __init__(self, purpose="Perceptron", input=False, activationFunction = defaultActivationFunction):
        self.updated = False #stores whether a node has already been updated on a current feed-forward

        self.value = 0.0 #stores the value contained in a node
        self.bias = 0.0 #stores the bias
        self.activationFunction = activationFunction #stores the activation function which is applied once all the values have been connected

        self.input = input #stores whether a node is an input node or not
        self.weights = {} #This should store the weights of the previous nodes connected to it

        self.purpose = purpose #Purpose should only be different for input/output nodes

    def getValues(self) -> float: # A recursive method which will go through the neural network
        if self.updated == True: #if it has already had its value updated (including input nodes)
            return self.value
        
        elif len(self.weights) == 0: #if it has no further connections but is not an input neuron
            #This should only happen on nodes with no inputs that are not the input nodes themselves
            self.value = self.activationFunction(self.bias) #sets the value to the bias to the node if not an input
            self.updated = True
            return self.value

        else: #assigns the new value as the sum of the children * weights
            self.value = 0.0 #Resets the initial value to 0 before adding all the weights
            for node in self.weights: #loops through each child
                self.value += node.getValues()*self.weights[node]
            self.value += self.bias #adds the bias
            self.value = self.activationFunction(self.value) #then applies the activation function to the node
            self.updated = True
            return self.value

    def checkForInfiniteRecursion(self) -> bool: #This method returns True if the node does not cause an infinite recursion to happen, and True if it does
        if self.updated == True: #if this node has already been visited then
            self.updated = False #resets the state
            return False
        else: #if the node has not been visited, then check all connections before it
            self.updated = True
            for node in self.weights:
                if node.checkForInfiniteRecursion() == False: #if an infinite recursion has occurred
                    self.updated = False #resets the state
                    return False

            #If all nodes before it work, then mark it as unvisited in case of branch splitting
            self.updated = False
                
            #if it passes through all connecting nodes without a collision or if there are no connections, there is no infinite recursion on this iteration
            return True


if __name__ == "__main__":
    run = True

    #This is a complicated, valid neural network
    n = Neural_Network('''{0:{"Input":True, "Purpose":"Constant"}, 1:{"Input":True, "Purpose":"AngleToClosestFood"}, 2:{0:0.5}, 3:{1:0.5}, 4:{0:0.5}, 5:{"Output":True, 3:0.5, 2:0.5}, 6:{"Output":True, 4:0.5,0:0.5,1:0.5}}''')

    #n2 is a clearly invalid network (3 -> 2 -> 1 -> 2 -> 1 -> ...)
    n2 = Neural_Network('''{0:{"Input":True}, 1:{0:1, 2:1}, 2:{1:1}, 3:{"Output":True, 2:1}}''')

    #n3 is a clearly valid network: (3 -> 2 -> 1 -> 0)
    n3 = Neural_Network('''{0:{"Input":True}, 1:{0:1}, 2:{1:1}, 3:{"Output":True, 2:1}}''')


    while run: #This while loop continually mutates the network and prints it in the terminal

        n3 = Neural_Network(n3.returnMutatedNetwork(1,1,1))
        print(n3)
        
        #Other characteristics which you may want to view
        #print(len(n3.nodes))
        #print(len(n3.inputNodes))

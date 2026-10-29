This folder contains my (Melvin Page's) evolution simulator project.

== SAMPLE INFORMATION ==
This sample does not contain all the source code files for my project, however it does show the neural network file, which is an important part of the project.
- Neural_Network.py : 

Showcases a custom neural network architecture which is used by creatures within the simulation, where there are no set 'layers' but rather nodes recursively get their values by taking the sum of each connected node and applying an activation function.
There are a variety of hardcoded names for inputs and outputs that this neural network can take, which include direction/proximity to nearest food and other creatures.
Each node can be assigned a unique activation function from a predetermined list.
Some neural networks will have an invalid configuration, so methods are implemented which return whether a network is valid.
Output neurons are forced to have a preset activation function to ensure the output of a specific node from the network is what the user intends it to be.

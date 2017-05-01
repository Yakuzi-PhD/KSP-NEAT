# KSP-NEAT
>>>>> Neuro-Evolution of Augmenting Topologies in Evolving a Return Launch Vehicle in KSP <<<<<

This program uses a genetic algorithm to evolve a neural network in order to land the first stage of a rocket at its original launch site in Kerbal Space Program.
This program is based on the the final approach of the SpaceX Falcon 9 return mission and exhibits similar behaviour (convergent evolution). 
The AI included has been trained over 100 generations encompassing +/- 2500 organisms that have been selected to reproduce based on their global fitness. An average of 25 organisms
were evaluated for their respective fitness each generation. The 5 fittest organisms were selected to seed the next generation, and served as parents/templates for the remaining 
20 organisms through cross-over and point mutations of their neural networks (i.e. adjusting weigths and adding/removing nodes and edges).

Organism fitness is calculated based on:
- Integrity of the rocket
- Distance to target coordinates
- Landing speed
- Vessel pitch when landed
- Remaining fuel


>>>>> Framework and Implementation <<<<<

Optimizing artificial neural networks through genetic algorithms is an alternative to commonly used neural net optimization methods (e.g. backpropagation), particularly in the absence of labeled data. 
Here, the custom coded NEAT_final_approach program uses the kRPC remote procedure calling server framework to communicate with KSP and the neat_python framework in order to generate and evolve populations of rockets with 
artificial neural network genomes. Populations are defined in the main config file, which also specifies the mutation types and rates, as well as the input and output layer for the neural network genome. Upon generating a population,
individual organisms(vessels) are loaded in the game environment and selected vessel/flight telemetry data is used as input for the organism's neural net. Signals from the output layer get processed and communicated back to KSP
where they directly influence the organism's behaviour. As soon as an (positive or negative) end condition is triggered, the overall fitness of the organism is calculated based on its performance in the fitness landscape 
(potentially involving multipliers/penalties). Subsequently the next organism is loaded for fitness evaluation. After the last organism of a generation has been evaluated, the fittest members of the population are copied to 
a new generation, where they produce the rest of the populations neural networks through point and cross-over mutations. After the specified number of generations has been achieved or the maximum fitness criterium declared in 
the config file has been met, the simulation will come to a halt and the fittest organism of the last generation will be declared winner and organism stats, fitness history and species history will be displayed 
using neat_python's visualize module (WIP).


>>>>> Requirements <<<<<

Python 3.6.1
Kerbal Space Program 1.2.2


>>>>> Installation Instructions <<<<<

1. Download and install Kerbal Space Program version 1.2.2(x64). It can be purchased at:
	https://kerbalspaceprogram.com/kspstore/
2. Copy the contents of the "Put in KSP folder" in the root KSP_1.2.2 folder (this includes the kRPC mod by djungelorm, the KSP_NEAT folder, neat-python by CodeReclaimers and a KSP save game file).
3. Run the game. At the startup screen press "Start Game", then "Resume Saved". Select and load "test".
4. At the spacecenter screen, select the the "NEAT_Test_02BX" vessel on the launchpad and click "Fly".
5. At the launchpad, press the kRPC network/server-like symbol on the right-hand side of the screen, then press "start server".
6. With the game running, open the "NEAT_final_approach" python file in the "KSP_NEAT" folder in the root KSP directory with Python's IDLE, then press "F5" to start the evolution procedure (default 100 generations).
7. Copy generation save state files from the "Checkpoints" folder to the "KSP_NEAT" folder to observe emergent behaviour of the organisms by means of the evolutionary selection process.


>>>>> Known Issues <<<<<

There is a memory leak in KSP upon loading saved game states, so after a couple of generations the game may become sluggish and eventually your computer will run out of memory.
Accordingly, restart the game every so often, the program will resume the evolution precess from the last saved generation (default every 1 generation).


Enjoy!

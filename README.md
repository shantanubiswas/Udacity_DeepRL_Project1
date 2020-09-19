[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Udacity Deep Reinforcement Learning Nanodegree::Project 1: Navigation

### Introduction

This project repository contains Shantanu Biswas’s work for the Udacity's Deep Reinforcement Learning Nanodegree Project 1: Navigation. For this project, I have trained an agent to navigate and collect yellow bananas in a large, square world.

### Project's goal
 - The environment is square world filled with yellow and blue bananas. 
 - A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
 - The goal is to pick up the yellow bananas and avoid the blue ones. 
 - The task is episodic, and the end goal is to reach an average score of +13 over 100 consecutive episodes.


### Project's Environment

For this project, I have trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Project Environment Setup/ Configuring Dependencies:

### Step 1: Clone the DRLND Repository and installing dependencies
#### 1. Set Up for the  python environment to run the code in this repository
#### Environment
 - conda create --name drlnd python=3.6
 - source activate drlnd

#### 2. Install of OpenAI gym packages
#### OpenAI gym
 - git clone https://github.com/openai/gym.git
 - cd gym
 - pip install -e .
 - pip install -e .[classic_control]
 - conda install swig
 - pip install -e .[box2d]

#### 3.  Other Dependencies
 - git clone https://github.com/udacity/deep-reinforcement-learning.git
 - cd deep-reinforcement-learning/python
 - pip install .

NOTE: List of the installed dependencies can be found in “requirements.txt” attached with the project.

### Trouble shooting:

If you get the error for installing torch library:
On Windows:
1.	Open deep-reinforcement-learning\python\requirements.txt
2.	Remove the line torch==0.4.0
On Anaconda:
3.	conda install pytorch=0.4.0 -c pytorch
4.	cd deep-reinforcement-learning/python
pip install .
conda install pytorch=0.4.0 -c pytorch


#### 4. Create an IPython kernel for the drlnd environment
#### Kernel
python -m ipykernel install --user --name drlnd --display-name "drlnd"

#### 5. Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.


### Step 2: Download the Unity Environment

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 


### Description
 - dqn_agent.py: code for the agent used in the environment
 - model.py: code containing the Q-Network used as the function approximator by the agent
 - Navigation_EnvironmentExploration.ipynb: explore the unity environment and verify the setup
 - Navigation_Solution.ipynb: notebook containing the solution (Each cell to be executed to train the agent)
 - Navigation_Pixels.ipynb: notebook containing the code for the pixel-action problem (optional)
 - dqn.pth: saved model weights for the original DQN model
 - double_dqn.pth: saved model weights for the Double DQN model
 - Duelling_dqn.pth: saved model weights for the Dueling DQN model
 - Report.md: The submission includes a file in the root of the GitHub repository that provides a description of the implementation.



### Steps to Run the Project

1. Ensure that the Banana.exe is at the same level as the Navigation.ipynb
2. Use jupyter to run the Navigation.ipynb notebook: jupyter notebook Navigation.ipynb
3. Follow the instructions in Navigation.ipynb to get started with training your own agent.
4. To train the agent run the cells in order. They will initialize the environment and train until it reaches the goal condition of +13.
5. A graph of the scores during training will be displayed after training.
6. To watch a trained smart agent, follow the instructions below:
 - DQN: If you want to run the original DQN algorithm, use the checkpoint dqn.pth for loading the trained model. Also, choose the parameter qnetwork as QNetwork while defining the agent and the parameter update_type as dqn.
 - Double DQN: If you want to run the Double DQN algorithm, use the checkpoint ddqn.pth for loading the trained model. Also, choose the parameter qnetwork as QNetwork while defining the agent and the parameter update_type as double_dqn.
 - Dueling Double DQN: If you want to run the Dueling Double DQN algorithm, use the checkpoint dddqn.pth for loading the trained model. Also, choose the parameter qnetwork as DuelingQNetwork while defining the agent and the parameter update_type as double_dqn.


###  Results
Plot showing the score per episode over all the episodes.
The best performance was achieved by Double DQN where the reward of +13 was achieved in 355 episodes. 

| Double DQN                                 | DQN                                | Dueling DQN                                         |
| ------------------------------------------ | ---------------------------------- | --------------------------------------------------- |
| ![double-dqn](results/double_dqn_new_scores.png) | ![dqn](results/dqn_new_scores.png) | ![dueling double dqn](results/duelling_dqn_new_scores.png) |


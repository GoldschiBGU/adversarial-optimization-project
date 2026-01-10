# **Adversarial Optimization Project**

**Heterogeneous Multi-Agent Reinforcement Learning with Alternating Policy Optimization**

## **📸 Visualization**

The framework optimizes the placement of defensive units against adaptive adversaries.


*Figure 1: Visualization of the simulation map including target trajectories.*

### **Simulation Demo**

Below is a demonstration of the agent performance at Level 6 (76.6% success rate).
[**Simulation Level 1 (100% IR)**](https://github.com/user-attachments/assets/65891db3-2c6a-4a80-9574-edf28b66ab63)

[**▶️ Watch Simulation Video (Level 6\)**](https://www.google.com/search?q=animations/Level%25206%252076.6%2525/simulation_level6_76.6.mp4)

## **🛠️ Installation**

1. Clone the repository:  
   git clone \[https://github.com/yourusername/adversarial-optimization-project.git\](https://github.com/yourusername/adversarial-optimization-project.git)  
   cd adversarial-optimization-project

2. Install the required dependencies:  
   pip install \-r requirements.txt

## **🚀 Usage Guide**

Follow this step-by-step workflow to recreate the results or train new models.

### **Step 1: Initialization**

Before training, define the environment and generate the initial baseline.

1. Design the Town Data:  
   Edit the town\_data.xlsx file located in the initialization folder to set up your environment topology.  
2. Generate Initial Placements:  
   Run the greedy algorithm to create the initial placements file.  
   python initialization/greedy\_algo.py

### **Step 2: Training**

Train the agents using the Alternating Policy Optimization method.

* **Train with HMARL-APO (Recommended):**  
  python train/HMARL-APO.py

* *Alternative: Train with standard HMARL:*  
  python train/HMARL.py

### **Step 3: Evaluation**

After training, evaluate the model to generate statistical results.

python train/evaluation.py

*Note: This will create a new folder named model\_{Number} containing the results. The {Number} is defined by the user during this step.*

### **Step 4: Results & Visualization**

Analyze the performance of your specific model number.

* **Print Performance Statistics:**  
  python "results analysis/print\_performance.py"

* Generate Pareto Plots:  
  Visualize the trade-off between resources and interception rates.  
  \# Single model plot  
  python "results analysis/model\_pareto\_plot.py"

  \# Compare against other models (Pareto Frontier)  
  python "results analysis/model\_final\_pareto\_plot.py"

## **📊 Key Results**

| Method | Interception Rate (Complex Scenario) |
| :---- | :---- |
| **HMARL-APO (Ours)** | **72.2%** |
| Baseline A | 47.2% |
| Baseline B | 28.7% |

## **📜 License**

Distributed under the MIT License. See LICENSE for more information.

## **📧 Contact**

Project Link: [https://github.com/yourusername/adversarial-optimization-project](https://www.google.com/search?q=https://github.com/yourusername/adversarial-optimization-project)

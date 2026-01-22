# **Adversarial Optimization Project**


## **📸 Visualization**

The framework optimizes the placement of defensive units against adaptive adversaries.

### **Simulation Demo**

**Click to watch simulations demo:**


**Level 1:**

[**▶️ Simulation Level 1 (100% IR)**](https://github.com/user-attachments/assets/65891db3-2c6a-4a80-9574-edf28b66ab63)

**Level 2:**

[**▶️ Simulation Level 2 (93.3% IR)**](https://github.com/user-attachments/assets/1fac046b-c077-44c5-a1b8-300ed320432c)

**Level 3:**

[**▶️ Simulation Level 3 (86.6% IR)**](https://github.com/user-attachments/assets/1fe6b889-f8bd-4364-ac43-88658f70966c)

**Level 4:**

[**▶️ Simulation Level 4 (73.3% IR)**](https://github.com/user-attachments/assets/8f7a5e72-4d71-4a68-8d80-8d6960546680)

**Level 5:**

[**▶️ Simulation Level 5 (63.3% IR)**](https://github.com/user-attachments/assets/0c770201-6e61-4e87-96f5-d5e1cc876a21)

**Level 6:**

[**▶️ Simulation Level 6 (76.6% IR)**](https://github.com/user-attachments/assets/5745607b-d3cb-4699-a1db-5b098e7fb758)

## **🛠️ Installation**

Install the required dependencies:  
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

Train the agents.

* **Train with HMRD-PPO-Simul (Recommended):**  
  python train/HMRD-PPO-Simul.py

* *Alternative: Train with Alternating version:*  
  python train/HMRD-PPO-Alt.py

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

## **📜 License**

Distributed under the MIT License. See LICENSE for more information.

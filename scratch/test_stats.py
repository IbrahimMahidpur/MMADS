import pandas as pd
from multimodal_ds.agents.statistical_agent import StatisticalReasoningAgent
import logging

logging.basicConfig(level=logging.INFO)

file_path = r'C:\Users\imahi\Downloads\multimodal_ds_production\multimodal_ds\data\brfss_sample_dataset.csv'
df = pd.read_csv(file_path)
print(f"Loaded DF: {df.shape}")

agent = StatisticalReasoningAgent(session_id="test")
print("Starting validation...")
report = agent.validate_dataset(df)
print("Validation finished!")
print(report['interpretation'])

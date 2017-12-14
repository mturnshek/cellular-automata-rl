from grow.dqn_agent import DQN_Agent

print("\n\n Training small model DQN agent... \n\n")
dqn_agent = DQN_Agent(model_type='small', display=False)
dqn_agent.create_model()
dqn_agent.train()

print("\n\n Training low-high-low model DQN agent... \n\n")
dqn_agent = DQN_Agent(model_type='low_high_low', display=False)
dqn_agent.create_model()
dqn_agent.train()

print("\n\n Training deep model DQN agent... \n\n")
dqn_agent = DQN_Agent(model_type='deep', display=False)
dqn_agent.create_model()
dqn_agent.train()

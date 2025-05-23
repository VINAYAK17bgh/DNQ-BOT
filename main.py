import numpy as np
from stock_trading_env import StockTradingEnv
from dqn_agent import DQNAgent
from data_loader import load_stock_data
from visualization import plot_results

def main():
    # Load and prepare data
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    print(f"Loading data for {symbol} from {start_date} to {end_date}...")
    df = load_stock_data(symbol, start_date, end_date)
    
    # Initialize environment
    print("Initializing trading environment...")
    env = StockTradingEnv(
        df=df,
        initial_balance=10000,
        lookback_window=60
    )
    
    # Initialize agent with TF 2.19 compatible parameters
    print("Initializing DQN agent...")
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        action_size=env.action_space.n,
        memory_size=10000,
        batch_size=64,
        gamma=0.95,
        learning_rate=0.001  # Changed from 'lr' to 'learning_rate'
    )
    
    # Training parameters
    episodes = 100
    update_target_every = 10
    
    print(f"Starting training for {episodes} episodes...")
    agent.train(
        env=env,
        episodes=episodes,
        update_target_every=update_target_every
    )
    
    # Evaluation
    print("Evaluating trained agent...")
    state = env.reset()
    done = False
    portfolio_values = []
    actions_history = []
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        # Record portfolio value and action
        current_price = df.iloc[env.current_step]['Close']
        portfolio_values.append(env.balance + (env.shares * current_price))
        actions_history.append(action)
        
        state = next_state
    
    # Calculate final metrics
    initial_balance = env.initial_balance
    final_value = portfolio_values[-1]
    returns = (final_value - initial_balance) / initial_balance * 100
    
    print("\n=== Training Results ===")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {returns:.2f}%")
    
    # Plot results
    plot_results(
        portfolio_values=portfolio_values,
        prices=df['Close'].values[-len(portfolio_values):],
        actions=actions_history
    )
    
    # Save trained model
    print("Saving trained model...")
    agent.model.save('models/dqn_trading_tf2.19.h5')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
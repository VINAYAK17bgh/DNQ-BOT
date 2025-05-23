import matplotlib.pyplot as plt

def plot_results(portfolio_values, prices):
    plt.figure(figsize=(12, 6))
    
    # Portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title('Trading Performance')
    plt.ylabel('Value ($)')
    plt.legend()
    
    # Price and actions
    plt.subplot(2, 1, 2)
    plt.plot(prices, label='Stock Price')
    plt.xlabel('Time Step')
    plt.ylabel('Price ($)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
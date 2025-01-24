import matplotlib.pyplot as plt

def plot_predictions(actual, predicted):
    plt.figure(figsize=(10,5))
    plt.plot(actual, label='Actual Values')
    plt.plot(predicted, label='Predicted Values', linestyle='dashed')
    plt.legend()
    plt.title("Actual vs Predicted Temperature")
    plt.show()

import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    # Lightweight, non-blocking plotting for scripts
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='score')
    plt.plot(mean_scores, label='mean_score')
    plt.ylim(ymin=0)
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.draw()
    plt.pause(0.001)

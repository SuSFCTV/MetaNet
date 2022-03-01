import seaborn as sns
import matplotlib.pyplot as plt


def plotting(losses):
    sns.set(style="whitegrid", font_scale=1.4)
    plt.figure(figsize=(12, 8))
    plt.plot(losses['train'], label="train")
    plt.plot(losses['val'], label="val")
    plt.legend()
    plt.savefig('results/result_resnet.png')

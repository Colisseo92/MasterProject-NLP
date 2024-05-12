import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot3dAccuracyPlot(x, y, z, title, numbers, values):
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=z, s=50)
    pop_a = mpatches.Patch(color='#531561', label='2 sentiments')
    pop_b = mpatches.Patch(color='#3B6F93', label='3 sentiments')
    pop_c = mpatches.Patch(color='#49BD86', label='4 sentiments')
    pop_d = mpatches.Patch(color='#FDE725', label='5 sentiments')
    ax.legend(handles=[pop_a, pop_b, pop_c, pop_d])
    ax.locator_params(axis='z', integer=True)
    ax.locator_params(axis='y', integer=True)
    plt.yticks(numbers, values,rotation=0,fontsize=4)
    ax.set_xlabel("accuracy")
    ax.set_zlabel("nombre de sentiments")
    plt.show()
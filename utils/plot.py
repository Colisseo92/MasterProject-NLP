import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot3dAccuracyPlot(x, y, z, title, numbersy, valuesy, x_label,z_label):
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=y, s=50)
    pop_a = mpatches.Patch(color='#531561', label='2 sentiments')
    pop_b = mpatches.Patch(color='#3B6F93', label='3 sentiments')
    pop_c = mpatches.Patch(color='#49BD86', label='4 sentiments')
    pop_d = mpatches.Patch(color='#FDE725', label='5 sentiments')
    ax.legend(handles=[pop_a, pop_b, pop_c, pop_d])
    ax.locator_params(axis='z', integer=True)
    ax.locator_params(axis='y', integer=True)
    plt.yticks(numbersy, valuesy, rotation=0, fontsize=4)
    ax.set_xlabel(x_label)
    ax.set_zlabel(z_label)
    plt.show()

if __name__ == "__main__":
    scores = [0.8211,0.6037614269999316,0.6623840071395619,0.734903017810898,0.8769273504853379,0.8411,0.7163134423495918,0.6234101005185004,0.5532635695787749,0.8752025334943629,0.7542,0.47005884882072324,0.2947526912834288,0.21645349963884924,0.6468397446487223,0.8109,0.6185063652528188,0.5789204596297903,0.545291827422793,0.8210621663282348,0.8046,0.8635587960282238,0.34281460593937674,0.2143385656149252,0.8656647141132041,0.7431,0.4652542404623762,0.4815727005170478,0.4999982382890263,0.6595952419371088,0.7619,0.6571428571428571,0.006661613123250612,0.003355178549407184,0.7792444976716313,0.7655000000000001,0.5162375829851558,0.37676539489631555,0.30032327396367353,0.7863443629952254]
    models = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7]
    score_type = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

    models_dict = {0: "LogisticRegression", 1: "SVM", 2: "KNeighborsClassifier", 3: "MLPClassifier",
                   4: "RandomForestClassifier", 5: "DecisionTreeClassifier", 6: "Naives Bayes / MultinomialNB",
                   7: "Naives Bayes / BernoulliNB"}
    score_dict = {0: "accuracy", 1: "precision", 2: "f1", 3: "recall", 4: "roc_auc"}

    #plot3dAccuracyPlot(scores,models,score_type,"test",list(models_dict.keys()),models_dict.values(),"precision","type de score")

    accuracy = []
    precision = []
    f1 = []
    recall = []
    roc = []
    modele = ["LogisticRegression", "SVM", "KNeighborsClassifier", "MLPClassifier", "RandomForestClassifier", "DecisionTreeClassifier", "Naives Bayes","Naives Bayes / MultinomialNB","Naives Bayes / BernoulliNB"]

    for i,j in enumerate(scores):
        if i%5 == 0:
            accuracy.append(j)
        if i%5 == 1:
            precision.append(j)
        if i%5 == 2:
            f1.append(j)
        if i%5 == 3:
            recall.append(j)
        if i%5 == 4:
            roc.append(j)






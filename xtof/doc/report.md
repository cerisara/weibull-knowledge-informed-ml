### stage 1

mar. 15 mars 2022 09:06:10 CET

Je train avec un seul objectif: obtenir un alignement equi-reparti.
Pour cela, je commence par diviser la TS en 100 segments egaux, puis
je calcule la moyenne des trames pour chaque segment, puis
je calcule la MSE loss entre chaque moyenne et le centroide associe a chaque segment
(ce centroide est aussi un parametre).

Il m'a fallu tuner le LR pour que ca converge, et j'obtiens le MSE loss suivant:

![](eq.png)

et j'affiche aussi le nombre maximum de trames projetees sur un segment avec Viterbi
(sachant que la TS d'origine fait 1000 points):

![](bin.png)

On obtient bien une projection assez bien equilibree.
Je peux passer au stage 2 = train le RUL pour voir si cette projection degenere, auquel cas
il faudra alterner RUL et MSE.



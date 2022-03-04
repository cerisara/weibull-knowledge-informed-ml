- le dataset IMS est composé de 2000 fichiers, chaque fichier = 1s de signal, intervalle entre les fichiers = 5 à 10 minutes, ou plus
- la methode de ce papier est de:
    - preprocess chaque fichier avec FFT + 20 bins
    - passe cela dans un MLP qui output le RUL (calculé à partir de la data du fichier dans son nom)
    - attention: il y a des choses bizarres:
        - elimination des "worst performing models" avec un critere de RMSE minimum sur train, dev ** et TEST ** !
        - presentation des results du "best performing model" == "best" sur le test ?!
        - le early stopping s'arrete tres tot: est-ce que le modele est bien calibre ?
        - pas de comparaison avec l'etat de l'art !!
        - les res (Fig. 14) n'ont pas l'air genial

dans paperswithcode, il n'y a qu'un papier sur IMS dataset, celui-ci, qui donne aussi des res sur FEMTO dataset.
Il y a 2 papiers sur FEMTO dataset.

l'autre papier fait du transfert learning (= train on one bearing, fnietune on another) de FEMTO a FEMTO...

pour FEMTO,
- chaque bearing a des mesures jusque 30,000 timesteps.
- l'eval se fait sur la prediction du RMSE(Health Index) + "score", ce qui permet de comparer avec la litterature


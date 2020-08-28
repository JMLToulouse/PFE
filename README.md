# Projet Fin d'études - Ismail ZNIBER

## 1er fichier Notebook : Automatic diagnosis of the 12-lead ECG using a deep neural network.
-----------


#### Resume:

* Ce notebook est une reprise en main d’un code déjà existant qu’on trouve sur le lien suivant : https://github.com/antonior92/automatic-ecg-diagnosis
* Le notebook contient deux grandes parties, la partie descriptive où je parle en français, contrairement à l’ancien article qui était publié en anglais, du jeu de données utilisé, le réseau neurone appliqué, les méthodes qu’ils ont servies afin d’avoir une bonne annotation, ensuite il y a la partie informatique, codé sous python et qui permet de donner des prédictions et de savoir l’efficacité du model ou du réseau neurone utilisé.  
* Le réseau neurones utilisé, est un DNN qui reste un peu compliqué et qui contient plusieurs layers. Les entrées de ce réseau était du type (827,4096,12) où 827 représente le nombre de patients, 4096 représente le nombre de données reçu de l’ECG et 12 représente le nombre de signaux.
Autrement dit, chaque patient nous fournis 12 signaux ECG avec une moyenne de 4096 données par signal. La matrice est de la forme suivante :

![alt tag](https://user-images.githubusercontent.com/70271267/91366796-01267700-e805-11ea-9597-ee3eb4401093.png)
![first matrice](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Chuge%20%5Cbegin%7Bpmatrix%7D%20%5Ba_%7B%281%2C1%29%7D%2Cb_%7B%281%2C1%29%7D%2Cc_%7B%281%2C1%29%7D%2Cd_%7B%281%2C1%29%7D%2Ce_%7B%281%2C1%29%7D%2Cf_%7B%281%2C1%29%7D%2Cg_%7B%281%2C1%29%7D%2Ch_%7B%281%2C1%29%7D%2Ci_%7B%281%2C1%29%7D%2Cj_%7B%281%2C1%29%7D%2Ck_%7B%281%2C1%29%7D%2Cl_%7B%281%2C1%29%7D%5D%2C%20%26%20%5Ccdots%20%26%20%2C%5Ba_%7B%281%2C4096%29%7D%2Cb_%7B%281%2C4096%29%7D%2Cc_%7B%281%2C4096%29%7D%2Cd_%7B%281%2C4096%29%7D%2Ce_%7B%281%2C4096%29%7D%2Cf_%7B%281%2C4096%29%7D%2Cg_%7B%281%2C4096%29%7D%2Ch_%7B%281%2C4096%29%7D%2Ci_%7B%281%2C4096%29%7D%2Cj_%7B%281%2C4096%29%7D%2Ck_%7B%281%2C4096%29%7D%2Cl_%7B%281%2C4096%29%7D%5D%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cvdots%20%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cvdots%20%5C%5C%20%5Ba_%7B%28827%2C1%29%7D%2Cb_%7B%28827%2C1%29%7D%2Cc_%7B%28827%2C1%29%7D%2Cd_%7B%28827%2C1%29%7D%2Ce_%7B%28827%2C1%29%7D%2Cf_%7B%28827%2C1%29%7D%2Cg_%7B%28827%2C1%29%7D%2Ch_%7B%28827%2C1%29%7D%2Ci_%7B%28827%2C1%29%7D%2Cj_%7B%28827%2C1%29%7D%2Ck_%7B%28827%2C1%29%7D%2Cl_%7B%28827%2C1%29%7D%5D%2C%20%26%20%5Ccdots%20%26%20%2C%5Ba_%7B%28827%2C4096%29%7D%2Cb_%7B%28827%2C4096%29%7D%2Cc_%7B%28827%2C4096%29%7D%2Cd_%7B%28827%2C4096%29%7D%2Ce_%7B%28827%2C4096%29%7D%2Cf_%7B%28827%2C4096%29%7D%2Cg_%7B%28827%2C4096%29%7D%2Ch_%7B%28827%2C4096%29%7D%2Ci_%7B%28827%2C4096%29%7D%2Cj_%7B%28827%2C4096%29%7D%2Ck_%7B%28827%2C4096%29%7D%2Cl_%7B%28827%2C4096%29%7D%5D%20%5Cend%7Bpmatrix%7D)

Tandis que les sorties sont du type (827,6) où 6 représente le nombre d’anomalies, ici on parle des maladies cardiaques (AF, RBBB, LBBB, 1dAvb, ST, SB), Chaque point de la matrice contient une probabilité entre 0 et 1, et peut être comprise comme la probabilité qu'une anomalie donnée soit présente. Cette matrice est de la forme :

![alt tag](https://user-images.githubusercontent.com/70271267/91367697-6da27580-e807-11ea-9338-3d7b15a04fd8.png)

* Dans la partie code je suis passé de _argparse_ aux classes, car la bibliothèques argparse n'était pas fonctionnelle sur jupyter Notebook (Voir partie code du notebook).
* Les résultats obtenus avec ce DNN dépassent les prédictions des médecins et des étudiants avec un f1 score qui dépasse 80% et des indices de spécificité supérieurs à 99% ce qui rend cette étude intéressante pour le monde de la médecine.



#### Requirements:

Ce code a été testé sur Python 3 avec Tensorflow == 1.15.2 et Keras == 2.2.4. Il n'a pas été mis à jour pour fonctionner avec Tensorflow 2.0 et supérieur. Voir la partie Requirements dans le notebook.

#### Scripts:

On a commencé par importer les bibliothèques nécessaires, ensuite on a codé le modèle qu’on va utiliser puis l’entrainer, après on a lancé les prédictions et on a fini par comparer les résultats avec les médecins et les étudiants.

## 2ème fichier Notebook : Application du package Ethik sur le jeu de données de Ribeiro
-----------




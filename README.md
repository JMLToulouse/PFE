# PFE

## 1er fichier Notebook : Automatic diagnosis of the 12-lead ECG using a deep neural network.
-----------


#### Résumé:

* Ce notebook est une reprise en main d’un code déjà existant qu’on trouve sur le lien suivant : https://github.com/antonior92/automatic-ecg-diagnosis
* Le notebook contient deux grandes parties, la partie descriptive où je parle en français _contrairement à l’ancien article qui était publié en anglais_ du jeu de données utilisé, le réseau neurone appliqué, les méthodes qu’ils ont servies afin d’avoir une bonne annotation, ensuite il y a la partie informatique, codé sous python et qui permet de donner des prédictions et de savoir l’efficacité du model ou du réseau neurone utilisé.  
* Le réseau neurones utilisé, est un DNN qui reste un peu compliqué et qui contient plusieurs layers. Les entrées de ce réseau était du type (827,4096,12) où 827 représente le nombre de patients, 4096 représente le nombre de données reçu de l’ECG et 12 représente le nombre de signaux.
Autrement dit, chaque patient nous fournis 12 signaux ECG avec une moyenne de 4096 données par signal. La matrice est de la forme suivante :

![alt tag](https://user-images.githubusercontent.com/70271267/91366796-01267700-e805-11ea-9597-ee3eb4401093.png)


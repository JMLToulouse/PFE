# Projet Fin d'études - Ismail ZNIBER

## 1er fichier Notebook : Automatic diagnosis of the 12-lead ECG using a deep neural network.
-----------


#### Resume:

* Ce notebook est une reprise en main d’un code déjà existant qu’on trouve sur le lien suivant : https://github.com/antonior92/automatic-ecg-diagnosis
* Le notebook contient deux grandes parties, la partie descriptive où je parle en français, contrairement à l’ancien article qui était publié en anglais, du jeu de données utilisé, le réseau neurone appliqué, les méthodes qu’ils ont servies afin d’avoir une bonne annotation, ensuite il y a la partie informatique, codé sous python et qui permet de donner des prédictions et de savoir l’efficacité du model ou du réseau neurone utilisé.  
* Le réseau neurones utilisé, est un DNN qui reste un peu compliqué et qui contient plusieurs layers. Les entrées de ce réseau était du type (827,4096,12) où 827 représente le nombre de patients, 4096 représente le nombre de données reçu de l’ECG et 12 représente le nombre de signaux.
Autrement dit, chaque patient nous fournis 12 signaux ECG avec une moyenne de 4096 données par signal. La matrice est de la forme suivante :

![alt tag](https://user-images.githubusercontent.com/70271267/91366796-01267700-e805-11ea-9597-ee3eb4401093.png)

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


#### Resume:
Dans ce Notebook, J’ai essayé de comprendre le fonctionnement de la bibliothèque Ethik qui a pour but d'expliquer les règles de décision entraînées et pour s'assurer qu'elles sont équitables. Elle a été créée par des enseignants et des chercheurs de l’université Paul sabatier-Toulouse.

Ethik fonctionne très bien et mène à des bonnes explications pour deux types d’entrées : les tableaux et les images. Cependant le jeu de données sur lequel je travaillais avait des signaux comme entrée et non pas des tableaux et des images. Mon travail était d’adapter Ethik et voir si on obtenait des résultats et des explications satisfaisantes si notre modèle avait des signaux comme entrée. Pour cela j’ai essayé dans un premier temps de passer d’un signal 3d à un signal 2d et d’appliquer le package sur ce nouveau signal, ensuite j’ai essayé de transformer le signal en une image, et finalement j’ai aussi essayé de projeter mon signal dans une base d’ondelettes.

Mon travail se résume alors à :

* Pouvoir jouer avec les hypothèses et les analyses, et voir les conséquences de nouvelles hypothèses.
	
* Avoir la possibilité d’interagir avec des modèles et des simulations qui rendent plus concrètes les idées abstraites présentées, et se faire ainsi une intuition de leur fonctionnement.
	
* Permettre de prendre connaissance du package, et de vérifier les affirmations des auteurs.
	
#### Package:
 
Vous pouvez trouvez toutes les informations nécessaires sur ce package ainsi que des exemeple en cliquant sur le lien suivant : https://github.com/XAI-ANITI/ethik

#### Requirements and installation:

Ce code a été testé sur Python 3.6 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ethik.

```bash
pip install ethik
```

#### Contributing:

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
	 



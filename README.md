## Système biométrique basé sur la paume de la main
Offre toutes les fonctionnalités nécessaires au développement d'un système d'identification d'une personne basé sur une photo de la paume de sa main. Les étapes couvertes sont : le prétraitement des images, l'extraction des régions d'intérêts et des descripteurs, et finalement la création d'un modèle d’authentification « One Class ». Le développement a été réalisé avec l’aide de la banque d’images « CASIA Palmprint Image Database » qui contient 5 502 images de paumes de la main, provenant de 312 sujets différents.

## Motivation
Projet développé dans le cadre du cours Projet en Informatique I - INF34515, à l’Université du Québec à Rimouski, sous la supervision de Yacine Yaddaden. Le but de ce projet est de me familiariser avec le traitement d’images, l’apprentissage automatique (machine learning), ainsi que de m’initier au monde de la recherche dans un contexte académique.

## Technologies
- [Python 3]( https://www.python.org/downloads/)
- [Scikit-Image]( https://scikit-image.org/)
- [Scikit-Learn]( https://scikit-learn.org/stable/)

## Installation
1.	Cloner le repository : https://github.com/jayParent/palm_biometrics.git
2.	Installer les modules nécessaires : pip install -r requirements.txt

## Utilisation
Pour obtenir de l’aide : py run.py -h
1.	Tester le programme avec les images données en exemple.
- Créer les dossiers nécessaires, extraire la région d’intérêt, appliquer le HOG et sauvegarder ces données en format binaire : py run.py -t -b palms_data
- Filtrer les regions d’intérêts, appliquer le PCA, créer les classifieurs et exécuter un test sur chaque sujet en combinant ses images à celles d’un autre sujet au hasard, pour ensuite tenter de déterminer si chaque image appartient au sujet, ou est un intrus : py run.py -t
2.	Tester le programme avec ses propres images.
- py run.py -b &lt;fichier&gt;
- py run.py -p &lt;fichier&gt;
## Crédits
L’extraction de la région d’intérêt est performée par le script retrouvé ici :
https://github.com/yyaddaden/PROIE

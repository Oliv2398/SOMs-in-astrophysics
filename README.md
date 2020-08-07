# Etude des différences entre les catalogues COSMOS et True Universe à l'aide des Self Organizing Maps



Rapport de stage de Master Physique 1 ainsi que tous les codes et données ayant servi à son élaboration.



&nbsp;

## Installation

Afin de pouvoir lire et exécuter le rapport, suivre les indications suivantes : 

```python
git clone https://github.com/Oliv2398/SOMs-in-astrophysics
cd SOMs-in-astrophysics

 # Création de l'environnement virtuel et installation des dépendances (à faire une seule fois)
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt

# Activation de l'environnement virtuel (il doit apparaitre à gauche du prompt)
source venv/bin/activate
# Lancement du notebook
jupyter notebook rapport_notebook.ipynb
```



&nbsp;

## Aperçu des résultats présents dans le rapport

Première approche avec un jeu de données de couleurs, random et uniforme :

SOM, distance map, ainsi que les poids de chaque couleur, pour un topologie rectangulaire :

![SOMs rectangular](figures/SOM_rect_norm_uniform.png)



Hitmap (gauche), activation de la carte rectangulaire en réponse à un jeu de données bleu, à droite on a remis la carte à l'issue de l'entrainement :

![SOMs rectangular](figures/hitmap.png)







---

Etude des catalogues True Universe et COSMOS

Evolution croisée de l'indice de sersic et de l'ellipticité pour le catalogue COSMOS

![COSMOS catalog](figures/GxCOSMOS.png)





SOM entrainé avec True Universe :

![SOM TU CS](figures/SOM_CS_TU.png)



Hitmap correspondante avec les données de COSMOS :

![heatmap COSMOS](figures/hitmap_CS_TU.png)
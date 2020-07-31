# Etude des catalogues COSMOS et True Universe à l'aide des Self Organizing Maps



Utilisation du package [minisom](https://github.com/JustGlowing/minisom) 



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
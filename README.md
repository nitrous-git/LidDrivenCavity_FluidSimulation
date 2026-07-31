## Description du projet

Ce dépôt contient le projet final de **Modélisation numérique en physique**. Il est dédié à la simulation et à l'analyse de l'écoulement d'un fluide dans une cavité entraînée par une paroi mobile (*Driven Cavity Flow*).

L'enjeu principal est de comparer deux approches numériques pour résoudre les équations de la dynamique des fluides :
* **Méthode SPH (Smoothed Particle Hydrodynamics) :** Approche lagrangienne sans maillage basée sur des particules.
* **Discrétisation sur grille :** Approche structurée pour le suivi du fluide eulerienne.

Pour pallier la lourdeur des calculs (notamment pour l'interaction des particules SPH), l'ensemble du code est optimisé en **Python pur** et accéléré de manière critique avec **Numba (JIT)**, permettant d'atteindre des performances proches du C/C++.

### Fonctionnalités & Analyses
* **Physique des fluides :** Résolution des équations de Navier-Stokes et gestion des conditions aux limites (paroi mobile).
* **Étude comparative :** Analyse quantitative des deux méthodes (temps de calcul, précision, conservation de la masse).
* **Visualisation :** Génération de graphiques et d'animations des champs de vitesse, de pression et des vortex.

### Stack Technique
* **Langage :** Python 3
* **Performance :** Numba (Just-In-Time compiler)
* **Calcul & Données :** NumPy
* **Visualisation :** Matplotlib

  
### Application 3D

Pour une application pratique en trois dimensions, consultez la simulation interactive développée avec Three.js :

- Simulation en ligne : [WaterSim Three.js](https://nitrous-git.github.io/WaterSim_ThreeJs/)
- Dépôt GitHub : [nitrous-git/WaterSim_ThreeJs](https://github.com/nitrous-git/WaterSim_ThreeJs)

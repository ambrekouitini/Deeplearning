# Reconnaissance de Chiffres MNIST

Ce projet implémente un système de reconnaissance de chiffres manuscrits utilisant un réseau de neurones convolutif. L'utilisateur peut dessiner un chiffre sur une interface web et obtenir une prédiction en temps réel.

Tester le site: https://deeplearning-tau.vercel.app/
## Description

Le modèle a été entraîné sur le dataset MNIST et atteint une précision de 99% sur les données de test. L'interface web moderne permet de dessiner directement dans le navigateur et d'obtenir des prédictions instantanées grâce à ONNX.js.

## Structure du projet

Le projet est organisé en plusieurs fichiers principaux. Le fichier temp.ipynb contient l'entraînement du modèle CNN. Le dossier mnist-web contient l'application web complète. Le script convert_to_onnx.py permet de convertir le modèle PyTorch au format ONNX pour une utilisation web.

## Installation

Commencez par cloner le repository sur votre machine locale:
```bash
git clone https://github.com/ambrekouitini/Deeplearning.git
cd Deeplearning
```
Créez ensuite un environnement virtuel Python pour isoler les dépendances:
```bash
python -m venv venv
source venv/bin/activate
```
Installez les dépendances nécessaires pour faire fonctionner le projet:
```bash
pip install torch torchvision onnx jupyter
```
## Utilisation

Pour entraîner le modèle, ouvrez le notebook Jupyter et exécutez toutes les cellules:
```bash
jupyter notebook temp.ipynb
```
Une fois le modèle entraîné, vous pouvez lancer l'interface web. Naviguez vers le dossier de l'application web:
```bash
cd mnist-web
```
Démarrez un serveur HTTP local pour servir les fichiers:
```bash
python -m http.server 8000
```
Ouvrez ensuite votre navigateur et allez à l'adresse http://localhost:8000 pour utiliser l'application.

## Technologies utilisées

Le projet utilise PyTorch pour l'entraînement du réseau de neurones. La conversion vers ONNX permet l'exécution dans le navigateur. L'interface utilisateur est développée en HTML, CSS et JavaScript vanilla. Le dataset MNIST fournit 70000 images de chiffres manuscrits pour l'entraînement.

## Performances

Le modèle atteint une précision de 99% sur les données de test MNIST. Le temps d'inférence est inférieur à 100 millisecondes. La taille du modèle converti est d'environ 500 kilobytes.

Auteur: Ambre Kouitini

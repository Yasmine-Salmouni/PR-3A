# Définition du modèle

## Contexte

La fabrication d’un pneu repose sur la superposition de plusieurs nappes de gomme. Cependant, les extrémités de ces nappes ne peuvent pas toujours être utilisées, ce qui génère des chutes de matière. Les jeter représenterait à la fois un coût économique et un impact environnemental important. Afin de limiter ce gaspillage, il est possible de recycler ces chutes en les réincorporant, en faibles quantités, dans la production de nouvelles nappes.

# But du modèle

Générer un plan de ré-incorporation optimal des chutes, en cherchant
le volume de chutes maximal qu’il est possible de recycler dans le plan de production fourni.

Un **plan de production** est le calendrier prévisionnel qui dicte ce qui doit être fabriqué, à quel moment et en quelle quantité.

# Définition du modèle

## Ensembles:

- Soit C l’ensemble des chutes.
    
    On peut définir un élément $c \in C$ par les caractéristiques techniques suivantes qui le distinguent dans le modèle :
    
    - **Un volume (masse) :** Chaque élément $c$ possède un poids spécifique exprimé en kilogrammes ($kg$), noté comme le stock_proj_chute.
     
    - **Un état de "vieillissement" :** Une chute n'est pas qu'un simple morceau de gomme ; elle est définie par une **DMC** (Date Minimale de Consommation) après traitement et une **DLC** (Date Limite de Consommation) avant qu'elle ne soit plus recyclable.
    
    -**Une fenêtre d'utilisation** $dc$ **:** Mathématiquement, à chaque chute $c$ est associé un couple (date min ; date max) qui restreint le moment où elle peut être ré-incorporée dans une nouvelle production. dc = [DMC, DLC]
    
- Soit DC l’ensemble des fenêtres d’utilisation des chutes.
- Soit P l’ensemble des productions.
    
    Chaque production $p$ est une opportunité de recyclage. C'est l'endroit où l'on va "cacher" les chutes de gomme en les mélangeant à la matière neuve.
    
- Soit DP l’ensemble des dates de production.
    
    Une date $dp \in DP$ représente généralement un jour spécifique ou un créneau de travail durant lequel on fabrique un lot de pneus.
    
    C'est la donnée qui permet de faire le lien entre les **chutes** (qui ont une date de péremption) et les **productions** (qui ont une date de fabrication).
    

## Indices:

- Soit c ∈ C une chute.
- Soit dc ∈ DC une fenˆetre d’utilisation d’une chute.
- Soit p ∈ P une production.
- Soit dp ∈ DP une date de production.

## Données d’entrée:

## Données dynamques:

Mises à jour quotidiennement

- $stock\_proj\_chute_{c,dc}$ : Le stock projeté des chutes (en kg).
    
    Représente le volume de la chute c utilisable sur la fenêtre de temps dc.
    
    Il y’a deux types de chutes. 
    
    Pour le premier jour de l’horizon on récupère le stock de chutes existant, pour le reste de l’horizon on estime le volume de chutes qui sera généré en fonction du plan de production.
    
    On rectifie quotidiennement les projections.
    
- $reincorpo\_maxi_{p,dp,c}$  représente le volume maximal de ré-incorporation d’une chute c dans la production p à la date dp.

## Constante:

- $seuil\_reincorpo\_mini$  représente la masse minimale de chute nécessaire pour engager une campagne de ré-incorporation dans un produit.

## Variables:

- $R_{c,dc,p,dp}$  : représente le volume de ré-incorporation dans la production p réalisée à la date dp de la chute c utilisable dans la fenêtre dc.
- $O_{p,dp,c}$  : est un booléen qui indique si on a choisi de ré-incorporer une chute c dans la production p réalisée à la date dp.

## Contraintes:

### Respect de la fenêtre d’utilisation de la chute

On ne crée que les variables $R_{c,dc,p,dp}$ pour lesquelles dp se trouve dans la fenêtre d’utilisation dc.

### Respect des volumes de ré-incorporation

- **Disponibilité du stock** : Le volume total de chutes ré-incorporées pour une chute $c$ donnée sur sa fenêtre $dc$ ne peut pas dépasser le volume de chutes générées ou en stock
On ne peut pas ré-incorporer un volume de chutes supérieur au volume des chutes générées.
    
    $\sum_{p\in P}\sum_{dp\in DP}R_{c,dc,p,dp}\le stock\_proj\_chute_{c,dc} \quad \forall c\in C, \forall dc\in DC$
    
- **Capacité d'absorption du produit** : Pour chaque chute, produit et date, le volume ré-incorporé doit être inférieur ou égal à la capacité maximale du produit ($reincorpo\_maxi_{p,dp,c}$), multipliée par la variable d'activation $O_{p,dp,c}$.
$\sum_{dc\in DC}R_{c,dc,p,dp} \le reincorpo\_maxi_{p,dp,c} \times O_{p,dp,c} \quad \forall p\in P, \forall dp\in DP, \forall c\in C$

- **Seuil minimal d'engagement** : Si on choisit de recycler une chute dans un produit, le volume doit atteindre au moins le seuil minimal ($seuil\_reincorpo\_mini$) pour justifier l'opération.
    
    $seuil\_reincorpo\_mini \times O_{p,dp,c} \le \sum_{dc\in DC}R_{c,dc,p,dp} \quad \forall p\in P, \forall dp\in DP, \forall c\in C$
    

Des deux der,ières contraintes on a la ontrainte: si O est nul alors La somme des R sur dc est nulle ce qui est logique

### Respect de la stratégie de ré-incorporation

On ne peut intégrer qu’une seule chute dans un produit sur une date de production.

$\sum_{c\in C}O_{p,dp,c}\le1 \quad \forall p\in P, \forall dp\in DP$

## Domaines des variables

- $R_{c,dc,p,dp} \in [0, \min(stock\_proj\_chute_{c,dc}, reincorpo\_maxi{p,dp,c})] \quad \forall c \in C, \forall dc \in DC, \forall p \in P, \forall dp \in DP$
- $O_{p,dp,c} \in \{0, 1\} \quad \forall p \in P, \forall dp \in DP, \forall c \in C$

## Fonction objectif:

- Maximiser le volume total de ré-incorporation.
- Maximiser le taux d’utilisation des chutes par produit et date de production (donc minimiser le nombre de ré-incorporations différentes) pour éviter l’éparpillement qui représente un cauchemard logistique pour l’entreprise.

$\text{Maximiser} \sum_{c\in C} \sum_{dc\in DC} \sum_{p\in P} \sum_{dp\in DP} (R_{c,dc,p,dp} - O_{p,dp,c})$
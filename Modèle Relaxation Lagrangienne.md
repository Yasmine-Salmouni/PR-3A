# Relaxation Lagrangienne

L'objectif est de décomposer le problème global en ignorant temporairement la contrainte de stock, tout en pénalisant son dépassement.

## **1. Identification de la contrainte "bloquante"**

La contrainte qui empêche de résoudre le problème produit par produit est la **Contrainte (1)** : le respect des volumes de stocks projetés.

• **Équation :** $\sum_{p\in P}\sum_{dp\in DP}R_{c,dc,p,dp}\le stock\_proj\_chute_{c,dc}$.

• On introduit un multiplicateur de Lagrange $\lambda_{c,dc} \ge 0$ pour chaque chute $c$ et fenêtre $dc$.

## 2. Construction de la Fonction Duale (Le Lagrangien)

On part de votre objectif initial qui est de maximiser le volume de ré-incorporation tout en minimisant le nombre de réglages (donc maximiser $R - O$).

On retire la contrainte de stock: $\sum_{p \in P} \sum_{dp \in DP} R_{c,dc,p,dp} \le stock\_proj\_chute_{c,dc}$

Et on l'insère dans l'objectif avec un **multiplicateur de Lagrange** $\lambda_{c,dc} \ge 0$. 

 La fonction devient :

$L = \underbrace{\sum_{c, dc, p, dp} (R_{c,dc,p,dp} - O_{p,dp,c})}_{\text{Objectif initial}} + \underbrace{\sum_{c, dc} \lambda_{c,dc} \left( stock\_proj\_chute_{c,dc} - \sum_{p, dp} R_{c,dc,p,dp} \right)}_{\text{Contrainte de stock "relaxée"}}$

$L(R, O, \lambda) = \sum_{c} \sum_{dc} (\sum_{p} \sum_{dp} (R_{c,dc,p,dp} - O_{p,dp,c}) + \lambda_{c,dc} \left( stock\_proj\_chute_{c,dc} - \sum_{p} \sum_{dp} R_{c,dc,p,dp} \right))$

---

Pour que la fonction $L$ (le Lagrangien) donne une solution correcte, il faut connaître les **valeurs exactes** des multiplicateurs $\lambda_{c,dc}$.
• Ces multiplicateurs représentent le "juste prix" de la rareté de chaque chute.
• Si vous fixez $\lambda$ au hasard (par exemple à 0 ou à une valeur fixe), l'optimisation va soit ignorer totalement le stock (si $\lambda$ est trop bas), soit ne rien recycler du tout (si $\lambda$ est trop haut).

• **L'étape A et l'étape B**, intégrées dans un algorithme, servent justement à "chercher" les bonnes valeurs de $\lambda$ par tâtonnements mathématiques.

**Étape A : Résolution du sous-problème (Borne Duale)**
• On utilise les $\lambda^k$ actuels pour calculer les scores de chaque chute par production : $Score = (1 - \lambda^k) \cdot R - 1$.
• On choisit la meilleure option pour chaque $(p, dp)$.
• On calcule la **Borne Duale** $Z_{dual}$. Cette borne va globalement **diminuer** au fil des itérations car les pénalités augmentent.

**Étape B : Heuristique de réparation (Borne Primale)**
• On prend les décisions de l'Étape A et on les ajuste pour ne jamais dépasser le `stock_proj_chute` réel.
• On s'assure que chaque volume respecte le `seuil_reincorpo_mini`.
• On calcule la **Borne Primale** $Z_{primal}$. Cette borne va globalement **augmenter** car on trouve de meilleures façons de répartir le stock limité.

**Étape C : Calcul du "Gap"**
• On calcule l'écart : $Gap = Z_{dual} - Z_{primal}$.
• Si cet écart est très petit (inférieur à un seuil $\epsilon$), on s'arrête : la solution est quasi-optimale.

**SINON** 

**Étape D : Mise à jour des pénalités (Le réglage)**
C'est l'étape cruciale. On ajuste les $\lambda$ pour l'itération $k+1$ en observant les violations de contraintes de l'Étape A :

pour un c,dc

• **Si une chute a été surconsommée** ($\sum_{p\in P}\sum_{dp\in DP}R_{c,dc,p,dp} > stock\_proj\_chute_{c,dc}$) : On **augmente** $\lambda_{c,dc}$. La chute devient plus "chère", ce qui forcera l'Étape A à moins l'utiliser à la prochaine itération.
• **Si une chute a été sous-consommée** ($\sum_{p\in P}\sum_{dp\in DP}R_{c,dc,p,dp} < stock\_proj\_chute_{c,dc}$) : On **diminue** $\lambda_{c,dc}$. La chute devient "moins chère", encourageant son utilisation.

---

## Étape A : Résolution des sous-problèmes (Borne Duale / Supérieure)

Grâce à la relaxation de la contrainte de stock, le problème global se fragmente. Pour chaque production $p$ à une date $dp$, vous devez prendre une décision indépendante des autres jours de production.

La contrainte de respect de la stratégie de ré-incorporation implique que:
Un sous-problème consiste à répondre à la question suivante pour chaque couple $(p, dp)$ :**"Quelle est la meilleure chute $c$ à ré-incorporer aujourd'hui pour maximiser mon gain net ?"**

### Le calcul du gain net par chute

Pour chaque chute $c$ potentiellement utilisable (c'est-à-dire si la date $dp$ est bien dans la fenêtre $dc$ ), vous calculez un score de rentabilité.

Le gain d'une chute $c$ pour une production $(p, dp)$ se calcule ainsi :
• **Revenu brut :** On prend le volume maximal possible, soit $reincorpo\_maxi_{p,dp,c}$.

• **Pénalité de stock :** On multiplie ce volume par le multiplicateur de Lagrange $\lambda_{c,dc}$ actuel. Ce $\lambda$ représente la "valeur" ou la rareté de la chute.

• **Coût d'activation :** On soustrait 1 (qui correspond au poids de la variable binaire $O_{p,dp,c}$ dans l'objectif).

**La formule du score local :**

$Score_{p,dp,c,dc} = [(1 - \lambda_{c,dc}) \times reincorpo\_maxi_{p,dp,c}] - 1​$

### La règle de sélection (L'Optimisation)

Pour un $(p, dp)$ donné, vous comparez tous les scores obtenus pour les différentes chutes $c \in C$:

**Identifier le maximum :** Vous cherchez la chute qui a le $Score(c)$ le plus élevé.
**Vérifier la rentabilité :** **Si le meilleur $Score(c) > 0$ :** Vous décidez de ré-incorporer cette chute. On pose $O_{p,dp,c} = 1$ et $R_{c,dc,p,dp} = reincorpo\_maxi_{p,dp,c}$.

• **Si le meilleur $Score(c) \le 0$ :** Aucune chute n'est assez rentable pour couvrir le coût d'activation et la pénalité de stock. On pose $O = 0$ et $R = 0$ pour toutes les chutes sur ce créneau

### Vérification des contraintes locales

- **Seuil minimal :** Si le volume $reincorpo\_maxi_{p,dp,c}$ est inférieur au $seuil\_reincorpo\_mini$, le score est automatiquement considéré comme nul ou négatif car la production est impossible
- **Fenêtre d'utilisation :** Vous ne calculez le score que si $dp \in dc$.
- **Unicité :** En ne choisissant que le *meilleur* score, vous respectez naturellement la contrainte de ne mettre qu'une seule chute par produit.

### Résultat de l'Étape A : La Borne Duale

**La Formule de la Borne Duale $Z_{dual}$**

$Z_{dual}(\lambda) = \sum_{p \in P} \sum_{dp \in DP} \max \left( 0, \max_{\substack{c \in C, dc \in DC \\ dp \in dc}} \{ Score_{p,dp,c,dc} \} \right) + \sum_{c \in C} \sum_{dc \in DC} (\lambda_{c,dc} \times stock\_proj\_chute_{c,dc})$

Le chiffre final est votre **Borne Duale**. C'est une borne "optimiste" car, à ce stade, vous avez peut-être utilisé $1500$ kg d'une chute alors que vous n'en aviez que $1000$ kg en stock. C'est l'**Étape B** qui viendra corriger cela.
Chaque sous-problème voit seulement son propre gain : $(1 - \lambda) \times R - 1$. Le sous-problème ne sait pas ce que les autres sous-problèmes ont décidé.

## Étape B : Phase de réparation

Comme nous l'avons vu, l'Étape A peut aboutir à une surconsommation de chutes car elle traite les productions $(p, dp)$ de manière isolée. L'Étape B intervient pour :
• **Réinstaurer la barrière physique** du stock.
• **Transformer** les décisions "optimistes" en un plan de production valide.

### Le fonctionnement : L'Heuristique de Réparation

Puisque le problème est complexe, on utilise généralement une approche gloutonne pour transformer la solution de l'étape A en solution réalisable :

**B1 : Faire la file d'attente (Le Tri)**

À l'étape A, plusieurs productions ont demandé la même chute $c$ parce qu'elles en avaient besoin au même moment. Si on additionne toutes ces demandes, on dépasse souvent le stock disponible.
B1 sert à décider qui passe en premier.

On fixe c,dc
• Tu listes toutes les productions qui ont dit "Je veux la chute $c$" (celles où $O_{p,dp,c} = 1$).
• Tu les classes, par exemple, de la plus grosse demande à la plus petite.
• **Pourquoi ?** Parce que si tu n'as pas assez de gomme pour tout le monde, tu préfères servir en priorité ceux qui recyclent de gros volumes pour atteindre ton objectif de maximisation.

**B2 : Distribuer la gomme (L'Allocation)**
Maintenant que tu as ta file d'attente, tu prends ton carnet de stock réel (`stock_proj_chute`) et tu distribues la matière.
**Pour chaque production dans la file d'attente :**
1. **Regarder le stock :** Est-ce qu'il reste assez de chute $c$ dans le réservoir ? 
2. **Servir :**
    ◦ S'il reste assez de stock : Tu donnes le volume demandé ($R_{c,dc,p,dp}$).
    ◦ S'il ne reste qu'un peu de stock : Tu donnes ce qu'il reste (le reliquat).
3. **Vérifier le seuil :** Si ce que tu donnes est plus petit que le `seuil_reincorpo_mini`, alors tu annules tout pour cette production (on ne lance pas une machine pour seulement 2 kg de gomme).
4. **Mettre à jour :** Tu soustrais ce que tu as donné de ton stock et tu passes à la personne suivante dans la file d'attente.

**Étape B.3 : Respecter l'unicité par produit**

On s'assure qu'une production donnée $(p, dp)$ n'utilise bien qu'**une seule chute** au maximum, conformément à la contrainte (4) du document.

### 3. Résultat : La Borne Primale ($Z_{primal}$)

Une fois ce tri et ces coupes effectués, vous obtenez un ensemble de variables qui respectent **toutes** les contraintes:
• Respect de la fenêtre d'utilisation $dc$.
• Respect strict du volume de stock disponible.
• Respect du seuil minimal de ré-incorporation.
• Respect de l'unicité de la chute par production.

Vous calculez alors la valeur de votre fonction objectif avec ces chiffres "réels" :

**$Z_{primal} = \sum \sum \sum \sum (R_{c,dc,p,dp} - O_{p,dp,c})$**

---

## Etape D:

A la première itération les lambdas sont initialisés à 0.
L'étape **D** est le cerveau de l'algorithme : c'est elle qui apprend de ses erreurs pour guider le modèle vers la solution optimale. 

pour chaque itération

**1. Mesurer l'erreur de stock (Le Sous-Gradient)**
Pour chaque couple chute/fenêtre $(c, dc)$, on calcule l'écart entre ce que l'Étape A a "rêvé" de consommer et ce qui est réellement disponible en stock. Cet écart est appelé le **sous-gradient** ($g_{c,dc}$) :
$g_{c,dc}^{(k)} = \left( \sum_{p \in P} \sum_{dp \in DP} R_{c,dc,p,dp}^{(k)} \right) - stock\_proj\_chute_{c,dc}$

**Si $g_{c,dc} > 0$** : Il y a surconsommation (violation de la contrainte). La pénalité doit augmenter.
**Si $g_{c,dc} < 0$** : Il y a sous-consommation (le stock n'est pas pleinement utilisé). La pénalité doit diminuer.

**2. Calculer le "Pas" de réglage ($\theta$)**
On ne change pas les pénalités de manière brute. On utilise un multiplicateur appelé le **pas d'apprentissage** ($\theta^{(k)}$). Il définit la force avec laquelle on réagit à l'erreur de stock.
Une méthode classique (méthode d'Held-Karp) consiste à calculer ce pas ainsi :                             $\theta^{(k)} = \alpha \cdot \frac{Z_{dual}^{(k)} - Z_{best\_primal}}{\sum (g_{c,dc}^{(k)})^2}$                                                                                                                                          

- **$Z_{dual} - Z_{best\_primal}$** : Représente l'écart (le Gap) entre votre borne supérieure et votre meilleure borne inférieure trouvée à l'étape B. (pourquoi best?)
- **$\alpha$** : Un paramètre de vitesse (commence souvent à 2.0). Si la borne duale ne s'améliore pas après quelques tours, on divise $\alpha$ par 2 pour affiner le réglage.

**Conseil type** : Divisez $\alpha$ par deux si la Borne Duale ne s'est pas améliorée (n'a pas diminué) pendant **10 itérations consécutives**.

**3. Appliquer la mise à jour**
On met à jour la pénalité pour l'itération suivante ($k+1$) avec la règle suivante:

$\lambda_{c,dc}^{(k+1)} = \max\left( 0, \lambda_{c,dc}^{(k)} + \theta^{(k)} \cdot g_{c,dc}^{(k)} \right)$
• **Le $\max(0, \dots)$** : Est crucial car un multiplicateur de Lagrange ne peut jamais être négatif. Si le calcul donne un chiffre négatif, on le remet à 0 (la ressource est considérée comme gratuite car abondante).
• **L'effet de levier** : Plus la violation de stock est grande (gros surplus de consommation), plus $\lambda$ augmentera fortement pour "calmer" les ardeurs de l'étape A au tour suivant.
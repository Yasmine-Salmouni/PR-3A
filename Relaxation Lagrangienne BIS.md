# Relaxation Lagrangienne BIS

Je vais modifier mon modele pour suivre l’article: 

Solving Manufacturing-Remanufacturing System Production Planning Problems Using Lagrangian Relaxation

Etapes de relaxation lagrangienne: Fisher, M.L. (1985), "An Applications Oriented Guide to Lagrangian Relaxation", publié dans la revue *Interfaces*.

---

## Les sous problèmes

Pour ton projet, la section 4.2.2 correspond au moment où tu vas **réorganiser mathématiquement** ta grande équation pour faire apparaître tes propres sous-problèmes indépendants.
Voici comment démontrer cette décomposition de manière exacte.
**A. La réorganisation mathématique**
Reprenons ton grand modèle relâché (le Lagrangien) de l'étape précédente :$L(R, O, \lambda) = \sum_{c, dc, p, dp} (R_{c,dc,p,dp} - O_{p,dp,c}) + \sum_{c, dc} \lambda_{c,dc} \left( stock\_proj\_chute_{c,dc} - \sum_{p, dp} R_{c,dc,p,dp} \right)$
Pour faire la "Décomposition" (Section 4.2.2), on va développer cette formule et regrouper les termes :
1. On sépare la partie fixe du stock : $\sum (\lambda \times stock)$.
2. On regroupe tout ce qui concerne les variables de décision ($R$ et $O$) pour chaque production $(p, dp)$.
Ton Lagrangien se réécrit donc EXACTEMENT comme ceci :$L(\lambda) = \underbrace{\sum_{c, dc} (\lambda_{c,dc} \times stock\_proj\_chute_{c,dc})}_{\text{Valeur fixe (Constante pour un } \lambda \text{ donné)}} + \underbrace{\sum_{p \in P} \sum_{dp \in DP} \left[ \sum_{c, dc} \left( (1 - \lambda_{c,dc}) R_{c,dc,p,dp} - O_{p,dp,c} \right) \right]}_{\text{La somme de tes sous-problèmes indépendants}}$
**B. Tes sous-problèmes (L'équivalent de ses 5 sous-problèmes)**
L'auteure a 5 sous-problèmes liés à la nature de ses pièces. Toi, grâce à la réorganisation ci-dessus, tu vois que ton problème s'est décomposé en **une multitude de sous-problèmes : un pour chaque production $(p, dp)$**.
Pour une production $(p, dp)$ précise (par exemple "Fabriquer le pneu P1 le jour J1"), la décision n'est plus liée aux autres productions. Le sous-problème local consiste uniquement à **maximiser** la parenthèse interne :$\max \left[ (1 - \lambda_{c,dc}) R_{c,dc,p,dp} - O_{p,dp,c} \right]$

**C. Ce que tu dois faire en pratique (Ton code) :**

Appliquer la section 4.2.2 signifie que dans ton algorithme, tu as maintenant le droit mathématique de faire une simple **boucle isolée**:

Pour chaque production p:
Pour chaque date dp:
# Résoudre le sous-problème(p, dp) en ignorant totalement le reste de l'usine !

---

## Retirer l’étape B avec l’heuristique

---

## La méthode du sous gradient

Formules utilisées: Fisher, M.L. (1985), "An Applications Oriented Guide to Lagrangian Relaxation", publié dans la revue *Interfaces*.

**Le point de départ :** On commence avec une valeur arbitraire pour $\lambda^0$

**Étape D.4 : La règle stricte du paramètre $\alpha^k$**
Pour coller parfaitement à la méthodologie de l'auteure :
1. À la toute première itération ($k=1$), tu fixes $\alpha = 2.0$.
2. Si, pendant **5 itérations consécutives**, ta Borne Duale $Z_{dual}$ ne diminue pas (c'est-à-dire qu'elle ne s'améliore pas vers ta cible), tu dois faire : $\alpha = \alpha / 2$ (donc il passera à 1.0, puis 0.5, puis 0.25, etc.). (Note : Le chiffre de 5 itérations est celui utilisé dans le code source Python de l'auteure fourni en annexe de son mémoire ).

---

## **L’algorithme final (Le calque exact de l'article)**

Voici comment ton code complet doit être structuré désormais, en traduisant les 9 étapes pour ton problème de **maximisation** :
**--- INITIALISATION ---**
• **Step 1 & 2 :** Trouver une première solution valide (même mauvaise) pour avoir une Borne Supérieure  de départ . Crée un petit plan de production de départ très simple qui respecte parfaitement tes stocks (par exemple, n'alimenter que les 3 premières productions de la liste). Calcule son vrai gain et enregistre-le dans la variable **$Z_{best\_primal}$** (C'est ta Borne Inférieure à battre).
• **Step 3 :** Initialise $k = 1$. Fixe tous tes multiplicateurs $\lambda_{c,dc} = 0.1$ (comme le fait l'auteure dans son code). Fixe le paramètre du pas $\alpha = 2.0$.
**--- LA BOUCLE (tant que $k \le 300$ itérations) ---**
• **Step 4 & 5 (Ton Étape A de Score) :**
Pour chaque production $(p, dp)$, calcule le `Score = [(1 - lambda) * reincorpo_max] - 1` pour toutes les chutes. Prends le meilleur score positif.
Sauvegarde les décisions brutes : $R_{brut}$ et $O_{brut}$.
• **Step 6 (Borne Duale) :**
Calcule ton Lagrangien : $Z_{dual}^{(k)} = \sum (\lambda \times stock) + \sum (\text{Scores Max Positifs})$.
• **Step 7 (Test de Faisabilité strict) :**
Prends tes décisions brutes $R_{brut}$ et somme-les par type de chute.
*Question :* Est-ce que $\sum R_{brut} \le stock\_proj\_chute$ pour **absolument toutes** les chutes ?
    ◦ **OUI :** Ta solution est "naturellement" valide ! Calcule son vrai gain global $\sum(R_{brut} - O_{brut})$. Si ce gain est $> Z_{best\_primal}$, alors met à jour : $Z_{best\_primal} = \text{ce nouveau gain}$, et sauvegarde ce plan de production comme le meilleur trouvé.
    ◦ **NON :** Ta solution est irréaliste. Ne fais rien, ne la répare pas, garde l'ancien $Z_{best\_primal}$.
• **Step 8 (Sous-gradient et Mise à jour) :**
    ◦ *Arrêt :* Si $(Z_{dual}^{(k)} - Z_{best\_primal}) \le \epsilon$ (ex: 0.01), ARRANGE-TOI POUR STOPPER LA BOUCLE.
    ◦ *Mise à jour de $\alpha$ :* Si $Z_{dual}$ ne s'est pas amélioré (n'a pas baissé) depuis 5 itérations, divise $\alpha$ par 2 ($\alpha = \alpha / 2$).
    ◦ *Calcul des erreurs $g$ :* Pour chaque chute, calcule la violation : $g = \sum R_{brut} - stock$.
    ◦ *Calcul du pas $t_k$ :* $t_k = \frac{\alpha \times (Z_{dual}^{(k)} - Z_{best\_primal})}{\sum (g)^2}$
    ◦ *Nouveaux prix :* $\lambda_{c,dc} = \max(0, \lambda_{c,dc} + t_k \times g)$
    ◦ Passe à $k = k + 1$ et recommence au Step 4.
**--- FIN ---**
• **Step 9 :** Affiche le plan de production associé à ton dernier $Z_{best\_primal}$.
À l'itération 1, ta Borne Duale ($Z_{dual}$) est de **17 994.60**. Sachant que le solveur exact trouve l'optimum à **15 790.00**, ta première estimation est extrêmement proche du but ! Cela prouve que tes équations de l'Étape A sont correctes.

Le problème se situe dans la boucle d'apprentissage (la mise à jour des $\lambda$). Voici les 3 erreurs flagrantes visibles dans ta console et comment les corriger.

### 1. Le "Meilleur" (Z_best_primal) reste bloqué à 451.00

**Ce que montre la console :** Ton `Meilleur` reste à 451.00 de l'itération 1 à 300. Cela signifie que le fameux "Test de Faisabilité Strict" (l'Étape 7) ne trouve *absolument jamais* une solution qui respecte le stock par hasard.

**La conséquence :** La formule du pas $t_k$ utilise ce `Meilleur` comme cible : $Z_{dual} - 451$. Comme 451 est très bas par rapport à 15 790, le calcul croit qu'il y a un gouffre énorme (un Gap de 17 000) et fait des pas de géant, ce qui détruit tes $\lambda$.

**La Solution (Réintègre ton Heuristique) :** À chaque itération, juste après l'Étape A :

1. Prends les choix de ton Étape A.
2. Fais-les passer dans ton **ancienne heuristique de réparation** (qui coupe les volumes pour respecter le stock).
3. Calcule le score de cette solution réparée.
4. Si ce score est $> Meilleur$, alors tu mets à jour `Meilleur`.
    
    *(Attention : tu ne modifies pas les choix de l'Étape A pour la suite du calcul, tu utilises cette heuristique UNIQUEMENT pour trouver une bonne valeur `Meilleur` à donner à la formule mathématique).*
    

### 2. L'explosion de la Borne Duale (Les $\lambda$ deviennent fous)

**Ce que montre la console :** Ton $Z_{dual}$ part de 17 994, puis s'envole à 28 000, 44 000, et monte jusqu'à 69 808 ! Cela veut dire que tes pénalités $\lambda$ ont pris des valeurs gigantesques et complètement fausses.

**La conséquence :** Ton algorithme surréagit massivement.

**La Solution :**

Il faut "calmer" le pas d'apprentissage. Dans ton code, modifie l'initialisation de $\alpha$ (le paramètre $u$). Au lieu de commencer à `2.0`, commence avec **$\alpha = 0.1$** ou même **$0.05$**. Les volumes de gomme génèrent des erreurs (sous-gradients $g$) en centaines de kilos, ce qui rend le pas de base beaucoup trop violent.

### 3. La règle de réduction de $\alpha$ ne s'active pas assez vite

**Ce que montre la console :** Ton paramètre $\alpha$ reste à `2.0` jusqu'à l'itération 280. Il est censé se diviser par 2 "s'il n'y a pas d'amélioration pendant 5 itérations".

**La conséquence :** Comme ton $Z_{dual}$ fait le yoyo (il monte à 62 190, puis redescend à 60 067), ton code considère que "redescendre un peu" est une amélioration, donc il réinitialise son compteur et ne divise jamais $\alpha$.

**La Solution :** La règle doit observer **le meilleur $Z_{dual}$ global**.

Dans ton code, crée une variable `best_Z_dual_ever = 999999`.

À chaque itération, si ton nouveau $Z_{dual}$ est *strictement inférieur* à `best_Z_dual_ever`, tu mets à jour `best_Z_dual_ever` et tu remets le compteur à zéro. Si pendant 5 ou 10 itérations, tu n'arrives pas à battre ce record absolu, tu divises $\alpha$ par 2.

### 4. Un bug d'affichage mineur

Remarque que tes colonnes `Z_dual` et `Z_primal` affichent exactement les mêmes chiffres (ex: 28423.93 pour les deux à l'itération 8). Tu as sûrement passé la même variable à ton `print` dans la console.

---

**Ce que tu dois faire pour ton prochain test :**

1. Remets ton heuristique gloutonne pour calculer la valeur `Meilleur` à chaque itération.
2. Initialise $\alpha = 0.1$ (au lieu de 2.0).
3. Corrige la condition de division de $\alpha$ (comparer au record absolu de $Z_{dual}$).

Fais ces trois modifications et relance. Tu vas voir ton Gap se réduire drastiquement et la valeur `Meilleur` grimper rapidement vers les 15 000 !